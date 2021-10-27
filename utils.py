import pandas
from pathlib import Path
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from grafica.plotly_utils.utils import line as grafica_line
from scipy import interpolate

path_to_base_TI_LGAD = Path('/home/alf/cernbox/projects/4D_sensors/TI-LGAD_FBK_RD50_1')
path_to_measurements_directory = path_to_base_TI_LGAD/Path('measurements_data')

def read_devices_sheet():
	df = pandas.read_excel(
		path_to_base_TI_LGAD/Path('doc/FBK TI-LGAD RD50 1.xlsx'),
		sheet_name = 'devices',
	)
	return df.loc[:, ~df.columns.str.contains('^Unnamed')].set_index('#')

def check_df_is_from_single_1D_scan(df):
	"""If df contains data that looks like that from a single 1D scan of one device, this function does nothing. Otherwise, it will rise ValueError."""
	from measurements_table import retrieve_measurement_type # Importing here to avoid circular import.
	if len(set(df['Measurement name'])) != 1:
		raise ValueError(f'`df` must contain data from a single measurement, but it seems to contain data from the following measurements: {set(df["Measurement name"])}.')
	measurement_name = sorted(set(df['Measurement name']))[0]
	if retrieve_measurement_type(measurement_name) != 'scan 1D':
		raise ValueError(f'`df` must contain data from a "scan 1D" measurement, but the measurement {repr(measurement_name)} is of type {repr(retrieve_measurement_type(measurement_name))}.')

def tag_left_right_pad(data_df):
	"""Given a data_df with data from a single 1D scan of two pads of a device, this function adds a column `Pad` indicating "left" or "right"."""
	check_df_is_from_single_1D_scan(data_df)
	channels = set(data_df['n_channel'])
	if len(channels) != 2:
		raise ValueError(f'`data_df` contains data concerning more than two channels. I can only tag left and right pads for two channels data.')
	left_data = data_df.loc[(data_df['n_position']<data_df['n_position'].mean())]
	right_data = data_df.loc[(data_df['n_position']>data_df['n_position'].mean())]
	for channel in channels:
		if left_data.loc[left_data['n_channel']==channel, 'Collected charge (V s)'].mean(skipna=True) > left_data.loc[~(left_data['n_channel']==channel), 'Collected charge (V s)'].mean(skipna=True):
			mapping = {channel: 'left', list(channels-{channel})[0]: 'right'}
		else:
			mapping = {channel: 'right', list(channels-{channel})[0]: 'left'}
	for n_channel in set(data_df['n_channel']):
		data_df.loc[data_df['n_channel']==n_channel, 'Pad'] = mapping[n_channel]
	return data_df

def read_measured_data_from(measurement_name: str):
	"""Reads the data from a 1D scan and returns a dataframe. The dataframe is returned "intact" in the sense that nothing is done, except that a column is added indicating the measurement name."""
	for scan_script_name in ['linear_scan_many_triggers_per_point','1D_scan', 'scan_1D']:
		try: # First try to read feather as it is much faster.
			df = pandas.read_feather(path_to_base_TI_LGAD/Path('measurements_data')/Path(measurement_name)/Path(scan_script_name)/Path('measured_data.fd'))
			df['Measurement name'] = measurement_name
			return df
		except FileNotFoundError:
			pass
		try:
			df = pandas.read_csv(path_to_base_TI_LGAD/Path('measurements_data')/Path(measurement_name)/Path(scan_script_name)/Path('measured_data.csv'))
			df['Measurement name'] = measurement_name
			return df
		except FileNotFoundError:
			pass
	raise FileNotFoundError(f'Cannot find measured data for measurement {repr(measurement_name)}.')

def line(error_y_mode=None, **kwargs):
	# I moved this function here https://github.com/SengerM/grafica/blob/main/grafica/plotly_utils/utils.py
	return grafica_line(error_y_mode, **kwargs)

def calculate_1D_scan_distance(positions):
	"""positions: List of positions, e.g. [(1, 4, 2), (2, 5, 2), (3, 7, 2), (4, 9, 2)].
	returns: List of distances starting with 0 at the first point and assuming linear interpolation."""
	return [0] + list(np.cumsum((np.diff(positions, axis=0)**2).sum(axis=1)**.5))

def calculate_1D_scan_distance_from_dataframe(df):
	check_df_is_from_single_1D_scan(df)
	x = df.groupby('n_position').mean()[f'x (m)']
	y = df.groupby('n_position').mean()[f'y (m)']
	z = df.groupby('n_position').mean()[f'z (m)']
	distances_df = pandas.DataFrame({'n_position': [i for i in range(len(set(df['n_position'])))], 'Distance (m)': calculate_1D_scan_distance(list(zip(x,y,z)))})
	distances_df.set_index('n_position')
	return distances_df

def calculate_normalized_collected_charge(df, window_size=125e-6, laser_sigma=9e-6):
	"""df must be the dataframe from a single 1D scan."""
	check_df_is_from_single_1D_scan(df)
	
	df['Normalized collected charge'] = df['Collected charge (V s)']
	mean_df = df.groupby(by = ['n_channel','n_pulse','n_position']).mean()
	mean_df = mean_df.reset_index()
	for n_pulse in sorted(set(mean_df['n_pulse'])):
		for n_channel in sorted(set(mean_df['n_channel'])):
			mean_df.loc[(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] = mean_df.loc[(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Collected charge (V s)']
			offset_factor = np.nanmin(mean_df.loc[(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'])
			mean_df.loc[(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] -= offset_factor
			scale_factor = np.nanmax(mean_df.loc[(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'])
			mean_df.loc[(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] /= scale_factor
			# Now I repeat for the df ---
			df.loc[(df['n_pulse']==n_pulse)&(df['n_channel']==n_channel), 'Normalized collected charge'] -= offset_factor
			df.loc[(df['n_pulse']==n_pulse)&(df['n_channel']==n_channel), 'Normalized collected charge'] /= scale_factor
	return df

def calculate_distance_offset_by_linear_interpolation(df):
	"""Given data from a 1D scan from two complete pixels (i.e. scanning from metal→silicon pix 1→silicon pix 2→metal) this function calculates (and applies) the offset in the `distance` column such that the edges of each metal→silicon and silicon→metal transitions are centered at 50 % of the normalized charge."""
	check_df_is_from_single_1D_scan(df)
	
	if 'Normalized collected charge' not in df.columns:
		df = calculate_normalized_collected_charge(df)
	if 'Pad' not in df.columns:
		df = tag_left_right_pad(df)
	
	mean_df = df.groupby(by = ['n_channel','n_pulse','n_position','Pad']).mean()
	mean_df = mean_df.reset_index()
	
	mean_df = mean_df.loc[mean_df['n_pulse']==1] # Will use only pulse 1 for this.
	mean_distance = mean_df['Distance (m)'].mean()
	mean_df['Distance (m)'] -= mean_distance
	metal_to_silicon_transition_distance = {}
	for pad in sorted(set(mean_df['Pad'])): # 'left' or 'right'
		if pad == 'left':
			distance_vs_normalized_collected_charge = interpolate.interp1d(
				x = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['Distance (m)']<-50e-6), 'Normalized collected charge'],
				y = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['Distance (m)']<-50e-6), 'Distance (m)'],
			)
		else:
			distance_vs_normalized_collected_charge = interpolate.interp1d(
				x = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['n_pulse']==1)&(mean_df['Distance (m)']>50e-6), 'Normalized collected charge'],
				y = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['n_pulse']==1)&(mean_df['Distance (m)']>50e-6), 'Distance (m)'],
			)
		metal_to_silicon_transition_distance[pad] = distance_vs_normalized_collected_charge(.5) # It is the distance in which the normalized collected charge is 0.5
	offset = np.mean(list(metal_to_silicon_transition_distance.values()))
	df['Distance offset by linear interpolation (m)'] = offset + mean_distance
	return df
	
def pre_process_raw_data(data_df):
	"""Given data from a single device, this function performs many "common things" such as calculating the distance, adding the "left pad" or "right pad", etc."""
	check_df_is_from_single_1D_scan(data_df)
	for channel, pad in tag_left_right_pad(data_df).items():
		data_df.loc[data_df['n_channel']==channel, 'Pad'] = pad
	distances_df = calculate_1D_scan_distance_from_dataframe(data_df)
	data_df.set_index('n_position', inplace=True)
	data_df = data_df.merge(distances_df, left_index=True, right_index=True)
	data_df = data_df.append(data_df, ignore_index=True)
	return data_df

def read_and_pre_process_1D_scan_data(measurement_name: str):
	from measurements_table import create_measurements_table # Import here to avoid circular import error.
	measurements_table_df = create_measurements_table()
	df = pre_process_raw_data(read_measured_data_from(measurement_name))
	df['Device'] = measurements_table_df.loc[measurement_name, 'Measured device']
	return df

def mean_std(df, by):
	"""Groups by `by` (list of columns), calculates mean and std, and creates one column with mean and another with std for each column not present in `by`.
	Example
	-------
	df = pandas.DataFrame(
		{
			'n': [1,1,1,1,2,2,2,3,3,3,4,4],
			'x': [0,0,0,0,1,1,1,2,2,2,3,3],
			'y': [1,2,1,1,2,3,3,3,4,3,4,5],
		}
	)

	mean_df = utils.mean_std(df, by=['n','x'])
	
	produces:
	
	   n  x    y mean     y std
	0  1  0  1.250000  0.500000
	1  2  1  2.666667  0.577350
	2  3  2  3.333333  0.577350
	3  4  3  4.500000  0.707107
	"""
	mean_df = df.groupby(by=by).agg(['mean','std'])
	mean_df.columns = [' '.join(col).strip() for col in mean_df.columns.values]
	return mean_df.reset_index()

def get_device_specs_string(device_name: str, humanize=False):
	devices_sheet_df = read_devices_sheet()
	device_name = int(device_name.replace('#',''))
	if humanize == False:
		return f'W{devices_sheet_df.loc[device_name,"wafer"]}-{devices_sheet_df.loc[device_name,"trench depth"]}-{devices_sheet_df.loc[device_name,"trenches"]}-{devices_sheet_df.loc[device_name,"trench process"]}-{devices_sheet_df.loc[device_name,"pixel border"]}-{devices_sheet_df.loc[device_name,"contact type"]}'
	else:
		SEPARATOR_CHAR = ','
		string = f'W{devices_sheet_df.loc[device_name,"wafer"]}'
		string += SEPARATOR_CHAR
		if devices_sheet_df.loc[device_name,"trench depth"] == 'D1':
			string += 'shallow'
		elif devices_sheet_df.loc[device_name,"trench depth"] == 'D2':
			string += 'medium depth'
		elif devices_sheet_df.loc[device_name,"trench depth"] == 'D3':
			string += 'deep'
		string += SEPARATOR_CHAR
		if devices_sheet_df.loc[device_name,"trenches"] == 1:
			string += '1 trench'
		else:
			string += '2 trenches'
		string += SEPARATOR_CHAR
		string += f'process {devices_sheet_df.loc[device_name,"trench process"]}'
		string += SEPARATOR_CHAR
		string += f'border {devices_sheet_df.loc[device_name,"pixel border"]}'
		string += SEPARATOR_CHAR
		string += f'contact {devices_sheet_df.loc[device_name,"contact type"]}'
		return string

def calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(measured_data_df, threshold_percent=50, window_size=125e-6):
	"""Receives a dataframe with the data from a single 1D scan, returns a float number."""
	check_df_is_from_single_1D_scan(measured_data_df)
	if 'Normalized collected charge' not in measured_data_df.columns:
		measured_data_df = calculate_normalized_collected_charge(measured_data_df)
	df = measured_data_df[['n_position','n_pulse','n_trigger','n_channel','Normalized collected charge','Distance (m)','Pad']]
	df = df.query('n_pulse==1') # Use only the first pulse.
	df = mean_std(df, by=['Distance (m)','Pad','n_pulse','n_position'])
	threshold_distance_for_each_pad = {}
	for pad in set(df['Pad']):
		if pad == 'left':
			rows = (df['Distance (m)'] > df['Distance (m)'].mean() - window_size/2) & (df['Pad']==pad)
		elif pad == 'right':
			rows = (df['Distance (m)'] < df['Distance (m)'].mean() + window_size/2) & (df['Pad']==pad)
		else:
			raise ValueError(f'Received a dataframe with {repr(pad)} in the column "Pad", I dont know that this means... It is supposed to be either "left" or "right"...')
		distance_vs_charge_linear_interpolation = interpolate.interp1d(
			x = df.loc[rows,'Normalized collected charge mean'],
			y = df.loc[rows,'Distance (m)'],
		)
		threshold_distance_for_each_pad[pad] = distance_vs_charge_linear_interpolation(threshold_percent/100)
	return {
		'Inter-pixel distance (m)': threshold_distance_for_each_pad['right']-threshold_distance_for_each_pad['left'],
		'Threshold (%)': threshold_percent,
		'Left pad distance (m)': threshold_distance_for_each_pad['left'],
		'Right pad distance (m)': threshold_distance_for_each_pad['right'],
	}

if __name__ == '__main__':
	data_df = read_and_pre_process_1D_scan_data('20211026023917_#65_1DScan_155V')
	print(calculate_interpad_distance_by_linear_interpolation_using_normalized_collected_charge(data_df))
	
