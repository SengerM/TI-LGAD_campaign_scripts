import pandas
from pathlib import Path
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from grafica.plotly_utils.utils import line as grafica_line

path_to_base_TI_LGAD = Path('/home/alf/cernbox/projects/4D_sensors/TI-LGAD_FBK_RD50_1')
path_to_measurements_directory = path_to_base_TI_LGAD/Path('measurements_data')

def read_devices_sheet():
	df = pandas.read_excel(
		path_to_base_TI_LGAD/Path('doc/FBK TI-LGAD RD50 1.xlsx'),
		sheet_name = 'devices',
	)
	return df.loc[:, ~df.columns.str.contains('^Unnamed')].set_index('#')

def tag_left_right_pad(data_df):
	if '#' in data_df and len(set(data_df['#'])) > 1:
		raise ValueError(f'`data_df` must contain data from a single device, I have received a dataframe with data from {len(set(data_df["#"]))} devices.')
	channels = set(data_df['n_channel'])
	if len(channels) != 2:
		raise ValueError(f'`data_df` contains data concerning more than two channels. I can only tag left and right pads for two channels data.')
	left_data = data_df.loc[(data_df['n_position']<data_df['n_position'].mean())]
	right_data = data_df.loc[(data_df['n_position']>data_df['n_position'].mean())]
	for channel in channels:
		if left_data.loc[left_data['n_channel']==channel, 'Collected charge (V s)'].mean(skipna=True) > left_data.loc[~(left_data['n_channel']==channel), 'Collected charge (V s)'].mean(skipna=True):
			return {channel: 'left', list(channels-{channel})[0]: 'right'}
		else:
			return {channel: 'right', list(channels-{channel})[0]: 'left'}

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
	x = df.groupby('n_position').mean()[f'x (m)']
	y = df.groupby('n_position').mean()[f'y (m)']
	z = df.groupby('n_position').mean()[f'z (m)']
	distances_df = pandas.DataFrame({'n_position': [i for i in range(len(set(df['n_position'])))], 'Distance (m)': calculate_1D_scan_distance(list(zip(x,y,z)))})
	distances_df.set_index('n_position')
	return distances_df

def calculate_normalized_collected_charge(df):
	"""df must be the dataframe from a single 1D scan."""
	from measurements_table import retrieve_measurement_type # Importing here to avoid circular import.
	
	if len(set(df['Measurement name'])) != 1:
		raise ValueError(f'`df` must contain data from a single measurement, but it seems to contain data from the following measurements: {set(df["Measurement name"])}.')
	measurement_name = sorted(set(df['Measurement name']))[0]
	if retrieve_measurement_type(measurement_name) != 'scan 1D':
		raise ValueError(f'`df` must contain data from a "scan 1D" measurement, but the measurement {repr(measurement_name)} is of type {repr(retrieve_measurement_type(measurement_name))}.')
	
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
	
def pre_process_raw_data(data_df):
	"""Given data from a single device, this function performs many "common things" such as calculating the distance, adding the "left pad" or "right pad", etc."""
	if '#' in data_df and len(set(data_df['#'])) > 1:
		raise ValueError(f'`data_df` must contain data from a single device, I have received a dataframe with data from {len(set(data_df["#"]))} devices.')
	for channel, pad in tag_left_right_pad(data_df).items():
		data_df.loc[data_df['n_channel']==channel, 'Pad'] = pad
	distances_df = calculate_1D_scan_distance_from_dataframe(data_df)
	data_df.set_index('n_position', inplace=True)
	data_df = data_df.merge(distances_df, left_index=True, right_index=True)
	data_df = data_df.append(data_df, ignore_index=True)
	return data_df

def read_and_pre_process_1D_scan_data(measurement_name: str):
	return pre_process_raw_data(read_measured_data_from(measurement_name))

if __name__ == '__main__':
	df = read_measured_data_from('20211024033857_#77_1DScan_88V')
	calculate_normalized_collected_charge(df)
