import pandas
from pathlib import Path
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from grafica.plotly_utils.utils import line as grafica_line
from scipy import interpolate
from scipy.stats import median_abs_deviation
import warnings

path_to_base_TI_LGAD = Path('/home/alf/cernbox/projects/4D_sensors/TI-LGAD_FBK_RD50_1')
path_to_measurements_directory = path_to_base_TI_LGAD/Path('measurements_data')
path_to_scripts_output_directory = path_to_base_TI_LGAD/Path('scripts')/Path('scripts_output')
path_to_dashboard_media_directory = Path('/home/alf/Desktop/TI-LGADs_dashboard/media')

k_MAD_TO_STD = 1.4826 # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation

def read_devices_sheet():
	df = pandas.read_excel(
		path_to_base_TI_LGAD/Path('doc/FBK TI-LGAD RD50 1.xlsx'),
		sheet_name = 'devices',
	)
	df['Device'] = df['#'].astype(str)
	return df.loc[:, ~df.columns.str.contains('^Unnamed')].set_index('Device')

def check_df_is_from_single_1D_scan(df):
	"""If df contains data that looks like that from a single 1D scan of one device, this function does nothing. Otherwise, it will rise ValueError."""
	from measurements_table import retrieve_measurement_type # Importing here to avoid circular import.
	if len(set(df['Measurement name'])) != 1:
		raise ValueError(f'`df` must contain data from a single measurement, but it seems to contain data from the following measurements: {set(df["Measurement name"])}.')
	measurement_name = sorted(set(df['Measurement name']))[0]
	if retrieve_measurement_type(measurement_name) != 'scan 1D':
		raise ValueError(f'`df` must contain data from a "scan 1D" measurement, but the measurement {repr(measurement_name)} is of type {repr(retrieve_measurement_type(measurement_name))}.')

def tag_left_right_pad(data_df):
	"""Given a data_df with data from a single 1D scan of two pads of a device, this function adds a new column indicating if the pad is "left" or "right"."""
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
	pad_df = pandas.DataFrame(index=data_df.index)
	for n_channel in set(data_df['n_channel']):
		pad_df.loc[data_df['n_channel']==n_channel, 'Pad'] = mapping[n_channel]
	data_df['Pad'] = pad_df

def read_measured_data_from(measurement_name: str):
	"""Reads the data from a 1D scan and returns a dataframe. The dataframe is returned "intact" in the sense that nothing is done, except that a column is added indicating the measurement name."""
	for scan_script_name in ['linear_scan_many_triggers_per_point','1D_scan', 'scan_1D']:
		try: # First try to read feather as it is much faster.
			df = pandas.read_feather(path_to_measurements_directory/Path(measurement_name)/Path(scan_script_name)/Path('measured_data.fd'))
			df['Measurement name'] = measurement_name
			break
		except FileNotFoundError:
			pass
		try:
			df = pandas.read_csv(path_to_measurements_directory/Path(measurement_name)/Path(scan_script_name)/Path('measured_data.csv'))
			df['Measurement name'] = measurement_name
			break
		except FileNotFoundError:
			pass
	if 'df' not in locals():
		raise FileNotFoundError(f'Cannot find measured data for measurement {repr(measurement_name)}.')
	return df.sort_values(by=['n_position','n_trigger','n_trigger','n_channel'])

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
	return distances_df.set_index('n_position')

def append_distance_column(df):
	"""Given a data frame with data from a single 1D scan, this function calculates the distance in meters at each `n_position` and then appends a new column "Distance (m)" to it."""
	check_df_is_from_single_1D_scan(df)
	distance_df = calculate_1D_scan_distance_from_dataframe(df)
	n_positions_df = pandas.DataFrame({'n_position': df['n_position']})
	n_positions_df.set_index('n_position', inplace=True)
	n_positions_df = n_positions_df.merge(distance_df, left_index=True, right_index=True)
	df.reset_index(inplace=True)
	df.set_index('n_position', inplace=True)
	df['Distance (m)'] = n_positions_df
	df.reset_index(inplace=True)
	if 'index' in df.columns:
		df.drop(columns='index', inplace=True)

def calculate_normalized_collected_charge(df, window_size=125e-6, laser_sigma=9e-6):
	"""df must be the dataframe from a single 1D scan. `window_size` and `laser_sigma` are used to know where we expect zero signal and where we expect full signal.
	Return a single-column-dataframe containint the value of the normalized collected charge at each row."""
	check_df_is_from_single_1D_scan(df)
	normalized_charge_df = pandas.DataFrame(index=df.index)
	normalized_charge_df['Normalized collected charge'] = df['Collected charge (V s)'].copy()
	if 'Pad' not in df.columns:
		raise RuntimeError(f'Before calling this function you have to call `tag_left_right_pad` function on your data frame.')
	if 'Distance (m)' not in df.columns:
		raise RuntimeError(f'Before calling this function you have to call `append_distance_column` function on your data frame.')
	for n_pulse in sorted(set(df['n_pulse'])):
		for pad in {'left','right'}:
			rows_where_I_expect_no_signal_i_e_where_there_is_metal = (df['Distance (m)'] < df['Distance (m)'].median() - window_size - 2*laser_sigma) | (df['Distance (m)'] > df['Distance (m)'].median() + window_size + 2*laser_sigma)
			if pad == 'left':
				rows_where_I_expect_full_signal_i_e_where_there_is_silicon = (df['Distance (m)'] > df['Distance (m)'].median() - window_size + 2*laser_sigma) & (df['Distance (m)'] < df['Distance (m)'].median() - 2*laser_sigma)
			elif pad == 'right':
				rows_where_I_expect_full_signal_i_e_where_there_is_silicon = (df['Distance (m)'] < df['Distance (m)'].median() + window_size - 2*laser_sigma) & (df['Distance (m)'] > df['Distance (m)'].median() + 2*laser_sigma)
			offset_to_subtract = normalized_charge_df.loc[rows_where_I_expect_no_signal_i_e_where_there_is_metal&(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'].median()
			normalized_charge_df.loc[(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'] -= offset_to_subtract
			scale_factor = normalized_charge_df.loc[rows_where_I_expect_full_signal_i_e_where_there_is_silicon&(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'].median()
			normalized_charge_df.loc[(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'] /= scale_factor
	return normalized_charge_df

def append_normalized_collected_charge_column(df, window_size=125e-6, laser_sigma=9e-6):
	"""Given a data frame with data from a single 1D scan, this function calculates the normalized collected charge and appends a new column to the data frame."""
	check_df_is_from_single_1D_scan(df)
	df['Normalized collected charge'] = calculate_normalized_collected_charge(df, window_size=window_size, laser_sigma=laser_sigma)

def calculate_distance_offset_by_linear_interpolation(df):
	"""Given data from a 1D scan from two complete pixels (i.e. scanning from metal→silicon pix 1→silicon pix 2→metal) this function calculates the offset in the `distance` column such that the edges of each metal→silicon and silicon→metal transitions are centered at 50 % of the normalized charge.
	Returns a single float number with the offset."""
	check_df_is_from_single_1D_scan(df)
	
	if 'Normalized collected charge' not in df.columns:
		raise RuntimeError(f'Before calling this function you must add the "normalized collected charge" column to the data frame by calling `append_normalized_collected_charge_column`.')
	
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
	return offset + mean_distance

def append_centered_distance_column(df):
	"""Given a df from a single 1D scan of two pixels, this function appends a new column 'Distance - offset (m)' such that the scan is centered in 0 using the 50 % of the collected charge at each metal-silicon interface."""
	df['Distance - offset (m)'] = df['Distance (m)'] - calculate_distance_offset_by_linear_interpolation(df)
	
def pre_process_raw_data(df):
	"""Given data from a single device, this function performs many "common things" such as calculating the distance, adding the "left pad" or "right pad", etc."""
	tag_left_right_pad(df)
	append_distance_column(df)
	append_normalized_collected_charge_column(df)
	append_centered_distance_column(df)
	return df

def read_and_pre_process_1D_scan_data(measurement_name: str):
	from measurements_table import create_measurements_table # Import here to avoid circular import error.
	measurements_table_df = create_measurements_table()
	df = read_measured_data_from(measurement_name)
	df = pre_process_raw_data(df)
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
	def MAD_std(x):
		return median_abs_deviation(x, nan_policy='omit')*k_MAD_TO_STD
	with warnings.catch_warnings(): # There is a deprecation warning that will be converted into an error in future versions of Pandas. When that happens, I will solve this.
		warnings.simplefilter("ignore")
		mean_df = df.groupby(by=by).agg(['mean','std',np.median,MAD_std])
	mean_df.columns = [' '.join(col).strip() for col in mean_df.columns.values]
	return mean_df.reset_index()

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
			x = df.loc[rows,'Normalized collected charge median'],
			y = df.loc[rows,'Distance (m)'],
		)
		threshold_distance_for_each_pad[pad] = distance_vs_charge_linear_interpolation(threshold_percent/100)
	return {
		'Inter-pixel distance (m)': threshold_distance_for_each_pad['right']-threshold_distance_for_each_pad['left'],
		'Threshold (%)': threshold_percent,
		'Left pad distance (m)': threshold_distance_for_each_pad['left'],
		'Right pad distance (m)': threshold_distance_for_each_pad['right'],
	}

def read_previously_calculated_time_resolution(measurement_name):
	time_resolution_df = pandas.read_feather(path_to_measurements_directory/Path(measurement_name)/Path('time_resolution_analysis')/Path('time_resolution.fd'))
	time_resolution_df['Measurement name'] = measurement_name
	return time_resolution_df

def read_previously_calculated_inter_pixel_distance(measurement_name):
	with open(path_to_measurements_directory/Path(measurement_name)/Path('calculate_interpixel_distance')/Path('interpixel_distance.txt'), 'r') as ifile:
		for line in ifile:
			if 'Inter-pixel distance (m) = ' in line:
				return float(line.replace('Inter-pixel distance (m) = ',''))

def read_previously_calculated_distance_calibration_factor(measurement_name):
	with open(path_to_measurements_directory/Path(measurement_name)/Path('fit_erf_and_calculate_calibration_factor')/Path('scale_factor.txt'), 'r') as ifile:
		for line in ifile:
			if 'multiply_distance_by_this_scale_factor_to_fix_calibration = ' in line:
				calibration_factor = float(line.replace('multiply_distance_by_this_scale_factor_to_fix_calibration = ',''))
	return calibration_factor

def read_previously_calculated_transimpedance(measurement_name):
	with open(path_to_measurements_directory/Path(measurement_name)/Path('calculate_collected_charge_in_Coulomb')/Path('calibration_measurement_used.txt'), 'r') as ifile:
		for line in ifile:
			if 'the transimpedance value of' in line:
				transimpedance = float(line.split('the transimpedance value of')[-1].replace(' Ω.',''))
	return transimpedance

def resample_measured_data(measured_data_df):
	resampled_df = measured_data_df.groupby(by=['n_channel', 'n_position', 'Pad']).sample(frac=1, replace=True)
	return resampled_df

class Bureaucrat:
	# This class is just to avoid reloading the files each time I need data.
	@property
	def devices_sheet_df(self):
		if not hasattr(self, '_devices_sheet_df'):
			self._devices_sheet_df = read_devices_sheet()
		return self._devices_sheet_df
	
	def get_devices_specs_dictionary(self, device_name: str):
		"""Returns a dictionary containing the row for such device in the devices_sheet."""
		device_name = device_name.replace('#','')
		return self.devices_sheet_df.loc[device_name].to_dict()
	
	def get_device_specs_string(self, device_name: str, humanize=False):
		devices_sheet_df = self.devices_sheet_df
		device_name = int(device_name.replace('#',''))
		device_layout = '4×4' if '4×4' in devices_sheet_df.loc[device_name,'type'] else '2×2'
		if humanize == False:
			string = f'W{devices_sheet_df.loc[device_name,"wafer"]}-'
			string += '{devices_sheet_df.loc[device_name,"trench depth"]}-'
			string += '{devices_sheet_df.loc[device_name,"trenches"]}-'
			string += '{devices_sheet_df.loc[device_name,"trench process"]}-'
			string += '{devices_sheet_df.loc[device_name,"pixel border"]}-'
			# ~ string += '{devices_sheet_df.loc[device_name,"contact type"]}-'
			string += '{device_layout}'
			return string
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
			# ~ string += SEPARATOR_CHAR
			# ~ string += f'contact {devices_sheet_df.loc[device_name,"contact type"]}'
			string += SEPARATOR_CHAR
			string += f'pads {device_layout}'
			return string

bureaucrat = Bureaucrat()

if __name__ == '__main__':
	measured_data = read_and_pre_process_1D_scan_data('20211031011655_#68_1DScan_138V')
	print(measured_data['n_position'])
	print('-------------------------------------------------------------')
	print('-------------------------------------------------------------')
	
	print(sorted(measured_data.columns))
