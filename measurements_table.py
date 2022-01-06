from pathlib import Path
import utils
import pandas
import datetime

possible_paths_of_1D_scan_scripts_backups = [
	Path('scan_1D/backup.scan_1D.py'),
	Path('linear_scan_many_triggers_per_point/backup.linear_scan_many_triggers_per_point.py'),
	Path('1D_scan/backup.1D_scan.py'),
]

def retrieve_device_name(measurement_name):
	for string in measurement_name.split('_'):
		if '#' in string:
			return string.replace('#','')
	return None

def retrieve_measurement_type(measurement_name):
	measurement_path = utils.path_to_measurements_directory/Path(measurement_name)
	list_of_directories_within_this_measurement = [p.parts[-1] for p in measurement_path.iterdir()]
	
	is_1D_scan = any(pattern.lower() in measurement_name.lower() for pattern in {'1DScan','LinearScan','linear_x','linear_y','linearx','lineary'}) or (any(subdir in list_of_directories_within_this_measurement for subdir in {'scan_1D', 'linear_scan_many_triggers_per_point', '1D_scan'}) and 'z_scan_to_find_focus' not in list_of_directories_within_this_measurement)
	is_2D_map = any(pattern.lower() in measurement_name.lower() for pattern in {'2D_map','2DMap','2DScan'})
	is_z_scan_to_find_focus = any(pattern.lower() in measurement_name.lower() for pattern in {'focus','z_scan'})
	is_beta_scan = any(pattern.lower() in measurement_name.lower() for pattern in {'beta scan','betaScan','beta'})
	is_IV_curve = 'IV' in measurement_name or (measurement_path/Path('IV_curve')).is_dir()
	is_1D_scan_sweeping_bias_voltage = 'sweeping_bias_voltage' in measurement_name and 'scan_1D_sweeping_bias_voltage' in list_of_directories_within_this_measurement
	is_laser_DAC_scan = any(pattern.lower() in measurement_name.lower() for pattern in {'laserDacScan','LaserIntensityScan'})
	
	if [is_1D_scan, is_2D_map, is_z_scan_to_find_focus, is_beta_scan, is_IV_curve, is_1D_scan_sweeping_bias_voltage, is_laser_DAC_scan].count(True) != 1: # Cannot determine what this measurement is...
		return None
	if is_1D_scan:
		return 'scan 1D'
	elif is_2D_map:
		return 'map 2D'
	elif is_z_scan_to_find_focus:
		return 'z scan for focus'
	elif is_beta_scan:
		return 'beta scan'
	elif is_IV_curve:
		return 'IV curve'
	elif is_1D_scan_sweeping_bias_voltage:
		return 'scan 1D sweeping bias voltage'
	elif is_laser_DAC_scan:
		return 'laser DAC scan'

def _retrieve_1D_scan_script_variable_from_backup(measurement_name, variable_name):
	measurement_path = utils.path_to_measurements_directory/Path(measurement_name)
	for path in [measurement_path/backup_path for backup_path in possible_paths_of_1D_scan_scripts_backups]:
		try:
			with open(path, 'r') as ifile:
				for line in ifile:
					if 'variables = locals(), # <-- Variables were registered at this point:' in line:
						for element in line.split(','):
							if variable_name in element.split(':')[0]:
								return element.split(':')[-1]
		except FileNotFoundError:
			pass
	return '?'

def retrieve_bias_voltage(measurement_name):
	if retrieve_measurement_type(measurement_name) not in {'scan 1D', 'z scan for focus'}:
		return '-'
	bias_voltage = _retrieve_1D_scan_script_variable_from_backup(measurement_name, 'bias_voltage')
	try:
		bias_voltage = float(bias_voltage)
	except:
		pass
	return bias_voltage

def retrieve_laser_DAC(measurement_name):
	measurement_path = utils.path_to_measurements_directory/Path(measurement_name)
	if retrieve_measurement_type(measurement_name) not in {'scan 1D', 'z scan for focus'}:
		return '-'
	laser_DAC = _retrieve_1D_scan_script_variable_from_backup(measurement_name, 'laser_DAC')
	try:
		laser_DAC = int(laser_DAC)
	except:
		pass
	return laser_DAC

def can_we_trust_this_measurement(measurement_name: str) -> 'yes, no, ?':
	"""Looks for information about whether the measurement is a good one so we can trust, in this case returns 'yes', or if it is a bad one and we cannot trust this data, in this case returns 'no'. If there is no certainty whether we can trust or not the measurement, returns '?'."""
	can_we_trust = '?'
	trust_script_results_file_path = utils.path_to_measurements_directory/Path(measurement_name)/Path('can_we_trust_this_measurement/result.txt')
	if trust_script_results_file_path.is_file():
		with open(trust_script_results_file_path, 'r') as ifile:
			for line in ifile:
				if 'can_we_trust' in line:
					result = line.split('=')[-1].lower()
					if 'yes' in result:
						can_we_trust = 'yes'
					elif 'no' in result:
						can_we_trust = 'no'
	return can_we_trust

def retrieve_measurement_temperature(measurement_name: str):
	"""Looks for the results of the `summarize_measurement_temperature.py` script and parses the information. Return type can vary, it can be a float or a string with a message."""
	temperature_summary_file_path = utils.path_to_measurements_directory/Path(measurement_name)/Path('summarize_measurement_temperature/temperature_summary.txt')
	if not temperature_summary_file_path.is_file():
		return '?'
	with open(temperature_summary_file_path, 'r') as ifile:
		for line in ifile:
			if 'This measurement was at room temperature without controlling it.' in line:
				return 'not controlled room temperature'
			if 'Could not find information about temperature' in line:
				return '?'
			if 'Temperature mean (°C) =' in line:
				try:
					temperature_mean = float(line.split('=')[-1])
				except:
					pass
			if 'Temperature std (°C) =' in line:
				try:
					temperature_std = float(line.split('=')[-1])
				except:
					pass
		if 'temperature_mean' in locals():
			if'temperature_std' in locals():
				if temperature_std/temperature_mean < 1/100: # Temperature was very stable and constant, return just the single number.
					return temperature_mean
				else: # Otherwise inform about the variance.
					return f'{temperature_mean}+-{temperature_std}'
			else: # 'temperature_std' not in locals()
				return temperature_mean
	return '?' # Default case.

def retrieve_measurement_when(measurement_name: str):
	"""Returns the measurement "when" as a datetime object."""
	return datetime.datetime.strptime(measurement_name.split('_')[0], "%Y%m%d%H%M%S")

def get_transimpedance_calibration(measurement_name: str) -> dict:
	"""Returns the transimpedance to be used in the measurement to convert charge in `V s` to `Coulomb`. Returns a dictionary of the form {'transimpedance (Ω)': float, 'calibration measurement name': str}"""
	CALIBRATION_MEASUREMENTS = {
		'October 2021 at room T': '20211005105459_#57_88V_BetaScan',
		'December 2021 at -20 °C': '20211229205638_#27_BetaScan_-20Celsius_99V',
	}
	measurement_when = retrieve_measurement_when(measurement_name)
	calibration_measurement_to_use = None
	if retrieve_measurement_when(CALIBRATION_MEASUREMENTS['October 2021 at room T']) < measurement_when < retrieve_measurement_when(CALIBRATION_MEASUREMENTS['December 2021 at -20 °C']):
		calibration_measurement_to_use = CALIBRATION_MEASUREMENTS['October 2021 at room T']
	else:
		measurement_temperature = retrieve_measurement_temperature(measurement_name)
		if isinstance(measurement_temperature, float) and -25 < measurement_temperature < -15:
			calibration_measurement_to_use = CALIBRATION_MEASUREMENTS['December 2021 at -20 °C']
	if calibration_measurement_to_use is None:
		raise RuntimeError(f'Cannot find an appropriate transimpedance calibration for measurement {repr(measurement_name)}.')
	
	TRANSIMPEDANCE_FILE_SUB_PATH = Path('PIN_diode_transimpedance_calibration_using_beta_scan/transimpedance_calibration.txt')
	with open(utils.path_to_measurements_directory/Path(calibration_measurement_to_use)/TRANSIMPEDANCE_FILE_SUB_PATH, 'r') as ifile:
		for line in ifile:
			if 'Transimpedance (Ω) =' in line:
				transimpedance = float(line.split('=')[-1])
	if 'transimpedance' not in locals():
		raise RuntimeError(f'Cannot get the transimpedance from the file {utils.path_to_measurements_directory/Path(calibration_measurement_to_use)/TRANSIMPEDANCE_FILE_SUB_PATH}')
	return {'transimpedance (Ω)': transimpedance, 'calibration measurement name': calibration_measurement_to_use}

def get_measurement_fluence(measurement_name: str) -> float:
	"""Returns the fluence as a float number in n_eq/cm^2 of the detector at the moment the measurement was performed."""
	measurement_when = retrieve_measurement_when(measurement_name)
	if measurement_when < datetime.datetime(year=2021, month=12, day=1):
		return 0 # This was before any irradiation had been carried out.
	measured_device = retrieve_device_name(measurement_name)
	if measured_device is None:
		return float('NaN') # We don't know what to do here...
	if measurement_when > utils.bureaucrat.devices_sheet_df.loc[measured_device,'irradiation date']:
		return utils.bureaucrat.devices_sheet_df.loc[measured_device,'neutrons (neq/cm^2)']

def create_measurements_table():
	measurements_df = pandas.DataFrame(
		{'Measurement name': [path.parts[-1] for path in sorted(utils.path_to_measurements_directory.iterdir()) if path.is_dir()]},
	)
	measurements_df.set_index('Measurement name', inplace=True)
	for measurement_name in measurements_df.index:
		measurements_df.loc[measurement_name, 'When'] = retrieve_measurement_when(measurement_name)
		measurements_df.loc[measurement_name, 'Measured device'] = retrieve_device_name(measurement_name)
		measurements_df.loc[measurement_name, 'Type'] = retrieve_measurement_type(measurement_name)
		measurements_df.loc[measurement_name, 'Bias voltage (V)'] = retrieve_bias_voltage(measurement_name)
		measurements_df.loc[measurement_name, 'Laser DAC'] = retrieve_laser_DAC(measurement_name)
	measurements_df['Can we trust?'] = measurements_df.index.map(can_we_trust_this_measurement)
	measurements_df['Temperature (°C)'] = measurements_df.index.map(retrieve_measurement_temperature)
	return measurements_df

if __name__ == '__main__':
	fpath = Path('measurements_table.xlsx')
	create_measurements_table().to_excel(fpath)
	print(f'Measurements table was saved in {fpath.absolute()}')
