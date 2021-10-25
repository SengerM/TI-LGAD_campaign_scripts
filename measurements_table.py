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
	is_IV_curve = 'IV' in measurement_name
	is_1D_scan_sweeping_bias_voltage = 'sweeping_bias_voltage' in measurement_name and 'scan_1D_sweeping_bias_voltage' in list_of_directories_within_this_measurement
	is_laser_DAC_scan = any(pattern.lower() in measurement_name.lower() for pattern in {'laserDacScan'})
	
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
				are_we_within_if__main__ = False
				for line in ifile:
					if 'variables = locals(), # <-- Variables were registered at this point:' in line:
						for element in line.split(','):
							if variable_name in element:
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

def create_measurements_table():
	measurements_df = pandas.DataFrame(
		{'Measurement name': [path.parts[-1] for path in sorted(utils.path_to_measurements_directory.iterdir()) if path.is_dir()]},
	)
	measurements_df.set_index('Measurement name', inplace=True)
	for measurement_name in measurements_df.index:
		measurements_df.loc[measurement_name, 'When'] = datetime.datetime.strptime(measurement_name.split('_')[0], "%Y%m%d%H%M%S")
		measurements_df.loc[measurement_name, 'Measured device'] = retrieve_device_name(measurement_name)
		measurements_df.loc[measurement_name, 'Type'] = retrieve_measurement_type(measurement_name)
		measurements_df.loc[measurement_name, 'Bias voltage (V)'] = retrieve_bias_voltage(measurement_name)
		measurements_df.loc[measurement_name, 'Laser DAC'] = retrieve_laser_DAC(measurement_name)
	
	return measurements_df

if __name__ == '__main__':
	with pandas.option_context('display.max_rows', None):  # more options can be specified also
		print(create_measurements_table().sort_values(['Bias voltage (V)', 'Type', 'Measurement name']))
	# ~ create_measurements_table().to_excel(utils.path_to_measurements_directory/Path('measurements_table.xlsx'))
