from pathlib import Path
import measurements_table as mt
from summarize_measurement_laser_DAC import script_core as summarize_measurement_laser_DAC
from summarize_measurement_temperature import script_core as summarize_measurement_temperature
from can_we_trust_this_measurement import script_core as can_we_trust_this_measurement
from calculate_interpixel_distance import script_core as calculate_interpixel_distance
from fit_erf_and_calculate_calibration_factor import script_core as fit_erf_and_calculate_calibration_factor
from calculate_collected_charge_in_Coulomb import script_core as calculate_collected_charge_in_Coulomb
from calculate_time_resolution import script_core as calculate_time_resolution

def script_core(measurement_name: str):
	if not mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
		raise ValueError(f'Measurement must be a `scan 1D` but measurement named {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	functions = {
		'summarize_measurement_laser_DAC': summarize_measurement_laser_DAC,
		'summarize_measurement_temperature': summarize_measurement_temperature,
		'can_we_trust_this_measurement': can_we_trust_this_measurement,
		'calculate_interpixel_distance': calculate_interpixel_distance,
		'fit_erf_and_calculate_calibration_factor': fit_erf_and_calculate_calibration_factor,
		'calculate_collected_charge_in_Coulomb': calculate_collected_charge_in_Coulomb,
		'calculate_time_resolution': calculate_time_resolution,
	}
	for name, func in functions.items():
		try:
			func(measurement_name)
		except Exception as e:
			print(f'{measurement_name}: Cannot {name}, reason: {repr(e)}')

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement. If "all", the script is applied to all linear scans.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	if args.directory.lower() != 'all':
		script_core(Path(args.directory).parts[-1])
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			if mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
				print(f'Processing {measurement_name}...')
				script_core(measurement_name)
