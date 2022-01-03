import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import measurements_table as mt

def script_core(measurement_name: str):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	can_we_trust = True
	reasons_not_to_trust = []
	
	# Check that amplifiers did not run into nonlinear mode ---
	DYNAMIC_RANGE = .9 # Volt
	AMMOUNT_OF_SIGNALS_WITHIN_DYNAMIC_RANGE = .95
	measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
	if len(measured_data_df.query(f'`Amplitude (V)` >= {DYNAMIC_RANGE}'))/len(measured_data_df) > 1-AMMOUNT_OF_SIGNALS_WITHIN_DYNAMIC_RANGE:
		can_we_trust = False
		reasons_not_to_trust.append(f'Amplitude is > {DYNAMIC_RANGE} V for (at least) the {(1-AMMOUNT_OF_SIGNALS_WITHIN_DYNAMIC_RANGE)*100:.2f} % of the events, amplifiers go into nonlinear regime.')
	
	# Devices ---
	UNTRUSTABLE_DEVICES = {'1','2','88'}
	if mt.retrieve_device_name(measurement_name) in UNTRUSTABLE_DEVICES:
		can_we_trust = False
		reasons_not_to_trust.append(f'Measured device name is {repr(mt.retrieve_device_name(measurement_name))} which is in the listed of "untrustable devices".')
	
	with open(bureaucrat.processed_data_dir_path/Path('result.txt'), 'w') as ofile:
		print(f'can_we_trust = {"yes" if can_we_trust else "no"}', file=ofile)
		if len(reasons_not_to_trust) > 0:
			print(f'\nReasons not to trust:', file=ofile)
			for reason in reasons_not_to_trust:
				print(f'- {reason}', file=ofile)

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Creates a file saying whether some measurement is reliable or not. For example if the amplitude is > 1 V this means that the amplifiers went into the nonlinear regime so we cannot trust the measurement.')
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
				if not (utils.path_to_measurements_directory/Path(measurement_name)/Path('can_we_trust_this_measurement')).is_dir():
					print(f'Processing {measurement_name}...')
					try:
						script_core(measurement_name)
					except Exception as e:
						print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
