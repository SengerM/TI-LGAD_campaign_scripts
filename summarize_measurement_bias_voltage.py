import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import measurements_table as mt
import datetime

measurements_table_df = mt.create_measurements_table()

def script_core(measurement_name: str, force=False):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	results_file_path = bureaucrat.processed_data_dir_path/Path('bias_voltage_summary.txt')
	
	with bureaucrat.verify_no_errors_context():
		measurement_when = mt.retrieve_measurement_when(measurement_name)
		
		with open(results_file_path, 'w') as ofile:
			if mt.retrieve_measurement_type(measurement_name) in {'scan 1D','beta scan'}:
				measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
				if 'Bias voltage (V)' in measured_data_df.columns:
					print(f"Bias voltage mean (V) = {measured_data_df['Bias voltage (V)'].mean(skipna=True)}", file=ofile)
					print(f"Bias voltage std (V) = {measured_data_df['Bias voltage (V)'].std(skipna=True)}", file=ofile)
				else:
					print('Could not find information about bias voltage, maybe it was not measured or I dont know where to find it...', file=ofile)

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Creates a file summarizing the value of the bias voltage for a measurement.')
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
		script_core(Path(args.directory).parts[-1], force=True)
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			print(f'Processing {measurement_name}...')
			try:
				script_core(measurement_name)
			except Exception as e:
				print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
