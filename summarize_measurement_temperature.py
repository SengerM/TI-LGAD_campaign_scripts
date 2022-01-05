import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import measurements_table as mt
import datetime

measurements_table_df = mt.create_measurements_table()

def script_core(measurement_name: str):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	results_file_path = bureaucrat.processed_data_dir_path/Path('temperature_summary.txt')
	
	if measurements_table_df.loc[measurement_name, 'When'] < datetime.datetime(year=2021, month=11, day=20):
		with open(results_file_path, 'w') as ofile:
			print('This measurement was at room temperature without controlling it.', file=ofile)
		return
	
	if mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
		measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
	elif mt.retrieve_measurement_type(measurement_name) == 'IV curve':
		measured_data_df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('IV_curve.py')/Path('measured_data.fd'))
	else: # Default when I don't know what to do...
		measured_data_df = None
	if measured_data_df is not None:
		if 'Temperature (°C)' not in measured_data_df.columns: # Temperature was not measured.
			with open(results_file_path, 'w') as ofile:
				print('Could not find information about temperature, looks like it was not measured.', file=ofile)
		else: # Temperature was measured.
			with open(results_file_path, 'w') as ofile:
				print(f"Temperature mean (°C) = {measured_data_df['Temperature (°C)'].mean(skipna=True)}", file=ofile)
				print(f"Temperature std (°C) = {measured_data_df['Temperature (°C)'].std(skipna=True)}", file=ofile)
	else:
		with open(results_file_path, 'w') as ofile:
			print('Could not find information about temperature, but because this script does not know how to interpret this type of measurement.', file=ofile)

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Creates a file summarizing the measurement temperature. Usually this will be just a single number, e.g. -20 Celsius, but the file may contain anything else like "undefined" or "non-constant" or whatever.')
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
			if not (utils.path_to_measurements_directory/Path(measurement_name)/Path('summarize_measurement_temperature')/Path('temperature_summary.txt')).is_file():
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
