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
	
	results_file_path = bureaucrat.processed_data_dir_path/Path('laser_DAC_summary.txt')
	
	measurement_when = mt.retrieve_measurement_when(measurement_name)
	
	with open(results_file_path, 'w') as ofile:
		if mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
			measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
			laser_DAC = int(measured_data_df['Laser DAC'].mean())
			if measurement_when < datetime.datetime(year=2021,month=12,day=1): # Before the TCT computer broke, I was using the "official driver" from Particulars.
				print(f'Laser DAC (mV) = {laser_DAC}', file=ofile)
			else: # Using my own driver.
				print(f'Laser DAC (digital units) = {laser_DAC}', file=ofile)
		elif mt.retrieve_measurement_type(measurement_name) == 'IV curve':
			print(f'Laser was not used during this measurement.', file=ofile)
		else:
			print('Could not find information about laser DAC, but because this script does not know how to interpret this type of measurement.', file=ofile)

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Creates a file summarizing the value of the laser DAC for a measurement. Usually this will be just a single number, e.g. "2000 mV" or "620 DU", but the file may contain anything else like "undefined" or "non-constant" or whatever.')
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
			if not (utils.path_to_measurements_directory/Path(measurement_name)/Path('summarize_measurement_laser_DAC')/Path('laser_DAC_summary.txt')).is_file():
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
