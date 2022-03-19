import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import numpy as np
import measurements_table as mt
import warnings
from inter_pixel_distance_analysis import SORT_VALUES_BY, PLOT_GRAPH_DIMENSIONS, annealing_time_to_label_for_the_plot, EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES

def script_core(measurement_name: str, force=False):
	if not mt.retrieve_measurement_type(measurement_name) == 'beta scan sweeping bias voltage':
		raise ValueError(f'Measurement must be a `beta scan sweeping bias voltage` but measurement named {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	collected_charge_df = pandas.DataFrame()
	with bureaucrat.verify_no_errors_context():
		for measurement_name in mt.get__BetaScan_sweeping_bias_voltage__list_of_fixed_voltage_scans(measurement_name):
			try:
				df = pandas.read_csv(utils.path_to_measurements_directory/Path(measurement_name)/Path('calculate_collected_charge_beta_scan/results.csv'))
			except FileNotFoundError:
				warnings.warn(f'Cannot read data from measurement {repr(measurement_name)}')
				continue
			collected_charge_df = collected_charge_df.append(
				{
					'Collected charge (V s) x_mpv': float(df.query(f'`Device name`=="{mt.retrieve_device_name(measurement_name)}"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to data"')['Value']),
					'Measurement name': measurement_name,
					'Bias voltage (V)': mt.retrieve_bias_voltage(measurement_name),
					'Fluence (neq/cm^2)/1e14': mt.get_measurement_fluence(measurement_name)/1e14,
					'Measurement name': measurement_name,
				},
				ignore_index = True,
			)
		
		df = collected_charge_df.sort_values(by='Bias voltage (V)')
		fig = utils.line(
			title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			data_frame = df,
			x = 'Bias voltage (V)',
			y = 'Collected charge (V s) x_mpv',
			hover_data = sorted(df),
			markers = 'circle',
		)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path('collected charge vs bias voltage.html')),
			include_plotlyjs = 'cdn',
		)
		collected_charge_df.to_csv(bureaucrat.processed_data_dir_path/Path('collected_charge_vs_bias_voltage.csv'))

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
		script_core(Path(args.directory).parts[-1], force=True)
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			if mt.retrieve_measurement_type(measurement_name) == 'beta scan sweeping bias voltage':
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name, force=True)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason: {repr(e)}.')

