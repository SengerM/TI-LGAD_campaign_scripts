import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import numpy as np
import measurements_table as mt

def script_core(measurement_name: str, force=False):
	if not mt.retrieve_measurement_type(measurement_name) == 'scan 1D sweeping bias voltage':
		raise ValueError(f'Measurement must be a `scan 1D sweeping bias voltage` but measurement named {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	iv_data_df = pandas.DataFrame()
	with bureaucrat.verify_no_errors_context():
		for measurement_name in mt.get__1DScan_sweeping_bias_voltage__list_of_fixed_voltage_scans(measurement_name):
			voltage_summary = pandas.read_csv(utils.path_to_measurements_directory/Path(measurement_name)/Path('summarize_measurement_bias_conditions/bias_voltage_summary.csv'))
			voltage_summary = voltage_summary.iloc[0]
			current_summary = pandas.read_csv(utils.path_to_measurements_directory/Path(measurement_name)/Path('summarize_measurement_bias_conditions/bias_current_summary.csv'))
			current_summary = current_summary.iloc[0]
			iv_data_df = iv_data_df.append(
				pandas.concat([voltage_summary,current_summary]),
				ignore_index = True,
			)
		for variable in {'Bias voltage (V)','Bias current (A)'}:
			for stat in {'mean','median'}:
				col = f'{variable} {stat}'
				iv_data_df[col] *= -1
		iv_data_df = iv_data_df.sort_values(by='Bias voltage (V) mean')
		
		iv_data_df.reset_index().to_feather(bureaucrat.processed_data_dir_path/Path('iv_data.fd'))
		
		fig = utils.line(
			title = f'Reconstructed IV curve from 1D scan data<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			data_frame = iv_data_df,
			x = 'Bias voltage (V) mean',
			y = 'Bias current (A) mean',
			error_x = 'Bias voltage (V) std',
			error_y = 'Bias current (A) std',
			error_y_mode = 'band',
			log_y = True,
			hover_data = sorted(iv_data_df),
		)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path('iv curve.html')),
			include_plotlyjs = 'cdn',
		)

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
			if mt.retrieve_measurement_type(measurement_name) == 'scan 1D sweeping bias voltage':
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason: {repr(e)}.')

