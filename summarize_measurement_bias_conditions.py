import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import measurements_table as mt
from scipy.stats import median_abs_deviation
import plotly.express as px

measurements_table_df = mt.create_measurements_table()

BIAS_STATS_VARIABLES = {'mean','std','median','MAD_std'}

def script_core(measurement_name: str, force=False):
	if not mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
		raise ValueError(f'Measurement must be a `scan 1D` but measurement named {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	with bureaucrat.verify_no_errors_context():
		measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
		measured_data_df = measured_data_df[['n_trigger','n_position','When','Bias voltage (V)','Bias current (A)']]
		measured_data_df = measured_data_df.dropna() # The bias voltage and current are measured only for some triggers, so drop all those without info.
		
		bias_voltage_dict = {}
		if 'Bias voltage (V)' in measured_data_df.columns:
			bias_voltage_dict['Bias voltage (V) mean'] = measured_data_df['Bias voltage (V)'].mean()
			bias_voltage_dict['Bias voltage (V) std'] = measured_data_df['Bias voltage (V)'].std()
			bias_voltage_dict['Bias voltage (V) median'] = measured_data_df['Bias voltage (V)'].median()
			bias_voltage_dict['Bias voltage (V) MAD_std'] = median_abs_deviation(measured_data_df['Bias voltage (V)'])*utils.k_MAD_TO_STD
		else:
			for var in BIAS_STATS_VARIABLES:
				bias_voltage_dict[f'Bias voltage (V) {var}'] = float('NaN')
		bias_voltage_df = pandas.DataFrame()
		bias_voltage_df = bias_voltage_df.append(bias_voltage_dict, ignore_index=True)
		if set(bias_voltage_df.columns) != {f'Bias voltage (V) {var}' for var in BIAS_STATS_VARIABLES}:
			raise RuntimeError(f'Something is wrong here!')
		bias_voltage_df.to_csv(bureaucrat.processed_data_dir_path/Path('bias_voltage_summary.csv'), index=False)
		
		bias_current_dict = {}
		if 'Bias current (A)' in measured_data_df.columns:
			bias_current_dict['Bias current (A) mean'] = measured_data_df['Bias current (A)'].mean()
			bias_current_dict['Bias current (A) std'] = measured_data_df['Bias current (A)'].std()
			bias_current_dict['Bias current (A) median'] = measured_data_df['Bias current (A)'].median()
			bias_current_dict['Bias current (A) MAD_std'] = median_abs_deviation(measured_data_df['Bias current (A)'])*utils.k_MAD_TO_STD
		else:
			for var in BIAS_STATS_VARIABLES:
				bias_current_dict[f'Bias current (A) {var}'] = float('NaN')
		bias_current_df = pandas.DataFrame()
		bias_current_df = bias_current_df.append(bias_current_dict, ignore_index=True)
		if set(bias_current_df.columns) != {f'Bias current (A) {var}' for var in BIAS_STATS_VARIABLES}:
			raise RuntimeError(f'Something is wrong here!')
		bias_current_df.to_csv(bureaucrat.processed_data_dir_path/Path('bias_current_summary.csv'), index=False)
		
		fig = px.line(
			measured_data_df,
			x = 'When',
			y = 'Bias voltage (V)',
			hover_data = ['n_trigger','n_position','Bias current (A)'],
		)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path('bias voltage vs time.html')),
			include_plotlyjs = 'cdn',
		)
		
		fig = px.line(
			measured_data_df,
			x = 'When',
			y = 'Bias current (A)',
			hover_data = ['n_trigger','n_position','Bias voltage (V)'],
		)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path('bias current vs time.html')),
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
			if not mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
				continue
			print(f'Processing {measurement_name}...')
			try:
				script_core(measurement_name)
			except Exception as e:
				print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
