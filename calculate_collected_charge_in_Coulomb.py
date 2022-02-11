import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import measurements_table as mt
from scipy.stats import median_abs_deviation

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
		measured_data_df = measured_data_df.query('n_pulse==1') # Keep only pulse 1 which is the one that goes with the calibration of the TCT's laser.
		
		transimpedance_dict = mt.get_transimpedance_calibration(measurement_name)
		transimpedance = transimpedance_dict['transimpedance (Ω)']
		measured_data_df['Collected charge (C)'] = measured_data_df['Collected charge (V s)']/transimpedance
		
		# Now select data points that are well within the metal openings where the charge should be independent of position ---
		useful_data_index_array_left_pixel = (measured_data_df['Pad']=='left') & (measured_data_df['Distance - offset (m)'] > -100e-6) & (measured_data_df['Distance - offset (m)'] < -50e-6)
		useful_data_index_array_right_pixel = (measured_data_df['Pad']=='right') & (measured_data_df['Distance - offset (m)'] > 50e-6) & (measured_data_df['Distance - offset (m)'] < 100e-6)
		useful_data_df = measured_data_df[useful_data_index_array_left_pixel | useful_data_index_array_right_pixel]
		
		collected_charge_statistics_df = pandas.DataFrame(
			{
				'Pad': ['left', 'right'],
			}
		)
		collected_charge_statistics_df.set_index('Pad', inplace=True)
		for pad in {'left','right'}:
			for unit in {'V s','C'}:
				collected_charge_statistics_df.loc[pad,f'Collected charge ({unit}) mean'] = useful_data_df.query(f'Pad=="{pad}"')[f'Collected charge ({unit})'].mean()
				collected_charge_statistics_df.loc[pad,f'Collected charge ({unit}) std'] = useful_data_df.query(f'Pad=="{pad}"')[f'Collected charge ({unit})'].std()
				collected_charge_statistics_df.loc[pad,f'Collected charge ({unit}) median'] = useful_data_df.query(f'Pad=="{pad}"')[f'Collected charge ({unit})'].median()
				collected_charge_statistics_df.loc[pad,f'Collected charge ({unit}) MAD_std'] = utils.k_MAD_TO_STD*median_abs_deviation(useful_data_df.query(f'Pad=="{pad}"')[f'Collected charge ({unit})'], nan_policy='omit')
		collected_charge_statistics_df.to_csv(bureaucrat.processed_data_dir_path/Path('collected_charge_statistics.csv'))
		
		with open(bureaucrat.processed_data_dir_path/Path('calibration_measurement_used.txt'), 'w') as ofile:
			print(f'Measurement {repr(transimpedance_dict["calibration measurement name"])} was used to obtain the transimpedance value of {transimpedance} Ω.', file=ofile)
		
		# Plot ---
		averaged_df = utils.mean_std(measured_data_df, by=['n_position','Pad'])
		for charge_units in ['V s','C']:
			for statistics in [('median','MAD_std')]:
				fig = utils.line(
					title = f'Collected charge in units of "{charge_units}"<br><sup>Measurement: {measurement_name}</sup>',
					data_frame = averaged_df,
					x = 'Distance - offset (m) mean',
					y = f'Collected charge ({charge_units}) {statistics[0]}',
					error_y = f'Collected charge ({charge_units}) {statistics[1]}',
					error_y_mode = 'band',
					color = 'Pad',
					labels = {
						'Distance - offset (m) mean': 'Distance - offset (m)',
						# ~ f'Collected charge ({charge_units}) mean': f'Collected charge ({charge_units})',
						'Pad': 'Pixel',
					}
				)
				for pad in ['left','right']:
					fig.add_trace(
						go.Scatter(
							x = useful_data_df.query(f'Pad=="{pad}"')['Distance - offset (m)'],
							y = useful_data_df.query(f'Pad=="{pad}"')[f'Collected charge ({charge_units})'],
							name = f'{pad} data',
							mode = 'markers',
						)
					)
				fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'collected charge in units of {charge_units} using {statistics}.html'.replace(' ','_'))), include_plotlyjs = 'cdn')
	
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
			if mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name, force=True)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason: {repr(e)}.')

