import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path

def script_core(measurement_name: str):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
	
	mean_df = utils.mean_std(measured_data_df, by=['n_position','n_pulse','Pad','Distance (m)'])
	
	for qtty in {'Collected charge (V s)','Normalized collected charge','t_50 (s)'}:
		fig = utils.line(
			data_frame = mean_df,
			x = 'Distance (m)',
			y = qtty + ' mean',
			error_y = qtty + ' std',
			error_y_mode = 'band',
			color = 'Pad',
			line_dash = 'n_pulse',
			labels = {
				'Distance (m)': 'Laser position (m)',
			},
			grouped_legend = True,
			title = qtty + f'<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
		)
		fig.update_layout(
			yaxis_title = qtty,
		)
		fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'{qtty}.html')), include_plotlyjs='cdn')
	
	# Left+right sum plot ---
	df = mean_df.query('n_pulse==1')
	summed_df = df.groupby(by=['Distance (m)']).sum().reset_index()
	summed_df['Pad'] = 'left+right'
	df = df.append(summed_df, ignore_index=True)
	fig = utils.line(
		data_frame = df,
		x = 'Distance (m)',
		y = 'Normalized collected charge mean',
		error_y = 'Normalized collected charge std',
		error_y_mode = 'band',
		color = 'Pad',
		labels = {
			'Distance - offset (m)': 'Laser position (m)',
			'Normalized collected charge mean': 'Normalized collected charge',
			'Pad': '',
		},
		title = f'Left+Right<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
	)
	fig.update_layout(
		yaxis_title = 'Normalized collected charge',
	)
	fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'left+right.html')), include_plotlyjs='cdn')

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Produce some nice plots for Valencia.')
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	script_core(Path(args.directory).parts[-1])
