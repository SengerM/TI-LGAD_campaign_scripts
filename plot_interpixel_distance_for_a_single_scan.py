import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def script_core(measurement_name: str):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
	measured_data_df = measured_data_df.query('n_pulse == 1')
	
	interpixel_distances_df = pandas.DataFrame()
	for threshold in sorted({11,22,33,44,50,55,66,77,88}):
		calculated_values = utils.calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(measured_data_df, threshold_percent=threshold)
		interpixel_distances_df = interpixel_distances_df.append(calculated_values, ignore_index=True)
	interpixel_distances_df.set_index('Threshold (%)', inplace=True)
	
	fig = utils.line(
		data_frame = utils.mean_std(measured_data_df, by=['Distance (m)','Pad']),
		x = 'Distance (m)',
		y = 'Normalized collected charge mean',
		error_y = 'Normalized collected charge std',
		error_y_mode = 'band',
		color = 'Pad',
		markers = True,
		title = f'Inter-pixel distance<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
	)
	annotations = []
	for threshold in interpixel_distances_df.index:
		arrow = go.layout.Annotation(
			dict(
				x = interpixel_distances_df.loc[threshold, 'Right pad distance (m)'],
				y = threshold/100,
				ax = interpixel_distances_df.loc[threshold, 'Left pad distance (m)'],
				ay = threshold/100,
				xref = "x", 
				yref = "y",
				showarrow = True,
				axref = "x", ayref='y',
				arrowhead = 3,
				arrowwidth = 1.5,
				arrowcolor = 'black' if int(threshold)==50 else 'gray',
			)
		)
		annotations.append(arrow)
		text = go.layout.Annotation(
			dict(
				ax = (interpixel_distances_df.loc[threshold, 'Left pad distance (m)']+interpixel_distances_df.loc[threshold, 'Right pad distance (m)'])/2,
				ay = threshold/100,
				x = (interpixel_distances_df.loc[threshold, 'Left pad distance (m)']+interpixel_distances_df.loc[threshold, 'Right pad distance (m)'])/2,
				y = threshold/100,
				xref = "x", 
				yref = "y",
				text = f'{interpixel_distances_df.loc[threshold,"Inter-pixel distance (m)"]*1e6:.1f} µm<br> ',
				axref = "x", ayref='y',
				font = {'color': 'black' if int(threshold)==50 else 'gray'},
			)
		)
		annotations.append(text)
	fig.update_layout(annotations = annotations)
	fig.write_html(str(bureaucrat.processed_data_dir_path/Path('inter_pixel_distance.html')), include_plotlyjs='cdn')


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Plots every thing measured in an xy scan.')
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
