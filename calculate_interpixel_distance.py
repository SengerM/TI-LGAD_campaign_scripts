import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def script_core(measurement_name: str, force=False):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_flag:
		return
	
	with bureaucrat.verify_no_errors_context():
		measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
		measured_data_df = measured_data_df.query('n_pulse == 1')
		
		utils.resample_measured_data(measured_data_df)
		
		interpixel_distances_df = pandas.DataFrame()
		for threshold in sorted({8,22,37,50,60+3,77,92}):
			calculated_values = utils.calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(measured_data_df, threshold_percent=threshold)
			interpixel_distances_df = interpixel_distances_df.append(calculated_values, ignore_index=True)
			if threshold == 50: # Bootstrap IPD ---
				bootstrapped_IPDs = [None]*11
				for k in range(len(bootstrapped_IPDs)):
					fake_IPD = utils.calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(
						utils.resample_measured_data(measured_data_df), 
						threshold_percent = threshold,
					)['Inter-pixel distance (m)']
					bootstrapped_IPDs[k] = fake_IPD
		interpixel_distances_df.set_index('Threshold (%)', inplace=True)
		
		with open(bureaucrat.processed_data_dir_path/Path('interpixel_distance.txt'), 'w') as ofile:
			print(f'Inter-pixel distance (m) = {utils.calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(measured_data_df, threshold_percent=50)["Inter-pixel distance (m)"]}', file=ofile)
		
		fig = utils.line(
			data_frame = utils.mean_std(measured_data_df, by=['Distance (m)','Pad']),
			x = 'Distance (m)',
			y = 'Normalized collected charge median',
			error_y = 'Normalized collected charge MAD_std',
			error_y_mode = 'band',
			color = 'Pad',
			labels = {
				'Normalized collected charge median': 'Normalized collected charge',
				'Distance (m)': 'Laser position (m)',
			},
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
		
		with open(bureaucrat.processed_data_dir_path/Path('interpixel_distance_bootstrapped_values.txt'), 'w') as ofile:
			for ipd in bootstrapped_IPDs:
				print(f'{ipd}', file=ofile)
		df = pandas.DataFrame({'IPD (m)': bootstrapped_IPDs})
		fig = px.histogram(
			title = f'Bootstrapped IPDs<br><sup>{bureaucrat.measurement_name}</sup>',
			data_frame = df, 
			x = 'IPD (m)', 
			marginal = 'rug', 
			nbins = int((df['IPD (m)'].max()-df['IPD (m)'].min())/.1e-6)
		)
		fig.write_html(str(bureaucrat.processed_data_dir_path/Path('bootstrapped_IPD_distribution.html')), include_plotlyjs='cdn')

if __name__ == '__main__':
	import argparse
	import measurements_table as mt
	
	parser = argparse.ArgumentParser(description='Plots every thing measured in an xy scan.')
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement. If "all", the inter-pixel distance is calculated for all linear scans.',
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
					script_core(measurement_name)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
