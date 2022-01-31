from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import InterpolatedUnivariateSpline #from scipy.interpolate import interp1d
from scipy.misc import derivative

def t_50_find_cuts(measured_data_df, n_channel):
	ecdf = ECDF(measured_data_df.query(f'n_channel=={n_channel}')['t_50 (s)'])
	t_50_points_for_sampling_ECDF = np.linspace(measured_data_df['t_50 (s)'].min(),measured_data_df['t_50 (s)'].max(),99)
	t_50_points_for_sampling_ECDF = t_50_points_for_sampling_ECDF[~np.isnan(t_50_points_for_sampling_ECDF)]
	t_50_points_for_sampling_ECDF = np.array(sorted(set(t_50_points_for_sampling_ECDF)))
	interpolated_ecdf = InterpolatedUnivariateSpline(t_50_points_for_sampling_ECDF, ecdf(t_50_points_for_sampling_ECDF), k=3) # This is a smooth version of the ECDF, that can be differentiated.
	t_50_axis = np.linspace(min(t_50_points_for_sampling_ECDF), max(t_50_points_for_sampling_ECDF), 999)
	dECDFdt_50 = interpolated_ecdf.derivative()(t_50_axis)
	d2ECDFdt_50_2 = interpolated_ecdf.derivative().derivative()(t_50_axis)
	t_50_center = t_50_axis[np.argmax(dECDFdt_50)]
	t_50_peak_width = t_50_axis[np.argmin(d2ECDFdt_50_2)] - t_50_axis[np.argmax(d2ECDFdt_50_2)]
	t_50_low_cut = t_50_center - t_50_peak_width
	t_50_high_cut = t_50_center + t_50_peak_width
	return t_50_low_cut, t_50_high_cut

def script_core(directory):
	bureaucrat = Bureaucrat(
		directory,
		variables = locals(),
	)
	
	try:
		measured_data_df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.fd'))
	except FileNotFoundError:
		measured_data_df = pandas.read_csv(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.csv'))
	
	cuts_df = pandas.DataFrame(columns=['n_channel','t_50 lower cut (s)','t_50 higher cut (s)'])
	for n_channel in sorted(set(measured_data_df['n_channel'])):
		lower_cut, higher_cut = t_50_find_cuts(measured_data_df, n_channel)
		cuts_df = cuts_df.append(
			{'n_channel': n_channel, 't_50 lower cut (s)': lower_cut, 't_50 higher cut (s)': higher_cut},
			ignore_index = True,
		)
	cuts_df = cuts_df.set_index('n_channel')
	cuts_df.to_csv(bureaucrat.processed_data_dir_path/Path(f't_50_cuts.csv'))
	
	# Tag events that satisfy the t_50 filter ---
	t_50_data_df = measured_data_df.pivot(
		index = 'n_trigger',
		columns = 'n_channel',
		values = 't_50 (s)',
	)
	satisfies = (t_50_data_df<=cuts_df['t_50 higher cut (s)'])&(t_50_data_df>=cuts_df['t_50 lower cut (s)'])
	temp_df = satisfies[sorted(set(satisfies.columns))[0]]
	for n_channel in sorted(set(satisfies.columns)):
		temp_df &= satisfies[n_channel]
	measured_data_df = measured_data_df.set_index('n_trigger')
	measured_data_df['Satisfies t_50 filter?'] = temp_df
	measured_data_df = measured_data_df.reset_index()
	
	for column in measured_data_df:
		if column in {'n_trigger','When','n_channel','Satisfies t_50 filter?'}:
			continue
		histogram_fig = px.histogram(
			measured_data_df,
			x = column,
			facet_row = 'n_channel',
			opacity = .75,
			title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			pattern_shape = 'Satisfies t_50 filter?',
			pattern_shape_map = {False: 'x', True: ''},
			marginal = 'rug',
		)
		if column == 't_50 (s)':
			ecdf_fig = px.ecdf(
				measured_data_df,
				x = column,
				color = 'n_channel',
				title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
				marginal = 'histogram',
			)
			ecdf_fig.write_html(
				str(bureaucrat.processed_data_dir_path/Path(f'{column} ECDF.html')),
				include_plotlyjs = 'cdn',
			)
			for n_channel in sorted(set(measured_data_df['n_channel'])):
				for fig in [histogram_fig, ecdf_fig]:
					fig.add_vrect(
						x0 = cuts_df.loc[n_channel, 't_50 lower cut (s)'],
						x1 = cuts_df.loc[n_channel, 't_50 higher cut (s)'],
						opacity = .25,
						line_width = 0,
						fillcolor = 'black',
						annotation_text = f'Acceptance region CH{n_channel}',
						annotation_textangle = -90,
					)
		
		histogram_fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path(f'{column} histogram.html')),
			include_plotlyjs = 'cdn',
		)
	
	columns_for_scatter_matrix_plot = set(measured_data_df.columns) - {'n_trigger','When','n_channel','Satisfies t_50 filter?','Noise (V)','Time over 20% (s)'} - {f't_{i*10} (s)' for i in [1,2,3,4,6,7,8,9]}
	df = measured_data_df
	df['n_channel'] = df['n_channel'].astype(str) # This is so the color scale is discrete.
	fig = px.scatter_matrix(
		df,
		dimensions = sorted(columns_for_scatter_matrix_plot),
		title = f'Scatter matrix plot<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
		color = 'n_channel',
		symbol = 'Satisfies t_50 filter?',
		symbol_map = {True: 'circle', False: 'x'},
		hover_data = ['n_trigger','Satisfies t_50 filter?'],
	)
	fig.update_traces(diagonal_visible=False, showupperhalf=False)
	for k in range(len(fig.data)):
		fig.data[k].update(
			selected = dict(
				marker = dict(
					opacity = 1,
					color = 'black',
				)
			),
			# ~ unselected = dict(
				# ~ marker = dict(
					# ~ opacity = 0.01
				# ~ )
			# ~ ),
		)
	fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path('scatter matrix')) + '.html',
			include_plotlyjs = 'cdn',
		)
	
########################################################################

if __name__ == '__main__':
	import argparse
	import measurements_table as mt
	import utils

	parser = argparse.ArgumentParser(description='Produces a list with the "nice triggers" using the `t_50` variable.')
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)

	args = parser.parse_args()
	if args.directory.lower() != 'all':
		script_core(Path(args.directory))
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			if mt.retrieve_measurement_type(measurement_name) == 'beta scan':
				print(f'Processing {measurement_name}...')
				try:
					script_core(utils.path_to_measurements_directory/Path(measurement_name))
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason {repr(e)}...')
