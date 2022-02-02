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
	t_50_low_cut = t_50_center - t_50_peak_width*2
	t_50_high_cut = t_50_center + t_50_peak_width*2
	return t_50_low_cut, t_50_high_cut

def apply_cuts(data_df, cuts_df):
	"""
	Given a dataframe `cuts_df` with one cut per row, e.g.
	```
				  variable  n_channel cut type     cut value
				  t_50 (s)          1    lower  1.341500e-07
				  t_50 (s)          1   higher  1.348313e-07
	Collected charge (V s)          4    lower  2.204645e-11

	```
	this function returns a series with the index `n_trigger` and the value
	either `True` or `False` stating if such trigger satisfies ALL the
	cuts at the same time. For example using the previous example a 
	trigger with charge 3e-12 and t_50 6.45e-8 will be `True` but if any
	of the variables in any of the channels is outside the range, it will
	be `False`.
	"""
	
	data_df = data_df.pivot(
		index = 'n_trigger',
		columns = 'n_channel',
		values = set(data_df.columns) - {'n_channel','n_channel'},
	)
	triggers_accepted_df = pandas.DataFrame({'accepted': True}, index=data_df.index)
	for idx, cut_row in cuts_df.iterrows():
		if cut_row['cut type'] == 'lower':
			triggers_accepted_df['accepted'] &= data_df[(cut_row['variable'],cut_row['n_channel'])] >= cut_row['cut value']
		elif cut_row['cut type'] == 'higher':
			triggers_accepted_df['accepted'] &= data_df[(cut_row['variable'],cut_row['n_channel'])] <= cut_row['cut value']
		else:
			raise ValueError(f'Received a cut of type `cut type={cut_type}`, dont know that that is...')
	return triggers_accepted_df

def script_core(directory):
	bureaucrat = Bureaucrat(
		directory,
		variables = locals(),
	)
	
	plots_dir_path = bureaucrat.processed_data_dir_path/Path('plots')
	plots_dir_path.mkdir(exist_ok=True, parents=True)
	
	try:
		measured_data_df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.fd'))
	except FileNotFoundError:
		measured_data_df = pandas.read_csv(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.csv'))
	
	with bureaucrat.verify_no_errors_context():
		cuts_df = pandas.read_excel(bureaucrat.measurement_base_path/Path('cuts.ods'))
		cuts_df.to_csv(bureaucrat.processed_data_dir_path/Path(f'cuts.csv'))
		
		filtered_triggers_df = apply_cuts(measured_data_df, cuts_df)
		measured_data_df = measured_data_df.set_index('n_trigger')
		measured_data_df['Accepted'] = filtered_triggers_df
		measured_data_df = measured_data_df.reset_index()
		
		filtered_triggers_df.reset_index().to_feather(bureaucrat.processed_data_dir_path/Path('filtered_triggers.fd'))
		
		for column in measured_data_df:
			if column in {'n_trigger','When','n_channel','Accepted'}:
				continue
			histogram_fig = px.histogram(
				measured_data_df,
				x = column,
				facet_col = 'n_channel',
				title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
				color = 'Accepted',
				color_discrete_map = {False: 'red', True: 'green'},
				pattern_shape_map = {False: 'x', True: ''},
				marginal = 'rug',
			)
			if column in set(cuts_df['variable']):
				ecdf_fig = px.ecdf(
					measured_data_df,
					x = column,
					color = 'n_channel',
					title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
					marginal = 'histogram',
					facet_row = 'Accepted',
				)
				cuts_to_draw_df = cuts_df.loc[cuts_df['variable']==column]
				if len(cuts_to_draw_df) > 0:
					for n_channel in sorted(set(cuts_to_draw_df['n_channel'])):
						for cut_type in sorted(set(cuts_to_draw_df.loc[cuts_to_draw_df['n_channel']==n_channel,'cut type'])):
							for fig in [histogram_fig, ecdf_fig]:
								fig.add_vline(
									x = float(cuts_df.loc[(cuts_df['n_channel']==n_channel)&(cuts_df['cut type']==cut_type)&(cuts_df['variable']==column), 'cut value']),
									opacity = .5,
									annotation_text = f'CH{n_channel} {cut_type} cut️',
									line_color = 'black',
									line_dash = 'dash',
									annotation_textangle = -90,
									annotation_position = 'bottom left',
								)
				ecdf_fig.write_html(
					str(plots_dir_path/Path(f'{column} ECDF.html')),
					include_plotlyjs = 'cdn',
				)
			
			histogram_fig.write_html(
				str(plots_dir_path/Path(f'{column} histogram.html')),
				include_plotlyjs = 'cdn',
			)
		
		columns_for_scatter_matrix_plot = set(measured_data_df.columns) - {'n_trigger','When','n_channel','Accepted','Noise (V)','Time over 20% (s)'} - {f't_{i*10} (s)' for i in [1,2,3,4,6,7,8,9]}
		df = measured_data_df
		df['n_channel'] = df['n_channel'].astype(str) # This is so the color scale is discrete.
		fig = px.scatter_matrix(
			df,
			dimensions = sorted(columns_for_scatter_matrix_plot),
			title = f'Scatter matrix plot<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			symbol = 'n_channel',
			color = 'Accepted',
			color_discrete_map = {False: 'red', True: 'green'},
			symbol_map = {True: 'circle', False: 'x'},
			hover_data = ['n_trigger'],
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
				str(plots_dir_path/Path('scatter matrix')) + '.html',
				include_plotlyjs = 'cdn',
			)
	
########################################################################

if __name__ == '__main__':
	import argparse
	import measurements_table as mt
	import utils

	parser = argparse.ArgumentParser(description='Cleans a beta scan according to some criterion.')
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
