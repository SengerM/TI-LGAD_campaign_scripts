from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.misc import derivative

def script_core(directory):
	bureaucrat = Bureaucrat(
		directory,
		variables = locals(),
	)
	
	try:
		measured_data_df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.fd'))
	except FileNotFoundError:
		measured_data_df = pandas.read_csv(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.csv'))
	
	interesting_columns = set()
	for column in measured_data_df:
		if column in {'n_trigger','When','n_channel'}: 
			continue
		interesting_columns.add(column)
		fig = px.histogram(
			measured_data_df,
			x = column,
			color = 'n_channel',
			barmode = 'overlay',
			opacity = .75,
			title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			marginal = 'rug',
		)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path(f'{column} histogram.html')),
			include_plotlyjs = 'cdn',
		)
		fig = px.ecdf(
			measured_data_df,
			x = column,
			color = 'n_channel',
			title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			marginal = 'histogram',
		)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path(f'{column} ECDF.html')),
			include_plotlyjs = 'cdn',
		)
	
	columns_for_scatter_matrix_plot = interesting_columns - {f't_{i*10} (s)' for i in [1,2,3,4,6,7,8,9]}
	df = measured_data_df
	df['n_channel'] = df['n_channel'].astype(str) # This is so the color scale is discrete.
	fig = px.scatter_matrix(
		df,
		dimensions = sorted(columns_for_scatter_matrix_plot),
		title = f'Scatter matrix plot<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
		color = 'n_channel',
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
			str(bureaucrat.processed_data_dir_path/Path('scatter matrix')) + '.html',
			include_plotlyjs = 'cdn',
		)

########################################################################

if __name__ == '__main__':
	import argparse
	import measurements_table as mt
	import utils

	parser = argparse.ArgumentParser(description='Makes plots with the distributions of the quantities parsed by the script "parse_raw_data_of_single_beta_scan.py".')
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
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
			if mt.retrieve_measurement_type(measurement_name) == 'beta scan':
				print(f'Processing {measurement_name}...')
				try:
					script_core(utils.path_to_measurements_directory/Path(measurement_name))
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason {repr(e)}...')
