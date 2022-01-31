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
	t_50_peak_width = t_50_axis[np.argmax(d2ECDFdt_50_2)] - t_50_axis[np.argmin(d2ECDFdt_50_2)]
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
	
	for column in {'t_50 (s)'}:
		histogram_fig = px.histogram(
			measured_data_df,
			x = column,
			color = 'n_channel',
			barmode = 'overlay',
			opacity = .75,
			title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			marginal = 'rug',
		)
		ecdf_fig = px.ecdf(
			measured_data_df,
			x = column,
			color = 'n_channel',
			title = f'{column}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			marginal = 'histogram',
		)
		for n_channel in sorted(set(measured_data_df['n_channel'])):
			for fig in [histogram_fig, ecdf_fig]:
				fig.add_vrect(
					x0 = cuts_df.loc[n_channel, 't_50 lower cut (s)'],
					x1 = cuts_df.loc[n_channel, 't_50 higher cut (s)'],
					opacity = .25,
					line_width = 0,
					fillcolor = 'black',
					annotation_text = f'CH{n_channel}',
				)
		
		histogram_fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path(f'{column} histogram.html')),
			include_plotlyjs = 'cdn',
		)
		ecdf_fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path(f'{column} ECDF.html')),
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
