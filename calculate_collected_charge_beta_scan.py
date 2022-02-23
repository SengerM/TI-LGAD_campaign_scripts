import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import measurements_table as mt
from grafica.plotly_utils.utils import scatter_histogram
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from clean_beta_scan import binned_fit_langauss, hex_to_rgba
from beta_scan_time_resolution import resample_measured_data
from landaupy import landau, langauss # https://github.com/SengerM/landaupy

def script_core(measurement_name: str, force=False, n_bootstrap=0):
	if not mt.retrieve_measurement_type(measurement_name) == 'beta scan':
		raise ValueError(f'Measurement must be a `beta scan` but measurement named {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	with bureaucrat.verify_no_errors_context():
		try:
			measured_data_df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.fd'))
		except FileNotFoundError:
			measured_data_df = pandas.read_csv(bureaucrat.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.csv'))
		
		beta_scan_was_cleaned = bureaucrat.job_successfully_completed_by_script('clean_beta_scan.py')
		if beta_scan_was_cleaned:
			df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('clean_beta_scan.py')/Path('clean_triggers.fd'))
			df = df.set_index('n_trigger')
			measured_data_df = measured_data_df.set_index('n_trigger')
			measured_data_df['accepted'] = df['accepted']
			measured_data_df = measured_data_df.reset_index()
		else: # We accept all triggers...
			measured_data_df['accepted'] = True
		measured_data_df = measured_data_df.query('accepted==True') # From now on we drop all useless data.
		
		fig = go.Figure()
		fig.update_layout(
			title = f'Langauss fit on collected charge<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			xaxis_title = 'Collected charge (V s)',
			yaxis_title = 'Probability density',
		)
		colors = iter(px.colors.qualitative.Plotly)
		results_df = pandas.DataFrame()
		for n_bootstrap_iter in range(n_bootstrap+1):
			if n_bootstrap_iter > 0:
				raise NotImplementedError(f'Cannot perform bootstrap at the moment, there is a strange error in Pandas and I dont have time to fix now.')
			this_iteration_data_df = measured_data_df if n_bootstrap_iter == 0 else resample_measured_data(measured_data_df)
			for device_name in sorted(set(measured_data_df['device_name'])):
				samples_to_fit = this_iteration_data_df.query('accepted==True').query(f'device_name=={repr(device_name)}')['Collected charge (V s)']
				popt, _, hist, bin_centers = binned_fit_langauss(samples_to_fit)
				results_df = results_df.append(
					{
						'Variable': 'Collected charge (V s) x_mpv',
						'Device name': device_name,
						'Value': popt[0],
						'Type': 'fit to data' if n_bootstrap_iter == 0 else 'bootstrapped fit to resampled data',
					},
					ignore_index = True,
				)
				
				if n_bootstrap_iter == 0:
					this_channel_color = next(colors)
					fig.add_trace(
						scatter_histogram(
							samples = samples_to_fit,
							error_y = dict(type='auto'),
							density = True,
							name = f'Data {device_name}',
							line = dict(color = this_channel_color),
							legendgroup = device_name,
						)
					)
					x_axis = np.linspace(min(bin_centers),max(bin_centers),999)
					fig.add_trace(
						go.Scatter(
							x = x_axis,
							y = langauss.pdf(x_axis, *popt),
							name = f'Langauss fit {device_name}<br>x<sub>MPV</sub>={popt[0]:.2e}<br>ξ={popt[1]:.2e}<br>σ={popt[2]:.2e}',
							line = dict(color = this_channel_color, dash='dash'),
							legendgroup = device_name,
						)
					)
					fig.add_trace(
						go.Scatter(
							x = x_axis,
							y = landau.pdf(x_axis, popt[0], popt[1]),
							name = f'Landau component {device_name}',
							line = dict(color = f'rgba{hex_to_rgba(this_channel_color, .3)}', dash='dashdot'),
							legendgroup = device_name,
						)
					)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path(f'langauss fit.html')),
			include_plotlyjs = 'cdn',
		)
		results_df.to_csv(bureaucrat.processed_data_dir_path/Path(f'results.csv'),index=False)

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
	N_BOOTSTRAP = 0
	if args.directory.lower() != 'all':
		script_core(Path(args.directory).parts[-1], force=True, n_bootstrap=N_BOOTSTRAP)
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			if mt.retrieve_measurement_type(measurement_name) == 'beta scan':
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name, n_bootstrap=N_BOOTSTRAP)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason: {repr(e)}.')

