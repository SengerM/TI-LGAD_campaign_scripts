import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import measurements_table as mt
import grafica
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit

def gaussian(x, mu, sigma, amplitude=1):
	return amplitude/sigma/(2*np.pi)**.5*np.exp(-((x-mu)/sigma)**2/2)

def display_dataframe(df):
	print(df)
	print(sorted(df.columns))

def resample_measured_data(measured_data_df):
	resampled_df = measured_data_df.pivot(
		index = 'n_trigger',
		columns = 'n_channel',
		values = set(measured_data_df.columns) - {'n_trigger','n_channel'},
	)
	resampled_df = resampled_df.sample(frac=1, replace=True)
	resampled_df = resampled_df.stack()
	resampled_df = resampled_df.reset_index()
	return resampled_df

def calculate_Delta_t_df(data_df):
	"""`data_df` must possess the following columns: 'n_trigger','n_pulse','t_10 (s)','t_20 (s)',...,'t_90 (s)'"""
	for col in {'n_trigger','n_pulse'}.union({f't_{i} (s)' for i in [10,20,30,40,50,60,70,80,90]}):
		if col not in data_df.columns:
			raise ValueError(f'{repr(col)} is not a column of `data_df`.')
	
	data_df = data_df.copy() # Don't want to touch the original...
	data_df.set_index('n_trigger', inplace=True)
	pulse_1_df = data_df.query(f'n_pulse==1')
	pulse_2_df = data_df.query(f'n_pulse==2')
	
	Delta_t_df = pandas.DataFrame()
	for k1 in [10,20,30,40,50,60,70,80,90]:
		for k2 in [10,20,30,40,50,60,70,80,90]:
			temp_df = pandas.DataFrame()
			temp_df['Δt (s)'] = pulse_1_df[f't_{k1} (s)'] - pulse_2_df[f't_{k2} (s)']
			temp_df['k_1 (%)'] = k1
			temp_df['k_2 (%)'] = k2
			temp_df.reset_index(inplace=True)
			temp_df.set_index(['n_trigger','k_1 (%)','k_2 (%)'], inplace=True)
			Delta_t_df = Delta_t_df.append(temp_df)
	Delta_t_df.reset_index(inplace=True)
	Delta_t_df = Delta_t_df.dropna()
	return Delta_t_df

def calculate_Delta_t_fluctuations_df(Delta_t_df):
	Delta_t_fluctuations_df = Delta_t_df.groupby(by=['k_1 (%)','k_2 (%)']).agg(median_abs_deviation).reset_index()
	Delta_t_fluctuations_df.drop('n_trigger', axis=1, inplace=True)
	Delta_t_fluctuations_df.rename(columns={'Δt (s)': 'MAD(Δt) (s)'}, inplace=True)
	Delta_t_fluctuations_df['k MAD(Δt) (s)'] = Delta_t_fluctuations_df['MAD(Δt) (s)']*utils.k_MAD_TO_STD
	return Delta_t_fluctuations_df.set_index(['k_1 (%)','k_2 (%)'])

def find_best_k1_k2(Delta_t_fluctuations_df):
	if Delta_t_fluctuations_df.index.names != ['k_1 (%)', 'k_2 (%)']:
		raise ValueError(f"The indices of `Delta_t_fluctuations_df` must be ['k_1 (%)', 'k_2 (%)'].")
	return Delta_t_fluctuations_df['k MAD(Δt) (s)'].idxmin() # k1, k2

def plot_cfd(Delta_t_fluctuations_df):
	pivot_table_df = pandas.pivot_table(
		Delta_t_fluctuations_df,
		values = 'k MAD(Δt) (s)',
		index = 'k_1 (%)',
		columns = 'k_2 (%)',
		aggfunc = np.mean,
	)
	fig = go.Figure(
		data = go.Contour(
			z = pivot_table_df.to_numpy(),
			x = pivot_table_df.index,
			y = pivot_table_df.columns,
			contours = dict(
				coloring ='heatmap',
				showlabels = True, # show labels on contours
			),
			colorbar = dict(
				title = 'k MAD(Δt)',
				titleside = 'right',
			),
			hovertemplate = 'k<sub>1</sub>: %{y:.0f} %<br>k<sub>2</sub>: %{x:.0f} %<br>k MAD(Δt): %{z:.2e} s',
			name = '',
		),
	)
	k1_min, k2_min = find_best_k1_k2(Delta_t_fluctuations_df)
	fig.add_trace(
		go.Scatter(
			x = [k2_min],
			y = [k1_min],
			mode = 'markers',
			hovertext = [f'<b>Minimum</b><br>k<sub>1</sub>: {k1_min:.0f} %<br>k<sub>2</sub>: {k2_min:.0f} %<br>k MAD(Δt): {Delta_t_fluctuations_df.loc[(k1_min,k2_min),"k MAD(Δt) (s)"]*1e12:.2f} ps'],
			hoverinfo = 'text',
			marker = dict(
				color = '#61ff5c',
			),
			name = '',
		)
	)
	fig.update_yaxes(
		scaleanchor = "x",
		scaleratio = 1,
	)
	fig.update_layout(
		xaxis_title = 'k<sub>2</sub> (%)',
		yaxis_title = 'k<sub>1</sub> (%)',
	)
	return fig

def draw_median_and_MAD_vertical_lines(plotlyfig, center: float, amplitude: float, text: str):
	plotlyfig.add_vline(
		x = center,
		line_color = 'black',
	)
	for s in [-amplitude, amplitude]:
		plotlyfig.add_vline(
			x = center + s,
			line_color = 'black',
			line_dash = 'dash',
		)
	plotlyfig.add_annotation(
		x = center,
		y = .25,
		ax = center + amplitude,
		ay = .25,
		yref = 'y domain',
		showarrow = True,
		axref = "x", ayref='y',
		arrowhead = 3,
		arrowwidth = 1.5,
	)
	plotlyfig.add_annotation(
		x = center + amplitude,
		y = .25,
		ax = center,
		ay = .25,
		yref = 'y domain',
		showarrow = True,
		axref = "x", ayref='y',
		arrowhead = 3,
		arrowwidth = 1.5,
	)
	plotlyfig.add_annotation(
		text = text,
		ax = center + amplitude/2,
		ay = .27,
		x = center + amplitude/2,
		y = .27,
		yref = "y domain",
		axref = "x", ayref='y',
	)

def plot_Delta_t_histogram(Delta_t_df, k1, k2):
	fig = grafica.new(
		xlabel = 'Δt (s)',
		ylabel = 'Number of events',
	)
	data_to_plot = Delta_t_df.loc[(Delta_t_df['k_1 (%)']==k1)&(Delta_t_df['k_2 (%)']==k2), 'Delta_t (s)']
	fig.histogram(
		samples = data_to_plot,
	)
	draw_median_and_MAD_vertical_lines(
		plotlyfig = fig.plotly_figure, 
		median = np.median(data_to_plot), 
		MAD = median_abs_deviation(data_to_plot)
	)
	return fig

def fit_gaussian_to_samples(samples, bins='auto'):
	hist, bins_edges = np.histogram(
		samples, 
		bins = bins,
	)
	x_values = bins_edges[:-1] + np.diff(bins_edges)[0]/2
	y_values = hist
	try:
		popt, pcov = curve_fit(
			gaussian,
			x_values,
			y_values,
			p0 = [np.median(samples),median_abs_deviation(samples)*utils.k_MAD_TO_STD,max(y_values)],
		)
		return popt[0], popt[1], popt[2]
	except RuntimeError: # This happens when the fit fails because there are very few samples.
		return float('NaN'),float('NaN'),float('NaN')

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
			measured_data_df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.fd'))
		except FileNotFoundError:
			measured_data_df = pandas.read_csv(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.csv'))
		
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
		
		if len(set(measured_data_df['n_channel'])) != 2:
			raise RuntimeError(f'Expecting data from only two channels for a beta scan (DUT and reference) but this scan seems to have data from the following channels: {set(measured_data_df["Channel number"])}.')
		channels = list(set(measured_data_df['n_channel']))
		
		for idx,n_channel in enumerate(channels): # This is so I can use the same framework as in the TCT where there is only one detector but two pulses.
			measured_data_df.loc[measured_data_df['n_channel']==n_channel,'n_pulse'] = idx+1
		
		final_results_df = pandas.DataFrame()
		bootstrapped_replicas_df = pandas.DataFrame()
		for k_bootstrap in range(n_bootstrap+1):
			
			bootstrapped_iteration = False
			if k_bootstrap > 0:
				bootstrapped_iteration = True
			
			if bootstrapped_iteration == False:
				data_df = measured_data_df.copy()
			else:
				data_df = resample_measured_data(measured_data_df)
			
			Delta_t_df = calculate_Delta_t_df(data_df)
			Delta_t_fluctuations_df = calculate_Delta_t_fluctuations_df(Delta_t_df)
			best_k1k2 = find_best_k1_k2(Delta_t_fluctuations_df)
			fitted_mu, fitted_sigma, fitted_amplitude = fit_gaussian_to_samples(Delta_t_df.set_index(['k_1 (%)','k_2 (%)']).loc[best_k1k2,'Δt (s)'])
			std = Delta_t_df.set_index(['k_1 (%)','k_2 (%)']).loc[best_k1k2,'Δt (s)'].std()
			
			bootstrapped_replicas_df = bootstrapped_replicas_df.append(
				{
					'k MAD(Δt) (s)': Delta_t_fluctuations_df.loc[best_k1k2,'k MAD(Δt) (s)'],
					'std (s)': std,
					'sigma from Gaussian fit (s)': fitted_sigma,
					'k_1 (%)': best_k1k2[0],
					'k_2 (%)': best_k1k2[1],
				},
				ignore_index = True,
			)
			if bootstrapped_iteration == True:
				continue
			
			# If we are here it is because we are not in a bootstrap iteration, it is the first iteration that is with the actual data.
			final_results_df = final_results_df.append(
				{
					'k MAD(Δt) (s)': Delta_t_fluctuations_df.loc[best_k1k2,'k MAD(Δt) (s)'],
					'std (s)': std,
					'sigma from Gaussian fit (s)': fitted_sigma,
					'k_1 (%)': best_k1k2[0],
					'k_2 (%)': best_k1k2[1],
					'type': 'estimator value on the data',
				},
				ignore_index = True,
			)
			
			fig = plot_cfd(Delta_t_fluctuations_df)
			fig.update_layout(
				title = f'Time resolution vs CFD thresholds<br><sup>Measurement: {bureaucrat.measurement_name}</sup>'
			)
			fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'CFD_plot.html')), include_plotlyjs = 'cdn')
			
			fig = grafica.new(
				ylabel = 'Number of events',
				xlabel = 'Δt (s)',
			)
			samples_for_plot = np.array(list(Delta_t_df.loc[(Delta_t_df['k_1 (%)']==best_k1k2[0])&(Delta_t_df['k_2 (%)']==best_k1k2[1]),'Δt (s)']))
			fig.plotly_figure.add_trace(
				grafica.plotly_utils.utils.scatter_histogram(
					samples = samples_for_plot,
					name = f'Measured data',
					error_y = dict(type='auto'),
				)
			)
			x_axis_values = sorted(list(np.linspace(min(samples_for_plot),max(samples_for_plot),99)) + list(np.linspace(fitted_mu-5*fitted_sigma,fitted_mu+5*fitted_sigma,99)))
			fig.scatter(
				x = x_axis_values,
				y = gaussian(x_axis_values, fitted_mu, fitted_sigma, fitted_amplitude),
				label = f'Fitted Gaussian (σ={fitted_sigma*1e12:.2f} ps)',
			)
			draw_median_and_MAD_vertical_lines(
				plotlyfig = fig.plotly_figure,
				center = np.median(list(Delta_t_df.loc[(Delta_t_df['k_1 (%)']==best_k1k2[0])&(Delta_t_df['k_2 (%)']==best_k1k2[1]),'Δt (s)'])),
				amplitude = Delta_t_fluctuations_df.loc[best_k1k2,'k MAD(Δt) (s)'],
				text = f"k MAD(Δt) = {Delta_t_fluctuations_df.loc[best_k1k2,'k MAD(Δt) (s)']*1e12:.2f} ps",
			)
			fig.title = f'Δt for k1={best_k1k2[0]}, k2={best_k1k2[1]}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>'
			fig.save(file_name = str(bureaucrat.processed_data_dir_path/Path(f'histogram k1 {best_k1k2[0]} k2 {best_k1k2[1]}.html')))
		
		bootstrapped_replicas_df[['k_1 (%)','k_2 (%)']] = bootstrapped_replicas_df[['k_1 (%)','k_2 (%)']].astype(int)
		
		bootstrapped_replicas_df_file_path = bureaucrat.processed_data_dir_path/Path('bootstrap_results.csv')
		if bootstrapped_replicas_df_file_path.is_file():
			bootstrapped_replicas_df = bootstrapped_replicas_df.append(
				pandas.read_csv(bootstrapped_replicas_df_file_path),
				ignore_index = True,
			)
		bootstrapped_replicas_df.to_csv(bootstrapped_replicas_df_file_path, index=False)
		
		stuff_to_append = dict(bootstrapped_replicas_df.std())
		stuff_to_append['type'] = 'std of the bootstrapped replicas'
		final_results_df = final_results_df.append(
			stuff_to_append,
			ignore_index = True,
		)
		
		fig = grafica.new(
			title = f'Bootstrap replicas of k MAD(Δt)<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			xlabel = 'Estimation of σ (s)',
			ylabel = 'Number of events',
		)
		fig.histogram(
			samples = bootstrapped_replicas_df['k MAD(Δt) (s)'],
			label = 'k MAD(Δt)',
		)
		fig.histogram(
			samples = bootstrapped_replicas_df['sigma from Gaussian fit (s)'],
			label = 'sigma from Gaussian fit',
		)
		fig.histogram(
			samples = bootstrapped_replicas_df['std (s)'],
			label = 'std',
		)
		fig.save(file_name = str(bureaucrat.processed_data_dir_path/Path(f'histogram bootstrap.html')))
		
		
		final_results_df.to_csv(bureaucrat.processed_data_dir_path/Path('results.csv'), index=False)

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
	N_BOOTSTRAP = 33
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

