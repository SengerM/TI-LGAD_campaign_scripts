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

def script_core(measurement_name: str, force=False):
	if not mt.retrieve_measurement_type(measurement_name) == 'beta scan':
		raise ValueError(f'Measurement must be a `beta scan` but measurement named {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_flag:
		return
	
	with bureaucrat.verify_no_errors_context():
		try:
			measured_data_df = pandas.read_feather(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.fd'))
		except FileNotFoundError:
			measured_data_df = pandas.read_csv(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('measured_data.csv'))
		
		if len(set(measured_data_df['n_channel'])) != 2:
			raise RuntimeError(f'Expecting data from only two channels for a beta scan (DUT and reference) but this scan seems to have data from the following channels: {set(measured_data_df["Channel number"])}.')
		channels = list(set(measured_data_df['n_channel']))
		
		bootstrapped_replicas_df = pandas.DataFrame()
		for k_bootstrap in range(5):
			bootstrapped_iteration = False
			if k_bootstrap > 0:
				bootstrapped_iteration = True
			
			if bootstrapped_iteration == False:
				data_df = measured_data_df.copy()
			else:
				data_df = resample_measured_data(measured_data_df)
			
			data_df.set_index('n_trigger', inplace=True)
			
			pulse_1_df = data_df.query(f'n_channel=={channels[0]}')
			pulse_2_df = data_df.query(f'n_channel=={channels[1]}')
			
			Delta_t_df = pandas.DataFrame()
			for k1 in [10,20,30,40,50,60,70,80,90]:
				for k2 in [10,20,30,40,50,60,70,80,90]:
					temp_df = pandas.DataFrame()
					temp_df['Delta_t (s)'] = pulse_1_df[f't_{k1} (s)'] - pulse_2_df[f't_{k2} (s)']
					temp_df['k_1 (%)'] = k1
					temp_df['k_2 (%)'] = k2
					temp_df.reset_index(inplace=True)
					temp_df.set_index(['n_trigger','k_1 (%)','k_2 (%)'], inplace=True)
					Delta_t_df = Delta_t_df.append(temp_df)
			Delta_t_df.reset_index(inplace=True)
			Delta_t_df = Delta_t_df.dropna()
			Delta_t_std_df = Delta_t_df.groupby(by=['k_1 (%)','k_2 (%)']).agg(median_abs_deviation).reset_index()
			Delta_t_std_df.drop('n_trigger', axis=1, inplace=True)
			Delta_t_std_df.rename(columns={'Delta_t (s)': 'Delta_t std (s)'}, inplace=True)
			
			sigma_minimum = Delta_t_std_df['Delta_t std (s)'].min()
			k1_min = list(Delta_t_std_df.loc[Delta_t_std_df['Delta_t std (s)']==sigma_minimum,'k_1 (%)'])[0]
			k2_min = list(Delta_t_std_df.loc[Delta_t_std_df['Delta_t std (s)']==sigma_minimum,'k_2 (%)'])[0]
			
			if bootstrapped_iteration == True:
				bootstrapped_replicas_df = bootstrapped_replicas_df.append(
					{
						'sigma_Delta_t (s)': sigma_minimum,
						'k_1 (%)': k1_min,
						'k_2 (%)': k2_min,
					},
					ignore_index = True,
				)
				continue
			
			with open(bureaucrat.processed_data_dir_path/Path('final_result.txt'), 'w') as ofile:
				print(f'σ<sub>Δt</sub> (s) = {sigma_minimum}', file=ofile)
				for idx,k in enumerate([k1_min, k2_min]):
					print(f'constant fraction discriminator k_{idx+1} (%) = {k:.0f}', file=ofile)
			
			pivot_table_df = pandas.pivot_table(
				Delta_t_std_df,
				values = 'Delta_t std (s)',
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
						title = 'σ<sub>Δt</sub>',
						titleside = 'right',
					),
					hovertemplate = 'k<sub>1</sub>: %{x:.0f} %<br>k<sub>2</sub>: %{y:.0f} %<br>σ<sub>Δt</sub>: %{z:.1e} s',
					name = '',
				),
			)
			fig.add_trace(
				go.Scatter(
					x = [k2_min],
					y = [k1_min],
					mode = 'markers',
					hovertext = [f'<b>Minimum</b><br>k<sub>1</sub>: {k1_min:.0f} %<br>k<sub>2</sub>: {k2_min:.0f} %<br>Minimum value: {sigma_minimum*1e12:.2f} ps'],
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
				title = dict(
					text = f'Time resolution vs CFD thresholds<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
				),
			)
			fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'CFD_plot.html')), include_plotlyjs = 'cdn')
			
			histograms_path = bureaucrat.processed_data_dir_path
			histograms_path.mkdir(parents=True, exist_ok=True)
			for k1 in {k1_min}:#sorted(set(Delta_t_df['k_1 (%)'])):
				for k2 in {k2_min}:#sorted(set(Delta_t_df['k_2 (%)'])):
					fig = grafica.new(
						title = f'Δt for k1={k1} %, k2={k2}<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
						xlabel = 'Δt (s)',
						ylabel = 'Number of events',
					)
					data_to_plot = Delta_t_df.loc[(Delta_t_df['k_1 (%)']==k1)&(Delta_t_df['k_2 (%)']==k2), 'Delta_t (s)']
					fig.histogram(
						samples = data_to_plot,
					)
					plotlyfig = fig.plotly_figure
					median = np.median(data_to_plot)
					plotlyfig.add_vline(
						x = median,
						annotation_text = 'Median',
						line_color = 'black',
					)
					MAD = median_abs_deviation(data_to_plot)
					for s in [-MAD, MAD]:
						plotlyfig.add_vline(
							x = median + s,
							line_color = 'black',
							line_dash = 'dash',
						)
					plotlyfig.add_annotation(
						x = median,
						y = .25,
						ax = median + MAD,
						ay = .25,
						yref = 'y domain',
						showarrow = True,
						axref = "x", ayref='y',
						arrowhead = 3,
						arrowwidth = 1.5,
					)
					plotlyfig.add_annotation(
						x = median + MAD,
						y = .25,
						ax = median,
						ay = .25,
						yref = 'y domain',
						showarrow = True,
						axref = "x", ayref='y',
						arrowhead = 3,
						arrowwidth = 1.5,
					)
					plotlyfig.add_annotation(
						text = f'MAD = {MAD*1e12:.2f} ps',
						ax = median + MAD/2,
						ay = .27,
						x = median + MAD/2,
						y = .27,
						yref = "y domain",
						axref = "x", ayref='y',
					)
					fig.save(file_name = str(histograms_path/Path(f'histogram k1 {k1} k2 {k2}.html')))
		
		bootstrapped_replicas_df[['k_1 (%)','k_2 (%)']] = bootstrapped_replicas_df[['k_1 (%)','k_2 (%)']].astype(int)
		bootstrapped_replicas_df.to_csv(bureaucrat.processed_data_dir_path/Path('bootstrap_results.csv'))
		
		
		
		fig = grafica.new(
			title = f'Bootstrap replicas of σ<sub>Δt</sub><br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			xlabel = 'σ<sub>Δt</sub> (s)',
			ylabel = 'Number of events',
		)
		fig.histogram(
			samples = bootstrapped_replicas_df['sigma_Delta_t (s)'],
		)
		
		
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
			if mt.retrieve_measurement_type(measurement_name) == 'beta scan':
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason: {repr(e)}.')

