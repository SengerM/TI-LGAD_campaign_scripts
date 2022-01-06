import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def do_k1_k2_colormap_plots_with_position_slider(Delta_t_std_df):
	k1_k2_time_resolution_table = pandas.pivot_table(
		Delta_t_std_df,
		values = 'Delta_t std (s)',
		index = ['n_position','Pad','k_1 (%)'],
		columns = ['k_2 (%)'],
	)
	figures = {
		'left': go.Figure(),
		'right': go.Figure(),
	}
	for pad in {'left','right'}:
		for n_position in sorted(set(Delta_t_std_df.reset_index()['n_position'])):
			# ~ px.imshow(k1_k2_time_resolution_table.loc[(n_position,pad,)]).show() # Reference figure.
			df = k1_k2_time_resolution_table.loc[(n_position,pad)]
			figures[pad].add_trace(
				go.Heatmap(
					z = df.to_numpy(),
					x = df.index,
					y = df.columns,
					hovertemplate =
					'k<sub>1</sub> (%): %{y}<br>' +
					'k<sub>2</sub> (%): %{x}<br>' +
					'std(t<sub>1</sub>-t<sub>2</sub>) (s): %{z}',
					visible = False,
				)
			)
		steps = []
		for i in range(len(figures[pad].data)):
			step = dict(
				method="update",
				args=[{"visible": [False] * len(figures[pad].data)},
				{"title": "Slider switched to step: " + str(i)}],  # layout attribute
			)
			step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
			steps.append(step)
		sliders = [dict(
			active=0,
			currentvalue={"prefix": "n_position: "},
			pad={"t": 50},
			steps=steps
		)]
		figures[pad].update_layout(sliders=sliders)
		figures[pad].update_layout(
			xaxis_title = 'k<sub>2</sub> (%)',
			yaxis_title = 'k<sub>1</sub> (%)',
		)
		figures[pad].update_yaxes(
			scaleanchor = "x",
			scaleratio = 1,
		)
	for pad in figures:
		figures[pad].show()

def script_core(measurement_name: str):
	if not mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
		raise ValueError(f'Measurement must be a `scan 1D` but measurement named {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
	measured_data_df.set_index(['n_position','n_trigger','Pad','Distance (m)'], inplace=True)
	pulse_1_df = measured_data_df.query('n_pulse==1')
	pulse_2_df = measured_data_df.query('n_pulse==2')
	
	Delta_t_df = pandas.DataFrame()
	for k1 in [10,20,30,40,50,60,70,80,90]:
		for k2 in [10,20,30,40,50,60,70,80,90]:
			temp_df = pandas.DataFrame()
			temp_df['Delta_t (s)'] = pulse_1_df[f't_{k1} (s)'] - pulse_2_df[f't_{k2} (s)']
			temp_df['k_1 (%)'] = k1
			temp_df['k_2 (%)'] = k2
			temp_df.reset_index(inplace=True)
			temp_df.set_index(['n_position','n_trigger','Pad','Distance (m)','k_1 (%)','k_2 (%)'], inplace=True)
			Delta_t_df = Delta_t_df.append(temp_df)
	Delta_t_df.reset_index(inplace=True)
	Delta_t_std_df = Delta_t_df.groupby(by=['n_position','Pad','Distance (m)','k_1 (%)','k_2 (%)']).std().reset_index()
	Delta_t_std_df.rename(columns={'Delta_t (s)': 'Delta_t std (s)'}, inplace=True)
	Delta_t_std_df['Time resolution (s)'] = Delta_t_std_df['Delta_t std (s)']/2**.5
	
	Delta_t_std_df.reset_index(drop=True).to_feather(bureaucrat.processed_data_dir_path/Path('time_resolution.fd'))
	
	k1 = 50
	k2 = 50
	fig = utils.line(
		data_frame = Delta_t_std_df.loc[(Delta_t_std_df['k_1 (%)']==k1)&(Delta_t_std_df['k_2 (%)']==k2)],
		x = 'Distance (m)',
		y = 'Time resolution (s)',
		color = 'Pad',
		log_y = True,
		labels = {
			'Time resolution (s)': 'Time resolution (s) (σ<sub>Δt</sub>/√2)',
		},
		title = f'Time resolution @ k<sub>1</sub>={k1} %, k<sub>2</sub>={k2} %<br><sup>Measurement name: {bureaucrat.measurement_name}</sup>',
	)
	fig.write_html(str(bureaucrat.processed_data_dir_path)/Path('time_resolution_vs_distance.html'), include_plotlyjs = 'cdn')

if __name__ == '__main__':
	import argparse
	import measurements_table as mt
	
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
		script_core(Path(args.directory).parts[-1])
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			if mt.retrieve_measurement_type(measurement_name) == 'scan 1D' and not (utils.path_to_measurements_directory/Path(measurement_name)/Path('time_resolution_analysis')/Path('time_resolution.fd')).is_file():
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason: {repr(e)}.')

