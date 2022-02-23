import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import measurements_table as mt
import datetime
from inter_pixel_distance_analysis import SORT_VALUES_BY, PLOT_GRAPH_DIMENSIONS, annealing_time_to_label_for_the_plot, EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES

measurements_table_df = mt.create_measurements_table()

# For each of the "scan 1D seeping bias voltage" I collect all the single voltage 1D scans that were created in such "voltage sweep scan".
scans_and_sub_measurements_df = pandas.DataFrame()
for root_measurement_name in measurements_table_df.query('Type=="scan 1D sweeping bias voltage"').index:
	with open(utils.path_to_measurements_directory/Path(root_measurement_name)/Path('scan_1D_sweeping_bias_voltage/README.txt'), 'r') as ifile:
		for idx, line in enumerate(ifile):
			if idx == 0:
				continue
			scans_and_sub_measurements_df = scans_and_sub_measurements_df.append(
				{
					'Scan name': line.replace('\n',''), 
					'Voltage scan measurement name': root_measurement_name
				}, 
				ignore_index=True
			)
scans_and_sub_measurements_df.set_index('Scan name', inplace=True)

time_resolution_df = pandas.DataFrame()
for measurement_name in scans_and_sub_measurements_df.index:
	if measurement_name in scans_and_sub_measurements_df.index:
		if scans_and_sub_measurements_df.loc[measurement_name,'Voltage scan measurement name'] in EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES:
			continue
	time_resolution_results_file_path = utils.path_to_measurements_directory/Path(measurement_name)/Path('calculate_time_resolution/final_result.txt')
	if not time_resolution_results_file_path.is_file():
		print(f'Cannot find time resolution calculation for measurement "{measurement_name}". Missing file is {time_resolution_results_file_path}.')
		print(f'Will skip it...')
		continue
	with open(time_resolution_results_file_path, 'r') as ifile:
		this_measurement_time_resolution = None
		this_measurement_k1 = None
		this_measurement_k2 = None
		for line in ifile:
			if 'time resolution (s) =' in line:
				this_measurement_time_resolution = float(line.split('=')[-1])
			if 'constant fraction discriminator k_1 (%) =' in line:
				this_measurement_k1 = int(line.split('=')[-1])
			if 'constant fraction discriminator k_2 (%) =' in line:
				this_measurement_k2 = int(line.split('=')[-1])
	time_resolution_df = time_resolution_df.append(
		{
			'Time resolution (s)': this_measurement_time_resolution,
			'k_1 (%)': this_measurement_k1,
			'k_2 (%)': this_measurement_k2,
			'Measurement name': measurement_name,
			'Fluence (neq/cm^2)/1e14': mt.get_measurement_fluence(measurement_name)/1e14,
			'Annealing time': mt.get_measurement_annealing_time(measurement_name),
			'Laser DAC': mt.retrieve_laser_DAC(measurement_name),
		},
		ignore_index = True,
	)

time_resolution_df.set_index('Measurement name', inplace=True)
time_resolution_df['Voltage scan measurement name'] = scans_and_sub_measurements_df['Voltage scan measurement name']
for column in measurements_table_df.columns:
	time_resolution_df[column] = measurements_table_df[column]
time_resolution_df.reset_index(inplace=True)
time_resolution_df.set_index('Measured device', inplace=True)
for column in utils.bureaucrat.devices_sheet_df.columns:
	time_resolution_df[column] = utils.bureaucrat.devices_sheet_df[column]

time_resolution_df = time_resolution_df.sort_values(
	by = SORT_VALUES_BY,
	ascending = True,
)

df = time_resolution_df.reset_index()
df = df.query('`Can we trust?`=="yes"')
df['Annealing time label'] = df['Annealing time'].apply(annealing_time_to_label_for_the_plot)
df = df.query('`Bias voltage (V)`>20')
df = df.query('`Annealing time label`==""')
fig = utils.line(
	data_frame = df,
	line_group = 'Voltage scan measurement name',
	x = 'Bias voltage (V)',
	y = 'Time resolution (s)',
	title = f'Time resolution (TCT) vs bias voltage<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
	**PLOT_GRAPH_DIMENSIONS,
)
# ~ fig.update_yaxes(type='log',)
fig.write_html(str(utils.path_to_scripts_output_directory/Path('time_resolution_vs_bias_voltage.html')), include_plotlyjs = 'cdn')
fig.add_hline(
	y = 30e-12,
	line_dash = "dash",
	# ~ annotation_text = '30 ps',
)
fig.add_annotation(
	dict(
		name="draft watermark",
		text="PRELIMINARY",
		textangle=-30,
		opacity=0.1,
		font=dict(color="black", size=100),
		xref="paper",
		yref="paper",
		x=0.5,
		y=0.5,
		showarrow=False,
	)
)
fig.write_html(str(utils.path_to_dashboard_media_directory/Path('time_resolution_vs_bias_voltage.html')), include_plotlyjs = 'cdn')
