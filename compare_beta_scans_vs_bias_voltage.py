import pandas
import utils
import plotly.express as px
import numpy as np
from inter_pixel_distance_analysis import SORT_VALUES_BY, PLOT_GRAPH_DIMENSIONS, EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES, annealing_time_to_label_for_the_plot, COLORS_FOR_EACH_FLUENCE_DICT, PRELIMINARY_ANNOTATION
import measurements_table as mt
from pathlib import Path
import datetime
import warnings

try:
	PLOT_GRAPH_DIMENSIONS['hover_data'].remove('Laser DAC')
except ValueError:
	pass
try:
	PLOT_GRAPH_DIMENSIONS.pop('grouped_legend')
except ValueError:
	pass

measurements_table_df = mt.create_measurements_table()

time_resolution_df = pandas.DataFrame()
collected_charge_df = pandas.DataFrame()
for measurement_name in measurements_table_df.query("Type=='beta scan sweeping bias voltage'").index:
	_ = utils.path_to_measurements_directory/Path(measurement_name)/Path('create_beta_scan_time_resolution_vs_bias_voltage/time_resolution_vs_bias_voltage.csv')
	if _.is_file():
		df_for_time_resolution = pandas.read_csv(_)
	_ = utils.path_to_measurements_directory/Path(measurement_name)/Path('create_beta_scan_charge_vs_bias_voltage/collected_charge_vs_bias_voltage.csv')
	if _.is_file():
		df_for_collected_charge = pandas.read_csv(_)
	for df in [df_for_collected_charge,df_for_time_resolution]:
		df['Voltage scan measurement name'] = measurement_name
		df['Fluence (neq/cm^2)/1e14'] = mt.get_measurement_fluence(measurement_name)/1e14
		df['Annealing time'] = mt.get_measurement_annealing_time(measurement_name)
		df['Device'] = mt.retrieve_device_name(measurement_name)
		df['Temperature (°C)'] = float('NaN')
	time_resolution_df = time_resolution_df.append(df_for_time_resolution, ignore_index=True)
	collected_charge_df = collected_charge_df.append(df_for_collected_charge, ignore_index=True)

list_of_dataframes = [time_resolution_df, collected_charge_df]
for k in range(len(list_of_dataframes)):
	list_of_dataframes[k] = list_of_dataframes[k].set_index('Device')
	for column in {'pixel border','trenches','trench depth','contact type','wafer'}:
		list_of_dataframes[k][column] = utils.bureaucrat.devices_sheet_df[column]
	list_of_dataframes[k] = list_of_dataframes[k].reset_index()
	list_of_dataframes[k] = list_of_dataframes[k].sort_values(by=SORT_VALUES_BY)
	list_of_dataframes[k]['Annealing time label'] = list_of_dataframes[k]['Annealing time'].apply(annealing_time_to_label_for_the_plot)
time_resolution_df = list_of_dataframes[0]
collected_charge_df = list_of_dataframes[1]

fig = utils.line(
	data_frame = time_resolution_df,
	title = f'Time resolution (beta source) vs bias voltage<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
	x = 'Bias voltage (V)',
	y = 'Time resolution (s)',
	error_y = 'sigma from Gaussian fit (s) bootstrapped error estimation',
	# ~ error_y_mode = 'band',
	line_group = 'Voltage scan measurement name',
	color_discrete_map = COLORS_FOR_EACH_FLUENCE_DICT,
	**PLOT_GRAPH_DIMENSIONS,
)
fig.update_traces(
	error_y = dict(width=2, thickness=1),
)
fig.add_annotation(PRELIMINARY_ANNOTATION)
fig.write_html(
	str(utils.path_to_dashboard_media_directory/Path('beta_source_time_resolution_vs_bias_voltage.html')), 
	include_plotlyjs = 'cdn',
)

fig = utils.line(
	data_frame = collected_charge_df,
	title = f'Collected charge (beta source) vs bias voltage<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
	x = 'Bias voltage (V)',
	y = 'Collected charge (V s) x_mpv',
	line_group = 'Voltage scan measurement name',
	color_discrete_map = COLORS_FOR_EACH_FLUENCE_DICT,
	**PLOT_GRAPH_DIMENSIONS,
)
fig.add_annotation(PRELIMINARY_ANNOTATION)
fig.write_html(
	str(utils.path_to_dashboard_media_directory/Path('beta_source_collected_charge_vs_bias_voltage.html')), 
	include_plotlyjs = 'cdn',
)
