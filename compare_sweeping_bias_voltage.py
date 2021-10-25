from pathlib import Path
from measurements_table import create_measurements_table
import utils
import pandas
import plotly.express as px
import plotly.graph_objects as go

MEASUREMENTS_TO_COMPARE = [
	'20211022095225_#1_sweeping_bias_voltage',
	'20211023000338_#45_sweeping_bias_voltage',
	'20211023190308_#77_sweeping_bias_voltage',
	'20211024163128_#65_sweeping_bias_voltage',
	'20211025184544_#65_sweeping_bias_voltage',
]

measurements_table_df = create_measurements_table()

names_of_measurements_with_data_list = []
for root_measurement_name in MEASUREMENTS_TO_COMPARE:
	with open(utils.path_to_measurements_directory/Path(root_measurement_name)/Path('scan_1D_sweeping_bias_voltage/README.txt'), 'r') as ifile:
		for idx, line in enumerate(ifile):
			if idx == 0:
				continue
			names_of_measurements_with_data_list.append(line.replace('\n',''))

measured_data = pandas.DataFrame()
for measurement_name in names_of_measurements_with_data_list:
	this_measurement_data = utils.read_and_pre_process_1D_scan_data(measurement_name)
	this_measurement_data['Device'] = measurements_table_df.loc[measurement_name, 'Measured device']
	this_measurement_data['Set bias voltage (V)'] = float(measurements_table_df.loc[measurement_name, 'Bias voltage (V)'])
	this_measurement_data['Laser DAC'] = measurements_table_df.loc[measurement_name, 'Laser DAC']
	this_measurement_data = utils.calculate_normalized_collected_charge(this_measurement_data)
	this_measurement_data = utils.calculate_distance_offset_by_linear_interpolation(this_measurement_data)
	this_measurement_data['Distance - offset by linear interpolation (m)'] = this_measurement_data['Distance (m)'] - this_measurement_data['Distance offset by linear interpolation (m)']
	measured_data = measured_data.append(this_measurement_data)

columns_to_group_by = ['Measurement name','Device','Pad','n_channel','n_pulse','n_position']
averaged_data_df = pandas.DataFrame()
for measurement_name in set(measured_data['Measurement name']):
	averaged_data_df = averaged_data_df.append(
		utils.calculate_mean_measured_values_at_each_position(measured_data[measured_data['Measurement name']==measurement_name], by=columns_to_group_by),
		ignore_index = True,
	)

for y in {'Normalized collected charge','Collected charge (V s)'}:
	fig = px.line(
		data_frame = averaged_data_df.loc[averaged_data_df['n_pulse']==1],
		x = 'Distance - offset by linear interpolation (m)',
		y = y,
		color = 'Device',
		symbol = 'Set bias voltage (V)',
		markers = True,
		color_discrete_sequence = px.colors.qualitative.D3,
	)
	fig.show()

left_mas_right_df = averaged_data_df.groupby(by = ['Measurement name','Set bias voltage (V)','Device','n_pulse','Laser DAC','n_position']).sum()
left_mas_right_df = left_mas_right_df.reset_index()
fig = utils.line(
	data_frame = left_mas_right_df.loc[left_mas_right_df['n_pulse']==1],
	x = 'Distance - offset by linear interpolation (m)',
	y = 'Normalized collected charge',
	error_y = 'Normalized collected charge std',
	error_y_mode = 'band',
	color = 'Device',
	symbol = 'Set bias voltage (V)',
	markers = True,
	color_discrete_sequence = px.colors.qualitative.D3,
)
fig.show()
