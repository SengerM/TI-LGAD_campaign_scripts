from pathlib import Path
from measurements_table import create_measurements_table
import utils
import pandas
import plotly.express as px
import plotly.graph_objects as go

MEASUREMENTS_TO_COMPARE = [
	'20211022095225_#1_sweeping_bias_voltage',
	'20211026214321_#1_sweeping_bias_voltage',
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
interpixel_distances_df = pandas.DataFrame({'Measurement name': names_of_measurements_with_data_list})
interpixel_distances_df.set_index('Measurement name', inplace=True)
for measurement_name in names_of_measurements_with_data_list:
	this_measurement_data = utils.read_and_pre_process_1D_scan_data(measurement_name)
	this_measurement_data['Set bias voltage (V)'] = float(measurements_table_df.loc[measurement_name, 'Bias voltage (V)'])
	this_measurement_data['Laser DAC'] = measurements_table_df.loc[measurement_name, 'Laser DAC']
	this_measurement_data = utils.calculate_normalized_collected_charge(this_measurement_data)
	this_measurement_data = utils.calculate_distance_offset_by_linear_interpolation(this_measurement_data)
	this_measurement_data['Device specs'] = utils.get_device_specs_string(list(set(this_measurement_data['Device']))[0], humanize=False)
	this_measurement_data['Distance - offset by linear interpolation (m)'] = this_measurement_data['Distance (m)'] - this_measurement_data['Distance offset by linear interpolation (m)']
	measured_data = measured_data.append(this_measurement_data)
	# Calculate interpad distance and insert other info in the dataframe ---
	for column in {'Device','Device specs','Set bias voltage (V)'}:
		interpixel_distances_df.loc[measurement_name, column] = list(set(this_measurement_data[column]))[0]
	interpixel_distances_df.loc[measurement_name, 'Inter-pixel distance (m)'] = utils.calculate_interpixel_distance_by_linear_interpolation_using_normalized_collected_charge(this_measurement_data)['Inter-pixel distance (m)']

measured_data = measured_data.loc[measured_data['n_pulse']==1]

averaged_data_df = utils.mean_std(measured_data, by=['Measurement name','Device','Device specs','Pad','n_channel','n_pulse','n_position', 'Distance - offset by linear interpolation (m)','Set bias voltage (V)','Laser DAC'])

figs = []
for y in {'Normalized collected charge','Collected charge (V s)'}:
	fig = utils.line(
		data_frame = averaged_data_df,
		x = 'Distance - offset by linear interpolation (m)',
		y = y + ' mean',
		color = 'Device specs',
		symbol = 'Set bias voltage (V)',
		error_y = y + ' std',
		error_y_mode = 'bands',
		line_dash = 'Pad',
		color_discrete_sequence = px.colors.qualitative.D3,
		title = str(y),
		hover_name = 'Measurement name',
	)
	figs.append(fig)

left_mas_right_df = averaged_data_df.groupby(by = ['Measurement name','Set bias voltage (V)','Device','Device specs','n_pulse','Laser DAC','n_position','Distance - offset by linear interpolation (m)']).sum()
left_mas_right_df = left_mas_right_df.reset_index()
fig = utils.line(
	data_frame = left_mas_right_df,
	x = 'Distance - offset by linear interpolation (m)',
	y = 'Normalized collected charge mean',
	error_y = 'Normalized collected charge std',
	error_y_mode = 'band',
	color = 'Device specs',
	symbol = 'Set bias voltage (V)',
	markers = True,
	color_discrete_sequence = px.colors.qualitative.D3,
	title = 'Left pad + right pad',
	hover_name = 'Measurement name',
)
figs.append(fig)

for idx,fig in enumerate(figs):
	for x in [-125e-6, 0, 125e-6]:
		fig.add_shape(type="line",
			yref = "paper",
			x0 = x,
			y0 = 0,
			x1 = x,
			y1 = 1,
			line = dict(
				color = 'black',
				dash = 'dash',
			),
		)

fig = px.scatter(
	data_frame = interpixel_distances_df.reset_index(),
	x = 'Set bias voltage (V)',
	y = 'Inter-pixel distance (m)',
	color = 'Device specs',
	hover_name = 'Measurement name',
	title = 'Inter-pixel distance vs bias voltage using normalized collected charge with linear interpolation',
)
figs.append(fig)

for idx,fig in enumerate(figs):
	fig.show()
	fig.write_html(f'figure_{idx}.html', include_plotlyjs='cdn')
