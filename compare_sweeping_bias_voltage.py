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

columns_to_group_by = ['Measurement name','Device','Pad','n_channel','n_pulse','Laser DAC','n_position']
averaged_data_df = measured_data.groupby(by = columns_to_group_by).mean()
averaged_data_df = averaged_data_df.reset_index()

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
	# ~ fig.update_layout(legend_orientation="h")
	fig.show()
