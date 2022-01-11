import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import measurements_table as mt
from calculate_interpixel_distance import script_core as calculate_interpixel_distance
import datetime

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

interpixel_distances_df = pandas.DataFrame()
for measurement_name in measurements_table_df.query('Type=="scan 1D"').index:
	try:
		this_measurement_IPD = utils.read_previously_calculated_inter_pixel_distance(measurement_name)
	except FileNotFoundError:
		this_measurement_IPD = float('NaN')
		print(f'Cannot find IPD for measurement {measurement_name}...')
	try:
		this_measurement_calibration_factor = utils.read_previously_calculated_distance_calibration_factor(measurement_name)
	except FileNotFoundError:
		this_measurement_calibration_factor = float('NaN')
		print(f'Cannot find distance calibration factor for measurement {measurement_name}...')
	try:
		this_measurement_belongs_to_the_voltage_scan = scans_and_sub_measurements_df.loc[measurement_name,'Voltage scan measurement name']
	except KeyError:
		this_measurement_belongs_to_the_voltage_scan = '?'
	interpixel_distances_df = interpixel_distances_df.append(
		{
			'Measurement name': measurement_name,
			'Measurement date': measurements_table_df.loc[measurement_name,'When'],
			'IPD (m)': this_measurement_IPD,
			'Distance calibration factor':this_measurement_calibration_factor,
			'Voltage scan measurement name': this_measurement_belongs_to_the_voltage_scan,
			'Fluence (neq/cm^2)/1e14': mt.get_measurement_fluence(measurement_name)/1e14,
		},
		ignore_index = True,
	)
interpixel_distances_df.set_index('Measurement name', inplace=True)
for col in {'Bias voltage (V)','Temperature (°C)','Can we trust?'}:
	interpixel_distances_df[col] = measurements_table_df[col]
interpixel_distances_df['Device'] = measurements_table_df['Measured device']
interpixel_distances_df.loc[interpixel_distances_df['Bias voltage (V)']=='?','Bias voltage (V)'] = 'NaN'
interpixel_distances_df['Bias voltage (V)'] = interpixel_distances_df['Bias voltage (V)'].astype(float)
interpixel_distances_df.reset_index(inplace=True)
interpixel_distances_df.set_index('Device', inplace=True)
interpixel_distances_df = interpixel_distances_df.join(utils.bureaucrat.devices_sheet_df)
interpixel_distances_df.set_index('Measurement name', inplace=True)
interpixel_distances_df.reset_index(inplace=True)

df = interpixel_distances_df.copy().reset_index()
df = df.query('`Can we trust?`=="yes"')
df['IPD with calibration (m)'] = df['IPD (m)']*df['Distance calibration factor']
df = df.sort_values(by=['Bias voltage (V)','trenches','trench depth'])
fig = utils.line(
	data_frame = df,
	line_group = 'Voltage scan measurement name',
	x = 'Bias voltage (V)',
	y = 'IPD with calibration (m)',
	facet_col = 'wafer',
	facet_row = 'trenches',
	text = 'Fluence (neq/cm^2)/1e14',
	color = 'trench depth',
	symbol = 'pixel border',
	line_dash = 'contact type',
	grouped_legend = True,
	hover_name = 'Measurement name',
	hover_data = ['Fluence (neq/cm^2)/1e14','Temperature (°C)'],
	labels = {
		'Collected charge (C) mean': 'Collected charge (C)',
		'Fluence (neq/cm^2)/1e14': 'fluence (n<sub>eq</sub>/cm<sup>2</sup>×10<sup>-14</sup>)',
		'IPD with calibration (m)': 'IPD (m)',
	},
	title = f'Inter pixel distsance vs bias voltage<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
)
fig.write_html(str(utils.path_to_scripts_output_directory/Path('ipd_vs_bias_voltage.html')), include_plotlyjs = 'cdn')

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
fig.write_html(str(utils.path_to_dashboard_media_directory/Path('ipd_vs_bias_voltage.html')), include_plotlyjs = 'cdn')
