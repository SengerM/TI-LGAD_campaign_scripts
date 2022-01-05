import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import measurements_table
from calculate_interpixel_distance import script_core as calculate_interpixel_distance

measurements_table_df = measurements_table.create_measurements_table()

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
interpixel_distances_df['Fluence (neq/cm^2)'] = interpixel_distances_df.loc[interpixel_distances_df['irradiation date']<interpixel_distances_df['Measurement date'],'neutrons (neq/cm^2×10e14)']
interpixel_distances_df.reset_index(inplace=True)

print(sorted(interpixel_distances_df.columns))

df = interpixel_distances_df.copy().reset_index()
df = df.query('`Can we trust?`=="yes"')
df = df[~df['Voltage scan measurement name'].isin(
	{
		'20211024163128_#65_sweeping_bias_voltage', # This measurement did not have all the pads grounded.
		'20211025184544_#65_sweeping_bias_voltage', # This is an old measurement that is incomplete in voltages. Now I have a new one with all the voltages.
		'20211023190308_#77_sweeping_bias_voltage', # Not all pads were grounded and results look weird.
	}
)]
df['IPD with calibration (m)'] = df['IPD (m)']*df['Distance calibration factor']
df = df.sort_values(by=['Bias voltage (V)','trenches','trench depth'])
def create_text_for_plot(row):
	T = row['Temperature (°C)']
	if not isinstance(T, float):
		T = ''
	return f'{row["Fluence (neq/cm^2)"]}, {T}'
df['text'] = df.apply(create_text_for_plot, axis=1)
fig = utils.line(
	data_frame = df,
	line_group = 'Voltage scan measurement name',
	x = 'Bias voltage (V)',
	y = 'IPD with calibration (m)',
	facet_col = 'wafer',
	facet_row = 'trenches',
	text = 'Fluence (neq/cm^2)',
	color = 'trench depth',
	symbol = 'pixel border',
	line_dash = 'contact type',
	grouped_legend = True,
	hover_name = 'Measurement name',
	labels = {
		'IPD with calibration (m)': 'IPD (m)',
	},
)
fig.show()
fig.write_html(str(utils.path_to_scripts_output_directory/Path('ipd_vs_bias_voltage.html')), include_plotlyjs = 'cdn')
