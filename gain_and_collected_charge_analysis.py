import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import measurements_table as mt
from calculate_interpixel_distance import script_core as calculate_interpixel_distance

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

collected_charge_df = pandas.DataFrame()
for measurement_name in measurements_table_df.query('Type=="scan 1D"').index:
	_df = pandas.read_csv(utils.path_to_measurements_directory/Path(measurement_name)/Path('calculate_collected_charge_in_Coulomb/collected_charge_statistics.csv'))
	try:
		this_measurement_belongs_to_the_voltage_scan = scans_and_sub_measurements_df.loc[measurement_name,'Voltage scan measurement name']
	except KeyError:
		this_measurement_belongs_to_the_voltage_scan = '?'
	collected_charge_df = collected_charge_df.append(
		{
			'Measurement name': measurement_name,
			'Measurement date': measurements_table_df.loc[measurement_name,'When'],
			'Voltage scan measurement name': this_measurement_belongs_to_the_voltage_scan,
			'Collected charge (C) mean': _df['Collected charge (C) mean'].mean(),
			'Collected charge (C) std': _df['Collected charge (C) std'].mean(),
			'Fluence (neq/cm^2)/1e14': mt.get_measurement_fluence(measurement_name)/1e14,
		},
		ignore_index = True,
	)

collected_charge_df.set_index('Measurement name', inplace=True)
for col in {'Bias voltage (V)','Temperature (°C)','Can we trust?','Measured device'}:
	collected_charge_df[col] = measurements_table_df[col]
collected_charge_df.loc[collected_charge_df['Bias voltage (V)']=='?','Bias voltage (V)'] = 'NaN'
collected_charge_df['Bias voltage (V)'] = collected_charge_df['Bias voltage (V)'].astype(float)
collected_charge_df.reset_index(inplace=True)
collected_charge_df.set_index('Measured device', inplace=True)
collected_charge_df = collected_charge_df.join(utils.bureaucrat.devices_sheet_df)
collected_charge_df.set_index('Measurement name', inplace=True)
collected_charge_df.reset_index(inplace=True)

df = collected_charge_df.copy().reset_index()
df = df.query('`Can we trust?`=="yes"')
df = df.sort_values(by=['Bias voltage (V)','trenches','trench depth'])
fig = utils.line(
	data_frame = df,
	line_group = 'Voltage scan measurement name',
	x = 'Bias voltage (V)',
	y = 'Collected charge (C) mean',
	error_y = 'Collected charge (C) std',
	error_y_mode = 'band',
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
	},
)
fig.show()
