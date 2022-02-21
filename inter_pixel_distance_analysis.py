import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import measurements_table as mt
from calculate_interpixel_distance import script_core as calculate_interpixel_distance
import datetime
import numpy as np
from scipy.stats import median_abs_deviation
import plotly.io as pio

SORT_VALUES_BY = [
	'Fluence (neq/cm^2)/1e14',
	'trench depth',
	'trenches',
	'pixel border',
	'contact type',
	'Bias voltage (V)',
]

PLOT_GRAPH_DIMENSIONS = dict(
	facet_col = 'trench depth',
	facet_row = 'trenches',
	# ~ text = 'Fluence (neq/cm^2)/1e14',
	color = 'Fluence (neq/cm^2)/1e14',
	symbol = 'pixel border',
	line_dash = 'contact type',
	grouped_legend = True,
	hover_name = 'Measurement name',
	hover_data = ['Fluence (neq/cm^2)/1e14','Temperature (°C)','Laser DAC'],
	labels = {
		'Collected charge (C) mean': 'Collected charge (C)',
		'Fluence (neq/cm^2)/1e14': 'fluence (n<sub>eq</sub>/cm<sup>2</sup>×10<sup>-14</sup>)',
		'IPD (m) calibrated': 'IPD (m)',
		'Annealing time label': 'Annealing time (days)',
		'trench depth': 'Trench depth',
		'trenches': 'Trenches',
		'pixel border': 'Pixel border',
		'contact type': 'Contact type',
	},
	text = 'Annealing time label',
)

EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES = {
	'20211025184544_#65_sweeping_bias_voltage',
	'20211114164120_#34_sweeping_bias_voltage',
	'20220107190757_#111_sweeping_bias_voltage',
	'20220108211249_#111_sweeping_bias_voltage',
}

COLORS_FOR_EACH_FLUENCE_DICT = {0:'#7d8591',15:'#ffae21',25:'#ed4545',35:'#aa55d9'}

pio.templates[pio.templates.default].layout.colorway = [COLORS_FOR_EACH_FLUENCE_DICT[key] for key in sorted(COLORS_FOR_EACH_FLUENCE_DICT)] # https://community.plotly.com/t/changing-default-color-palette-in-plotly-go-python-sunburst/42758

def annealing_time_to_label_for_the_plot(annealing_time):
	return '' if pandas.isnull(annealing_time) or annealing_time < datetime.timedelta(1) else f'{annealing_time.days}'

if __name__ == '__main__':
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
		if measurement_name in scans_and_sub_measurements_df.index:
			if scans_and_sub_measurements_df.loc[measurement_name,'Voltage scan measurement name'] in EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES:
				continue
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
		try:
			bootstrapped_IPDs = np.genfromtxt(utils.path_to_measurements_directory/Path(measurement_name)/Path('calculate_interpixel_distance')/Path('interpixel_distance_bootstrapped_values.txt'))
			IPD_bootstrap_std = median_abs_deviation(bootstrapped_IPDs)
		except OSError:
			IPD_bootstrap_std = float('NaN')
			
		interpixel_distances_df = interpixel_distances_df.append(
			{
				'Measurement name': measurement_name,
				'Measurement date': measurements_table_df.loc[measurement_name,'When'],
				'IPD (m)': this_measurement_IPD,
				'IPD std bootstrap (m)': IPD_bootstrap_std,
				'Distance calibration factor':this_measurement_calibration_factor,
				'Voltage scan measurement name': this_measurement_belongs_to_the_voltage_scan,
				'Fluence (neq/cm^2)/1e14': mt.get_measurement_fluence(measurement_name)/1e14,
				'Annealing time': mt.get_measurement_annealing_time(measurement_name),
				'Laser DAC': mt.retrieve_laser_DAC(measurement_name),
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

	interpixel_distances_df = interpixel_distances_df.sort_values(
		by = SORT_VALUES_BY,
		ascending = True,
	)

	df = interpixel_distances_df.copy().reset_index()
	df = df.query('`Can we trust?`=="yes"')
	df['Annealing time label'] = df['Annealing time'].apply(annealing_time_to_label_for_the_plot)
	df = df.query('`Bias voltage (V)`>20')
	df = df.query('`Annealing time label`==""')
	for col in {'IPD (m)','IPD std bootstrap (m)'}:
		df[f'{col} calibrated'] = df[col]*df['Distance calibration factor']
	fig = utils.line(
		data_frame = df,
		title = f'Inter pixel distsance vs bias voltage<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
		x = 'Bias voltage (V)',
		y = 'IPD (m) calibrated',
		# ~ error_y = 'IPD std bootstrap (m) calibrated',
		# ~ error_y_mode = 'band',
		line_group = 'Voltage scan measurement name',
		**PLOT_GRAPH_DIMENSIONS,
	)
	fig.update_yaxes(range=[-5e-6,25e-6])

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
