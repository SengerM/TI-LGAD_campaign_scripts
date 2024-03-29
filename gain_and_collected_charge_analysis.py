import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import measurements_table as mt
from calculate_interpixel_distance import script_core as calculate_interpixel_distance
import datetime
from inter_pixel_distance_analysis import SORT_VALUES_BY, PLOT_GRAPH_DIMENSIONS, annealing_time_to_label_for_the_plot, EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES, PRELIMINARY_ANNOTATION
from scipy.interpolate import interp1d

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
	if measurement_name in scans_and_sub_measurements_df.index:
		if scans_and_sub_measurements_df.loc[measurement_name,'Voltage scan measurement name'] in EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES:
			continue
	try:
		_df = pandas.read_csv(utils.path_to_measurements_directory/Path(measurement_name)/Path('calculate_collected_charge_in_Coulomb/collected_charge_statistics.csv'))
	except FileNotFoundError as e:
		print(f'Cannot process {repr(measurement_name)}, reason: {repr(e)}.')
		continue
	try:
		this_measurement_belongs_to_the_voltage_scan = scans_and_sub_measurements_df.loc[measurement_name,'Voltage scan measurement name']
	except KeyError:
		this_measurement_belongs_to_the_voltage_scan = '?'
	collected_charge_df = collected_charge_df.append(
		{
			'Measurement name': measurement_name,
			'Measurement when': measurements_table_df.loc[measurement_name,'When'],
			'Voltage scan measurement name': this_measurement_belongs_to_the_voltage_scan,
			'Collected charge (C) mean': _df['Collected charge (C) median'].mean(),
			'Collected charge (C) std': _df['Collected charge (C) MAD_std'].mean(),
			'Collected charge (V s) mean': _df['Collected charge (V s) median'].mean(),
			'Collected charge (V s) std': _df['Collected charge (V s) MAD_std'].mean(),
			'Fluence (neq/cm^2)/1e14': mt.get_measurement_fluence(measurement_name)/1e14,
			'Annealing time': mt.get_measurement_annealing_time(measurement_name),
			'Laser DAC': mt.retrieve_laser_DAC(measurement_name),
			'Laser DAC': mt.retrieve_laser_DAC(measurement_name),
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

# ~ for col in {'Collected charge (C) mean', 'Collected charge (C) std', 'Collected charge (V s) mean', 'Collected charge (V s) std'}:
	# ~ indices = collected_charge_df['Laser DAC'] == 620
	# ~ collected_charge_df.loc[indices,col] = 620/573*collected_charge_df.loc[indices,col]

# Now calculate the gain using as a reference the PIN diode scan ---

PIN_DIODE_MEASUREMENT_NAME = '20220129160141_#93_sweeping_bias_voltage'

pin_diode_df = collected_charge_df.query(f'`Voltage scan measurement name`=="{PIN_DIODE_MEASUREMENT_NAME}"')
pin_diode_average_collected_charge_Vs_interpolation = interp1d(
	x = pin_diode_df['Bias voltage (V)'],
	y = pin_diode_df['Collected charge (V s) mean'],
	bounds_error = False,
)

collected_charge_df['Gain'] = collected_charge_df['Collected charge (V s) mean']/pin_diode_average_collected_charge_Vs_interpolation(collected_charge_df['Bias voltage (V)'])

collected_charge_df['trenches'] = collected_charge_df['trenches'].astype(int)

collected_charge_df = collected_charge_df.sort_values(
	by = SORT_VALUES_BY,
	ascending = True,
)

df = collected_charge_df.copy().reset_index()
df = df.query('`Can we trust?`=="yes"')
df['Annealing time label'] = df['Annealing time'].apply(annealing_time_to_label_for_the_plot)
df = df.query('`Bias voltage (V)`>20')
# ~ # For annealing devices toogle comment of these lines ---
df = df.query('`Annealing time label`==""')
# ~ df = df.query('`trench depth`=="D3"')
# ~ df = df.query('trenches==2')
# ~ df = df.query('`pixel border`=="V3"')
# ~ # -------------------------------------------------------
fig = utils.line(
	data_frame = df,
	x = 'Bias voltage (V)',
	y = 'Collected charge (C) mean',
	# ~ error_y = 'Collected charge (C) std',
	# ~ error_y_mode = 'band',
	title = f'Collected charge vs bias voltage<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
	line_group = 'Voltage scan measurement name',
	**PLOT_GRAPH_DIMENSIONS,
)
fig.write_html(str(utils.path_to_scripts_output_directory/Path('collected_charge_vs_bias_voltage.html')), include_plotlyjs = 'cdn')

fig.add_annotation(PRELIMINARY_ANNOTATION)
fig.write_html(str(utils.path_to_dashboard_media_directory/Path('collected_charge_vs_bias_voltage.html')), include_plotlyjs = 'cdn')



fig = utils.line(
	data_frame = df,
	x = 'Bias voltage (V)',
	y = 'Gain',
	title = f'Gain vs bias voltage<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
	line_group = 'Voltage scan measurement name',
	**PLOT_GRAPH_DIMENSIONS,
)
fig.add_hline(
	y = 1,
	line_dash = 'dash',
)
fig.update_yaxes(
	type = 'log',
	# ~ range = [0, 50e-12]
)
fig.write_html(str(utils.path_to_scripts_output_directory/Path('gain_vs_bias_voltage.html')), include_plotlyjs = 'cdn')

fig.add_annotation(PRELIMINARY_ANNOTATION)
fig.write_html(str(utils.path_to_dashboard_media_directory/Path('gain_vs_bias_voltage.html')), include_plotlyjs = 'cdn')
