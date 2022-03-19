from pathlib import Path
import measurements_table as mt
import utils
import pandas
import plotly.express as px
import datetime
from inter_pixel_distance_analysis import SORT_VALUES_BY, PLOT_GRAPH_DIMENSIONS, annealing_time_to_label_for_the_plot, PRELIMINARY_ANNOTATION

NICE_MEASUREMENTS = {
	# Non irradiated ---
	'20211114162042_#34_IV_curve',
	'20211029152446_#36_IV_curve',
	'20211101175030_#52_IV_curve',
	'20211108173804_#20_IV_curve',
	'20211031171456_#4_IV_curve',
	'20211109150804_#7_IV_curve',
	'20211102133011_#84_IV_curve',
	'20211025163034_#65_IV_curve',
	'20211030180803_#68_IV_curve',
	'20211111071526_#54_IV_curve',
	'20211110131445_#23_IV_curve',
	'20211103171051_#14_IV_curve',
	'20211113164217_#87_IV_curve',
	'20211112065952_#78_IV_curve',
	'20211104203325_#45_IV_curve_not_all_pads_grounded',
	# ~ '20220107162047_#111_IV_curve_RoomT',
	# ~ '20220107170700_#111_IV_curve_-20°C',
	# ~ # Irradiated devices ---
	# ~ '20211230165413_#68_IVCurve',
	# ~ '20220103121223_#6_IV_curve',
	# ~ '20220104153927_#69_IV_curve',
	# ~ '20220105155316_#70_IV_curve',
	# ~ '20211023104738_#45_IVCurve',
	# ~ '20220110135435_#77',
	# ~ '20220111174509_#78_IV_curve',
	# ~ '20220113142754_#79_IV_curve',
	# ~ '20220114140017_#36',
	# ~ '20220115173257_#37_IV_curve',
	# ~ '20220116134824_#38_IV_curve',
	# ~ '20220117112120_#23_IV_curve',
	# ~ '20220118152944_#24_IV_curve',
	# ~ '20220119162721_#47_IV_curve',
	# ~ '20220119165825_#47_sweeping_bias_voltage'
	# ~ '20220121145445__#77_IV_curve_after_annealing_7_days',
	# ~ '20220123205735_#78_after_annealing_7_days',
	# ~ '20220128182805_#93_IV_curve',
	# ~ '20220212150246_#84_IV_curve',
	# ~ '20220216110654_#52_IV_curve',
	# ~ '20220217191137_#53_IV_curve',
	# ~ '20220219135338_#51_IV_curve',
	# ~ '20220219151553_#51_sweeping_bias_voltage', # DELETEME #################################################################
}

measurements_table_df = mt.create_measurements_table()

NICE_MEASUREMENTS = NICE_MEASUREMENTS.union(set(measurements_table_df.query('Type=="scan 1D sweeping bias voltage"').index))

iv_data_df = pandas.DataFrame()
for measurement_name in NICE_MEASUREMENTS:
	try:
		if mt.retrieve_measurement_type(measurement_name) == 'IV curve':
			try:
				df = pandas.read_feather(utils.path_to_measurements_directory/Path(measurement_name)/Path('IV_curve/measured_data.fd'))
			except FileNotFoundError:
				df = pandas.read_csv(utils.path_to_measurements_directory/Path(measurement_name)/Path('IV_curve/measured_data.csv'))
			except Exception as e:
				raise e
			df = df.loc[:int(len(df)/2)] # Keep only the "sweeping to high voltages" from the dual sweep.
			df['Bias voltage (V)'] *= -1
			df['Bias current (A)'] *= -1
			df = utils.mean_std(
				df,
				by = 'n_voltage',
			)
			df = df.iloc[::7, :] # Remove some elements, otherwise the plots are too dense.
		elif mt.retrieve_measurement_type(measurement_name) == 'scan 1D sweeping bias voltage':
			if mt.get_measurement_fluence(measurement_name) == 0:
				continue
			df = pandas.read_feather(utils.path_to_measurements_directory/Path(measurement_name)/Path('create_iv_curve_from_1D_scan_data/iv_data.fd'))
			df = df.query('`Bias voltage (V) mean`>33') # This is to remove points where the detectors were already dead.
		else:
			raise RuntimeError(f'Measurement {repr(measurement_name)} is of type {repr(mt.retrieve_measurement_type(measurement_name))}, dont know how to read that...')
	except:
		continue
	df['Measurement name'] = measurement_name
	df['Device'] = measurements_table_df.loc[measurement_name, 'Measured device']
	
	
	iv_data_df = iv_data_df.append(df, ignore_index=True)

iv_data_df.set_index('Device', inplace=True)
for col in ['wafer','gain','trench process','trench depth','trenches','pixel border','contact type']:
	iv_data_df[col] = utils.bureaucrat.devices_sheet_df[col]
iv_data_df.reset_index(inplace=True)
iv_data_df.set_index('Measurement name', inplace=True)
for measurement_name in iv_data_df.index:
	iv_data_df.loc[measurement_name, 'Fluence (neq/cm^2)/1e14'] = str(int(mt.get_measurement_fluence(measurement_name)/1e14))
	iv_data_df.loc[measurement_name, 'Annealing time'] = mt.get_measurement_annealing_time(measurement_name)
	iv_data_df.loc[measurement_name, 'Temperature (°C)'] = mt.retrieve_measurement_temperature(measurement_name)
iv_data_df.reset_index(inplace=True)

SORT_VALUES_BY.remove('Bias voltage (V)') # This is because the `SORT_VALUES_BY` list is for laser scans but here we are dealing with IV curves.
SORT_VALUES_BY.append('n_voltage')
iv_data_df = iv_data_df.sort_values(
	by = SORT_VALUES_BY,
	ascending = True,
)

try:
	PLOT_GRAPH_DIMENSIONS['hover_data'].remove('Laser DAC')
except ValueError:
	pass

df = iv_data_df.copy()
df['Annealing time label'] = df['Annealing time'].apply(annealing_time_to_label_for_the_plot)
df = df.query('`Annealing time label`==""')
fig = utils.line(
	title = f'IV curves<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
	data_frame = df,
	x = 'Bias voltage (V) mean',
	y = 'Bias current (A) mean',
	error_y = 'Bias current (A) std',
	error_x = 'Bias voltage (V) std',
	line_group = 'Measurement name',
	**PLOT_GRAPH_DIMENSIONS,
)
fig.write_html(str(utils.path_to_scripts_output_directory/Path('iv_curves.html')), include_plotlyjs = 'cdn')

fig.add_annotation(PRELIMINARY_ANNOTATION)
fig.write_html(str(utils.path_to_dashboard_media_directory/Path('iv_curves.html')), include_plotlyjs = 'cdn')
