from pathlib import Path
import measurements_table as mt
import utils
import pandas
import plotly.express as px
import datetime
from inter_pixel_distance_analysis import SORT_VALUES_BY, PLOT_GRAPH_DIMENSIONS

NICE_MEASUREMENTS = {
	'20211114162042_#34_IV_curve',
	'20211029152446_#36_IV_curve',
	'20211101175030_#52_IV_curve',
	'20211108173804_#20_IV_curve',
	'20211031171456_#4_IV_curve',
	'20211109150804_#7_IV_curve',
	'20220107162047_#111_IV_curve_RoomT',
	'20220107170700_#111_IV_curve_-20°C',
	'20211102133011_#84_IV_curve',
	'20211025163034_#65_IV_curve',
	'20211030180803_#68_IV_curve',
	'20211111071526_#54_IV_curve',
	'20211110131445_#23_IV_curve',
	'20211103171051_#14_IV_curve',
	'20211113164217_#87_IV_curve',
	'20211112065952_#78_IV_curve',
	'20211230165413_#68_IVCurve',
	'20220103121223_#6_IV_curve',
	'20220104153927_#69_IV_curve',
	'20220105155316_#70_IV_curve',
	'20211023104738_#45_IVCurve',
	'20220110135435_#77',
	'20220111174509_#78_IV_curve',
	'20220113142754_#79_IV_curve',
	'20220114140017_#36',
	'20220115173257_#37_IV_curve',
	'20220116134824_#38_IV_curve',
	'20220117112120_#23_IV_curve',
	'20220118152944_#24_IV_curve',
	'20220119162721_#47_IV_curve',
	'20220121145445__#77_IV_curve_after_annealing_7_days',
	'20220123205735_#78_after_annealing_7_days',
}

IV_measurements_table_df = mt.create_measurements_table().query('Type=="IV curve"')

measured_data_df = pandas.DataFrame()
for measurement_name in IV_measurements_table_df.index:
	if 'quick' in measurement_name:
		continue
	if measurement_name not in NICE_MEASUREMENTS:
		continue
	try:
		df = pandas.read_feather(utils.path_to_measurements_directory/Path(measurement_name)/Path('IV_curve/measured_data.fd'))
	except FileNotFoundError:
		df = pandas.read_csv(utils.path_to_measurements_directory/Path(measurement_name)/Path('IV_curve/measured_data.csv'))
	except Exception as e:
		raise e
	df['Bias voltage (V)'] *= -1
	df['Bias current (A)'] *= -1
	df['Measurement name'] = measurement_name
	device_name = IV_measurements_table_df.loc[measurement_name, 'Measured device']
	df['Device'] = device_name
	measured_data_df = measured_data_df.append(df, ignore_index=True)
measured_data_df = measured_data_df.loc[:, ~measured_data_df.columns.str.contains('^Unnamed')] # Remove this artifact from Excel.

mean_std_df = utils.mean_std(
	measured_data_df,
	by = ['Device','n_voltage','Measurement name'],
)

mean_std_df.set_index('Device', inplace=True)
for col in ['wafer','gain','trench process','trench depth','trenches','pixel border','contact type']:
	mean_std_df[col] = utils.bureaucrat.devices_sheet_df[col]
mean_std_df.reset_index(inplace=True)
mean_std_df.set_index('Measurement name', inplace=True)
for measurement_name in mean_std_df.index:
	mean_std_df.loc[measurement_name, 'Fluence (neq/cm^2)/1e14'] = str(int(mt.get_measurement_fluence(measurement_name)/1e14))
	mean_std_df.loc[measurement_name, 'Temperature (°C)'] = mt.retrieve_measurement_temperature(measurement_name)
mean_std_df.reset_index(inplace=True)

SORT_VALUES_BY.remove('Bias voltage (V)') # This is because the `SORT_VALUES_BY` list is for laser scans but here we are dealing with IV curves.
SORT_VALUES_BY.append('n_voltage')
mean_std_df = mean_std_df.sort_values(
	by = SORT_VALUES_BY,
	ascending = True,
)

fig = utils.line(
	title = f'IV curves<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
	data_frame = mean_std_df,
	x = 'Bias voltage (V) mean',
	y = 'Bias current (A) mean',
	error_y = 'Bias current (A) std',
	error_y_mode = 'band',
	log_y = True,
	line_group = 'Measurement name',
	**PLOT_GRAPH_DIMENSIONS,
)
fig.write_html(str(utils.path_to_scripts_output_directory/Path('iv_curves.html')), include_plotlyjs = 'cdn')

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
fig.write_html(str(utils.path_to_dashboard_media_directory/Path('iv_curves.html')), include_plotlyjs = 'cdn')
