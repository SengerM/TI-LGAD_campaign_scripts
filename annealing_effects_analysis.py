import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import measurements_table as mt
import datetime
from inter_pixel_distance_analysis import SORT_VALUES_BY, PLOT_GRAPH_DIMENSIONS, annealing_time_to_label_for_the_plot, EXCLUDE_VOLTAGE_SCAN_MEASUREMENTS_NAMES

measurements_table_df = mt.create_measurements_table()
measurements_table_df['Annealing time'] = measurements_table_df.index.map(mt.get_measurement_annealing_time)
measurements_table_df['Fluence (neq/cm^2/1e14)'] = measurements_table_df.index.map(mt.get_measurement_fluence)/1e14

# Find annealed devices ---
df = measurements_table_df.copy()
df = df.dropna()
df = df.loc[df['Annealing time']>datetime.timedelta(0,0,0)]
annealed_devices_set = set(df['Measured device'])

# Bias current analysis ------------------------------------------------
if True:
	iv_data_df = pandas.DataFrame()
	for measurement_name in measurements_table_df.loc[measurements_table_df['Measured device'].isin(annealed_devices_set)].query('Type=="IV curve" | Type=="scan 1D sweeping bias voltage"').index:
		if any([pattern in measurement_name for pattern in {'quick','after_going_to_PSI','20220110160208','20220121145445','20220123205735','20220112014240','20220113151136','20211023125955'}]):
			continue
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
				df = df.iloc[::3, :] # Remove some elements, otherwise the plots are too dense.
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
	fig = utils.line(
		title = f'IV curves for annealed devices<br><sup>Plot updated: {datetime.datetime.now()}</sup>',
		data_frame = df,
		x = 'Bias voltage (V) mean',
		y = 'Bias current (A) mean',
		error_y = 'Bias current (A) std',
		error_x = 'Bias voltage (V) std',
		# ~ log_y = True,
		line_group = 'Measurement name',
		**PLOT_GRAPH_DIMENSIONS,
	)
	fig.write_html(str(utils.path_to_dashboard_media_directory/Path('annealing_iv_curves.html')), include_plotlyjs = 'cdn')
