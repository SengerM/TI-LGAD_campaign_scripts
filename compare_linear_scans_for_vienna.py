import measurements_table as mt
import utils
from grafica.plotly_utils.utils import line
import pandas
from pathlib import Path
from inter_pixel_distance_analysis import COLORS_FOR_EACH_FLUENCE_DICT

LINEAR_SCANS_TO_COMPARE = {
	# ~ '20211113135645_#87_1DScan_201V',
	# ~ '20220204133426_#88_1DScan_500V',
	
	# Estos se ven bien ---
	'20211101224628_#52_1DScan_222.0V',
	'20220216113633_#52_1DScan_500V',
	'20220217211645_#53_1DScan_500V',
	'20220219151701_#51_1DScan_500V',
	
	# ~ # Con la serie #77, #78, #79 hay un poco de ranciedad ---
	# ~ '20211112092014_#78_1DScan_222V',
	# ~ '20220112014735_#78_1DScan_500V',
	# ~ '20220110160524_#77_1DScan_500V',
	# ~ '20220113151456_#79_1DScan_500V',
}

LABELS = {
	'Normalized collected charge median': 'Normalized collected charge',
	'Collected charge (V s) median': 'Collected charge (V s)',
	'Collected charge (C) median': 'Collected charge (C)',
	'Fluence (neq/cm^2)/1e14 median': 'Fluence n<sub>eq</sub>/cm<sup>2</sup>×10<sup>-14</sup>',
	'Distance - offset (m) median': 'Laser position (m)',
	'Normalized collected charge': 'Normalized collected charge',
	'Collected charge (V s)': 'Collected charge (V s)',
	'Collected charge (C)': 'Collected charge (C)',
	'Fluence (neq/cm^2)/1e14': 'Fluence n<sub>eq</sub>/cm<sup>2</sup>×10<sup>-14</sup>',
	'Distance - offset (m)': 'Laser position (m)',
	'Pad': 'Pixel',
}

if True: # Collected charge vs distance ---
	measured_data_df = pandas.DataFrame()
	for measurement_name in LINEAR_SCANS_TO_COMPARE:
		df = utils.read_and_pre_process_1D_scan_data(measurement_name)
		df = df.query('n_pulse==1')
		df['Fluence (neq/cm^2)/1e14'] = mt.get_measurement_fluence(measurement_name)/1e14
		df['Distance - offset (m)'] *= utils.read_previously_calculated_distance_calibration_factor(measurement_name),
		df['Collected charge (C)'] = df['Collected charge (V s)']/utils.read_previously_calculated_transimpedance(measurement_name)
		measured_data_df = measured_data_df.append(df, ignore_index = True)

	statistics_df = utils.mean_std(measured_data_df, by=['n_position','Pad','Measurement name'])

	statistics_df = statistics_df.sort_values(by=['Fluence (neq/cm^2)/1e14 median','Pad','n_position'], ascending = True)

	for y_axis in {'Collected charge (C)', 'Normalized collected charge'}:
		fig = line(
			data_frame = statistics_df,
			x = 'Distance - offset (m) median',
			y = f'{y_axis} median',
			error_y = f'{y_axis} MAD_std',
			error_y_mode = 'band',
			line_dash = 'Pad',
			color = 'Fluence (neq/cm^2)/1e14 median',
			labels = LABELS,
			hover_name = 'Measurement name',
			color_discrete_map = COLORS_FOR_EACH_FLUENCE_DICT,
			grouped_legend = True,
		)
		fig.show()

if True: # Time resolution vs distance ---
	time_resolution_df = pandas.DataFrame()
	for measurement_name in LINEAR_SCANS_TO_COMPARE:
		df = pandas.read_feather(utils.path_to_measurements_directory/Path(measurement_name)/Path('calculate_time_resolution/time_resolution_vs_distance_for_best_k1_k2.fd'))
		temp_df = utils.read_and_pre_process_1D_scan_data(measurement_name).groupby('n_position').mean()
		df = df.set_index('n_position')
		df['Distance - offset (m)'] = temp_df['Distance - offset (m)']
		df['Distance - offset (m)'] *= utils.read_previously_calculated_distance_calibration_factor(measurement_name)
		df['Measurement name'] = measurement_name
		df['Fluence (neq/cm^2)/1e14'] = mt.get_measurement_fluence(measurement_name)/1e14
		df = df.reset_index()
		time_resolution_df = time_resolution_df.append(df, ignore_index = True)
	
	time_resolution_df = time_resolution_df.sort_values(by=['Fluence (neq/cm^2)/1e14','Pad','n_position'], ascending = True)
	
	fig = line(
		data_frame = time_resolution_df,
		x = 'Distance - offset (m)',
		y = f'Time resolution (s)',
		line_dash = 'Pad',
		color = 'Fluence (neq/cm^2)/1e14',
		labels = LABELS,
		hover_name = 'Measurement name',
		color_discrete_map = COLORS_FOR_EACH_FLUENCE_DICT,
		grouped_legend = True,
		log_y = True,
	)
	fig.show()
