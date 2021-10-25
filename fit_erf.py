import pandas
import utils
from scipy import special
from lmfit import Model
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def metal_silicon_transition_model_function_left_pad(x, y_scale, laser_sigma, x_offset, y_offset):
	return y_scale*special.erf((x-x_offset)/laser_sigma*2**.5) + y_offset

def metal_silicon_transition_model_function_right_pad(x, y_scale, laser_sigma, x_offset, y_offset):
	return metal_silicon_transition_model_function_left_pad(-x, y_scale, laser_sigma, -x_offset, y_offset)

def fit_erf(df, windows_size=130e-6):
	"""Given a df with data from a single 1D scan, this function fits an erf (convolution of Gaussian and step) to each metal-silicon interface. Returns the fit result object by lmfit, one for each pad (left and right)."""
	utils.check_df_is_from_single_1D_scan(df)
	
	if 'Pad' not in df.columns:
		df = utils.tag_left_right_pad(df)
	if 'Normalized collected charge' not in df.columns:
		df = utils.calculate_normalized_collected_charge(df)
	df = df.loc[df['n_pulse']==1] # Use only pulse 1 for this.
	df = df.loc[df['Distance (m)'].notna()] # Drop rows that have NaN values in the relevant columns.
	df = df.loc[df['Normalized collected charge'].notna()] # Drop rows that have NaN values in the relevant columns.
	
	fit_results = {}
	fit_model_left_pad = Model(metal_silicon_transition_model_function_left_pad)
	fit_model_right_pad = Model(metal_silicon_transition_model_function_right_pad)
	for pad in set(df['Pad']):
		this_pad_df = df.loc[df['Pad']==pad]
		if pad == 'left':
			x_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']<this_pad_df['Distance (m)'].mean()-windows_size/2, 'Distance (m)']
			y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']<this_pad_df['Distance (m)'].mean()-windows_size/2, 'Normalized collected charge']
			fit_model = fit_model_left_pad
		elif pad == 'right':
			x_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']>this_pad_df['Distance (m)'].mean()+windows_size/2, 'Distance (m)']
			y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']>this_pad_df['Distance (m)'].mean()+windows_size/2, 'Normalized collected charge']
			fit_model = fit_model_right_pad
		parameters = fit_model.make_params(
			laser_sigma = 10e-6,
			x_offset = this_pad_df['Distance (m)'].mean()-windows_size if pad=='left' else this_pad_df['Distance (m)'].mean()+windows_size, # Transition metal→silicon in the left pad.
			y_scale = 1/2,
			y_offset = 1/2,
		)
		parameters['y_scale'].set(min=.1, max=.9)
		parameters['y_offset'].set(min=.1, max=.9)
		fit_results[pad] = fit_model.fit(y_data_for_fit, parameters, x=x_data_for_fit)
	return fit_results

def script_core(measurement_name: str):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
	
	WINDOWS_SIZE = 130e-6
	
	fit_results = fit_erf(measured_data_df, windows_size=WINDOWS_SIZE)
	results = pandas.DataFrame(columns = ['Pad'])
	results.set_index('Pad', inplace=True)
	for pad in fit_results:
		results.loc[pad,'Laser sigma (m)'] = fit_results[pad].params['laser_sigma'].value
		results.loc[pad,'Metal-silicon distance (m)'] = fit_results[pad].params['x_offset'].value
		results.loc[pad,'y_offset'] = fit_results[pad].params['y_offset'].value
		results.loc[pad,'y_scale'] = fit_results[pad].params['y_scale'].value
	results.to_csv(bureaucrat.processed_data_dir_path/Path('fit_results.csv'))
	
	fig = utils.line(
		data_frame = utils.calculate_mean_measured_values_at_each_position(measured_data_df, by=['n_position','Pad']),
		x = 'Distance (m)',
		y = 'Normalized collected charge',
		color = 'Pad',
		error_y = 'Normalized collected charge std',
		error_y_mode = 'bands',
		markers = '.',
	)
	for pad in results.index:
		if pad == 'left':
			df = measured_data_df.loc[measured_data_df['Distance (m)']<measured_data_df['Distance (m)'].mean()-WINDOWS_SIZE/2,'Distance (m)']
		else:
			df = measured_data_df.loc[measured_data_df['Distance (m)']>measured_data_df['Distance (m)'].mean()+WINDOWS_SIZE/2,'Distance (m)']
		x = np.linspace(min(df), max(df), 99)
		fig.add_trace(
			go.Scatter(
				x = x,
				y = fit_results[pad].eval(params=fit_results[pad].params, x = x),
				mode = 'lines',
				name = f'Fit for {pad} pad',
				line = dict(color='black', dash='dash'),
			)
		)
	fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'fit.html')), include_plotlyjs = 'cdn')
	
if __name__ == '__main__':
	import measurements_table as mt
	measurements_table_df = mt.create_measurements_table()
	for measurement in measurements_table_df.index:
		if mt.retrieve_measurement_type(measurement) == 'scan 1D':
			if (utils.path_to_measurements_directory/Path(measurement)/Path('fit_erf')).is_dir():
				continue
			print(f'Processing {repr(measurement)}...')
			try:
				script_core(measurement)
			except Exception as e:
				print(f'Cannot fit_erf to measurement {repr(measurement)}, reason: {repr(e)}.')
