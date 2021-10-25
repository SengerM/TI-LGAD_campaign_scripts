import pandas
import utils
from scipy import special
from lmfit import Model
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def metal_silicon_transition_model_function(x, y_scale, laser_sigma, x_offset, y_offset):
	return y_scale*special.erf((x-x_offset)/laser_sigma*2**.5) + y_offset

def fit_erf(df, windows_size=130e-6):
	"""Given a df with data from a single 1D scan, this function fits an erf (convolution of Gaussian and step) to each metal-silicon interface. Returns the fit result object by lmfit, one for each pad (left and right)."""
	utils.check_df_is_from_single_1D_scan(df)
	
	if 'Pad' not in df.columns:
		df = utils.tag_left_right_pad(df)
	if 'Subtracted distance offset (m)' not in df.columns:
		df = utils.calculate_distance_offset_by_linear_interpolation(df)
	if 'Normalized collected charge' not in df.columns:
		df = utils.calculate_normalized_collected_charge(df)
	df = df.loc[df['n_pulse']==1] # Use only pulse 1 for this.
	df = df.loc[df['Distance (m)'].notna()] # Drop rows that have NaN values in the relevant columns.
	df = df.loc[df['Normalized collected charge'].notna()] # Drop rows that have NaN values in the relevant columns.
	
	fit_results = {}
	fit_model = Model(metal_silicon_transition_model_function)
	for pad in set(df['Pad']):
		this_pad_df = df.loc[df['Pad']==pad]
		if pad == 'left':
			x_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']<-windows_size/2, 'Distance (m)']
			y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']<-windows_size/2, 'Normalized collected charge']
		elif pad == 'right':
			x_data_for_fit = -this_pad_df.loc[this_pad_df['Distance (m)']>windows_size/2, 'Distance (m)']
			y_data_for_fit = this_pad_df.loc[this_pad_df['Distance (m)']>windows_size/2, 'Normalized collected charge']
		parameters = fit_model.make_params(
			laser_sigma = 10e-6,
			x_offset = -windows_size, # Transition metal→silicon in the left pad.
			y_scale = 1/2,
			y_offset = 1/2,
		)
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
	results = pandas.DataFrame(columns = ['Pad','Laser sigma (m)', 'Metal-silicon distance from center (m)'])
	results.set_index('Pad', inplace=True)
	for pad in fit_results:
		results.loc[pad,'Laser sigma (m)'] = fit_results[pad].params['laser_sigma'].value
		results.loc[pad,'Metal-silicon distance from center (m)'] = (fit_results[pad].params['x_offset'].value**2)**.5
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
			df = measured_data_df.loc[measured_data_df['Distance (m)']<-WINDOWS_SIZE/2,'Distance (m)']
		else:
			df = measured_data_df.loc[measured_data_df['Distance (m)']>WINDOWS_SIZE/2,'Distance (m)']
		x = np.linspace(min(df), max(df), 99)
		fig.add_trace(
			go.Scatter(
				x = x,
				y = fit_results[pad].eval(params=fit_results[pad].params, x = x if pad == 'left' else -x),
				mode = 'lines',
				name = f'Fit for {pad} pad',
				line = dict(color='black', dash='dash'),
			)
		)
	fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'fit.html')), include_plotlyjs = 'cdn')

if __name__ == '__main__':
	measurements_to_process = [
		'20211025040241_#65_1DScan_99V',
		'20211025011141_#65_1DScan_88V',
		'20211024221940_#65_1DScan_77V',
		'20211024192714_#65_1DScan_66V',
		'20211024163129_#65_1DScan_55V',
	]
	for measurement in measurements_to_process:
		script_core(measurement)
