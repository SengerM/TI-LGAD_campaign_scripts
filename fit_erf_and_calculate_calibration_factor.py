import pandas
import utils
from scipy import special
from lmfit import Model
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import interpolate

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
		parameters['laser_sigma'].set(min=5e-6, max=22e-6)
		fit_results[pad] = fit_model.fit(y_data_for_fit, parameters, x=x_data_for_fit)
	return fit_results

def script_core(measurement_name: str, window_size=125e-6, force=False):
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	with bureaucrat.verify_no_errors_context():
		measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
		
		fit_results = fit_erf(measured_data_df, windows_size=window_size)
		results = pandas.DataFrame(columns = ['Pad'])
		results.set_index('Pad', inplace=True)
		for pad in fit_results:
			results.loc[pad,'Laser sigma (m)'] = fit_results[pad].params['laser_sigma'].value
			results.loc[pad,'Metal-silicon distance (m)'] = fit_results[pad].params['x_offset'].value
			results.loc[pad,'y_offset'] = fit_results[pad].params['y_offset'].value
			results.loc[pad,'y_scale'] = fit_results[pad].params['y_scale'].value
		results.to_csv(bureaucrat.processed_data_dir_path/Path('fit_results.csv'))
		
		fig = utils.line(
			data_frame = utils.mean_std(measured_data_df, by=['n_position','Pad', 'Distance (m)']),
			x = 'Distance (m)',
			y = 'Normalized collected charge mean',
			error_y = 'Normalized collected charge std',
			error_y_mode = 'band',
			color = 'Pad',
			markers = '.',
			title = f'Laser profile check<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
		)
		for pad in results.index:
			if pad == 'left':
				df = measured_data_df.loc[measured_data_df['Distance (m)']<measured_data_df['Distance (m)'].mean()-window_size/2,'Distance (m)']
			else:
				df = measured_data_df.loc[measured_data_df['Distance (m)']>measured_data_df['Distance (m)'].mean()+window_size/2,'Distance (m)']
			x = np.linspace(min(df), max(df), 99)
			fig.add_trace(
				go.Scatter(
					x = x,
					y = fit_results[pad].eval(params=fit_results[pad].params, x = x),
					mode = 'lines',
					name = f'Fit erf {pad} pad, σ<sub>laser</sub>={fit_results[pad].params["laser_sigma"].value*1e6:.1f} µm',
					line = dict(color='black', dash='dash'),
				)
			)
		fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'fit.html')), include_plotlyjs = 'cdn')
		
		# Now center data in Distance (m) = 0 and find calibration factor ---
		offset = measured_data_df['Distance (m)'].iloc[0] - measured_data_df['Distance - offset (m)'].iloc[0]
		x_50_percent = {}
		for pad in results.index:
			if pad == 'left':
				df = measured_data_df.loc[measured_data_df['Distance (m)']<measured_data_df['Distance (m)'].mean()-window_size/2,'Distance (m)']
			else:
				df = measured_data_df.loc[measured_data_df['Distance (m)']>measured_data_df['Distance (m)'].mean()+window_size/2,'Distance (m)']
			x = np.linspace(min(df), max(df), 99)
			y = fit_results[pad].eval(params=fit_results[pad].params, x = x)
			inverted_erf = interpolate.interp1d(
				x = y,
				y = x,
			)
			x_50_percent[pad] = float(inverted_erf(.5))
		multiply_distance_by_this_scale_factor_to_fix_calibration = 2*window_size/((x_50_percent['left']-x_50_percent['right'])**2)**.5
		with open(bureaucrat.processed_data_dir_path/Path('scale_factor.txt'), 'w') as ofile:
			print(f'multiply_distance_by_this_scale_factor_to_fix_calibration = {multiply_distance_by_this_scale_factor_to_fix_calibration}', file=ofile)
		
		for distance_col in {'Distance (m)','Distance - offset (m)'}:
			measured_data_df[f'{distance_col} calibrated'] = measured_data_df[distance_col]*multiply_distance_by_this_scale_factor_to_fix_calibration
		fig = utils.line(
			data_frame = utils.mean_std(measured_data_df, by=['n_position','Pad', 'Distance - offset (m) calibrated', 'Distance (m) calibrated']),
			x = 'Distance - offset (m) calibrated',
			y = 'Normalized collected charge mean',
			error_y = 'Normalized collected charge std',
			error_y_mode = 'band',
			color = 'Pad',
			markers = '.',
			title = f'Laser profile check after calibration applied<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
		)
		for pad in results.index:
			if pad == 'left':
				df = measured_data_df.loc[measured_data_df['Distance (m)']<measured_data_df['Distance (m)'].mean()-window_size/2,'Distance (m)']
			else:
				df = measured_data_df.loc[measured_data_df['Distance (m)']>measured_data_df['Distance (m)'].mean()+window_size/2,'Distance (m)']
			x = np.linspace(min(df), max(df), 99)
			fig.add_trace(
				go.Scatter(
					x = (x-offset)*multiply_distance_by_this_scale_factor_to_fix_calibration,
					y = fit_results[pad].eval(params=fit_results[pad].params, x = x),
					mode = 'lines',
					name = f'Fit erf {pad} pad, σ<sub>laser</sub>={fit_results[pad].params["laser_sigma"].value*1e6*multiply_distance_by_this_scale_factor_to_fix_calibration:.1f} µm',
					line = dict(color='black', dash='dash'),
				)
			)
		fig.write_html(str(bureaucrat.processed_data_dir_path/Path(f'after_calibration.html')), include_plotlyjs = 'cdn')
	
if __name__ == '__main__':
	import argparse
	import measurements_table as mt
	
	parser = argparse.ArgumentParser(description='Fits an ERF function to the metal-silicon interfaces of a linear scan and uses this information to calculate a calibration factor based on the real distance.')
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement. If "all", the script is applied to all linear scans.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	if args.directory.lower() != 'all':
		script_core(Path(args.directory).parts[-1])
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			if mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
				if not (utils.path_to_measurements_directory/Path(measurement_name)/Path('fit_erf_and_calculate_calibration_factor')/Path('fit_results.csv')).is_file():
					print(f'Processing {measurement_name}...')
					try:
						script_core(measurement_name)
					except Exception as e:
						print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
