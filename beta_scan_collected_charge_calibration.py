from pathlib import Path
import utils
import pandas
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, optimize
import numpy as np
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
import measurements_table as mt
import scipy.constants as constants

def script_core(measurement_name: str):
	if mt.retrieve_measurement_type(measurement_name) != 'beta scan':
		raise ValueError(f'Measurement must be of type `beta scan`, but measurement {repr(measurement_name)} is a {repr(mt.retrieve_measurement_type(measurement_name))}.')
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	measured_data_beta_scan_df = pandas.read_csv(bureaucrat.processed_by_script_dir_path('acquire_and_parse_with_oscilloscope.py')/Path('parsed_data.csv'))
	
	kde_collected_charge = stats.gaussian_kde(measured_data_beta_scan_df['Collected charge (V s)'].dropna())
	optimize_result = optimize.minimize(
		fun = lambda x: -kde_collected_charge(x),
		x0 = 30e-12, # Hardcoded value here...
		method = 'Nelder-Mead',
		tol = .01e-12,
	)
	collected_charge_distribution_maximum_in_Volts_time_seconds = optimize_result.x[0]
	
	fig = px.histogram(
		title = f'Collected charge with beta source in Volt*second<br><sup>Measurement: {measurement_name}</sup>',
		data_frame = measured_data_beta_scan_df,
		x = 'Collected charge (V s)',
		histnorm = 'probability density',
	)
	x_axis_values = np.linspace(measured_data_beta_scan_df['Collected charge (V s)'].min(),measured_data_beta_scan_df['Collected charge (V s)'].max(),99)
	fig.add_trace(
		go.Scatter(
			x = x_axis_values,
			y = kde_collected_charge(x_axis_values),
			mode = 'lines',
			showlegend = False,
		)
	)
	fig.add_vline(
		x = collected_charge_distribution_maximum_in_Volts_time_seconds,
		annotation_text = f'Collected charge maximum: {collected_charge_distribution_maximum_in_Volts_time_seconds*1e12:.1f}×10⁻¹² V s',
	)
	fig.write_html(bureaucrat.processed_data_dir_path/Path('collected_charge_Volt_times_second.html'), include_plotlyjs = 'cdn')
	
	# Now translate this number to Coulomb using the detector thickness ---
	measured_device_name = str(mt.retrieve_device_name(measurement_name))
	if utils.bureaucrat.get_devices_specs_dictionary(measured_device_name)['gain'] == 'yes':
		# The following calibration is for PIN diodes.
		exit()
	device_thickness = utils.bureaucrat.get_devices_specs_dictionary(measured_device_name)['substrate thickness (µm)']*1e-6
	def charge_vs_thickness(d: float):
		"""d: Device thickness, meters."""
		# https://sengerm.github.io/html-github-hosting/210721_Commissioning_of_Chubut_board/210721_Commissioning_of_Chubut_board.html#Theory%20of%20collected%20charge
		return constants.e*(31*np.log(d/1e-6)+128)*d/1e-6/3.65
	theoretical_MPV_charge = charge_vs_thickness(d = device_thickness) # This is the MPV we expect according to theory for a PIN with MIPs.
	
	system_transimpedance = collected_charge_distribution_maximum_in_Volts_time_seconds/theoretical_MPV_charge
	
	measured_data_beta_scan_df['Collected charge (C)'] = measured_data_beta_scan_df['Collected charge (V s)']/system_transimpedance
	
	kde_collected_charge = stats.gaussian_kde(measured_data_beta_scan_df['Collected charge (C)'].dropna())
	optimize_result = optimize.minimize(
		fun = lambda x: -kde_collected_charge(x),
		x0 = .5e-15, # Hardcoded value here...
		method = 'Nelder-Mead',
		tol = .01e-12,
	)
	collected_charge_distribution_maximum_in_Coulomb = optimize_result.x[0]
	
	fig = px.histogram(
		title = f'Collected charge with beta source in Coulomb<br><sup>Measurement: {measurement_name}</sup>',
		data_frame = measured_data_beta_scan_df,
		x = 'Collected charge (C)',
		histnorm = 'probability density',
	)
	x_axis_values = np.linspace(measured_data_beta_scan_df['Collected charge (C)'].min(),measured_data_beta_scan_df['Collected charge (C)'].max(),99)
	fig.add_trace(
		go.Scatter(
			x = x_axis_values,
			y = kde_collected_charge(x_axis_values),
			mode = 'lines',
			showlegend = False,
		)
	)
	fig.add_vline(
		x = collected_charge_distribution_maximum_in_Coulomb,
		annotation_text = f'Collected charge maximum: {collected_charge_distribution_maximum_in_Coulomb:.1e} C',
	)
	fig.write_html(bureaucrat.processed_data_dir_path/Path('collected_charge_Coulomb.html'), include_plotlyjs = 'cdn')
	
	with open(bureaucrat.processed_data_dir_path/Path('transimpedance_calibration.txt'), 'w') as ofile:
		print(f'Transimpedance (Ω) = {system_transimpedance}', file=ofile)

if __name__ == '__main__':
	import argparse
	import measurements_table as mt
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a beta scan measurement. If "all", the script is applied to all beta scans.',
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
			if mt.retrieve_measurement_type(measurement_name) == 'beta scan':
				if not (utils.path_to_measurements_directory/Path(measurement_name)/Path('beta_scan_collected_charge_calibration')/Path('calibration_data.csv')).is_file():
					print(f'Processing {measurement_name}...')
					try:
						script_core(measurement_name)
					except Exception as e:
						print(f'Cannot process {measurement_name}, reason: {repr(e)}.')

