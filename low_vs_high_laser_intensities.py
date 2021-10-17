import utils
from pathlib import Path
import pandas
import plotly.express as px
import numpy as np

MEASUREMENTS = {
	'20211008170143_#45_1MIP',
	'20211009111833_#45_10MIP',
	'20211012181944_#14_1MIP',
	'20211013012711_#14_0.25MIP',
}

data_df = pandas.DataFrame()
for measurement in MEASUREMENTS:
	temp_df = utils.read_measured_data_from(measurement)
	temp_df['Measurement'] = measurement
	temp_df['Device'] = measurement.split('_')[1]
	temp_df['Laser intensity'] = measurement.split('_')[-1]
	distances_df = utils.calculate_1D_scan_distance_from_dataframe(temp_df)
	temp_df.set_index('n_position', inplace=True)
	temp_df = temp_df.merge(distances_df, left_index=True, right_index=True)
	temp_df = temp_df.append(temp_df, ignore_index=True)
	temp_df.reset_index()
	data_df = data_df.append(temp_df, ignore_index = True)

# Calculate mean and std ---
GROUP_BY = ['n_pulse','n_channel','n_position','Measurement','Device','Laser intensity']
mean_df = data_df.groupby(by=GROUP_BY).mean()
std_df =  data_df.groupby(by=GROUP_BY).std()
for column in std_df:
	column_name = f"{column.split('(')[0]}std ({column.split('(')[-1]}" if '(' in column else f'{column} std'
	mean_df[column_name] = std_df[column]
mean_df.reset_index(inplace=True)

# Normalize collected charge ---
for measurement in MEASUREMENTS:
	for n_pulse in sorted(set(mean_df['n_pulse'])):
		for n_channel in sorted(set(mean_df['n_channel'])):
			mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] = mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Collected charge (V s)']
			mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] -= np.nanmin(mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'])
			division_factor = np.nanmax(mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'])
			mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] /= division_factor
			mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge std'] = mean_df.loc[(mean_df['Measurement']==measurement)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Collected charge std (V s)'] / division_factor

fig = utils.line(
	data_frame = mean_df.loc[mean_df['n_pulse']==1],
	x = 'Distance (m)',
	y = 'Normalized collected charge',
	error_y = 'Normalized collected charge std',
	color = 'Laser intensity',
	facet_row = 'Device',
	symbol = 'n_channel',
	error_y_mode = 'bands',
)
fig.write_html(file='laser intensity comparison.html',include_plotlyjs='cdn')
