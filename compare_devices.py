import utils
import numpy as np
from pathlib import Path
import pandas
import plotly.express as px
from scipy import interpolate

devices_df = utils.read_devices_sheet()
devices_that_have_been_measured_df = devices_df.loc[~devices_df['linear scan before irradiation'].isnull()]
data_df = pandas.DataFrame()
for device_number in devices_that_have_been_measured_df.index:
	temp_df = utils.read_measured_data_from(devices_that_have_been_measured_df.loc[device_number, 'linear scan before irradiation'])
	temp_df['#'] = str(device_number)
	temp_df = utils.pre_process_raw_data(temp_df)
	device_info = devices_that_have_been_measured_df.loc[devices_that_have_been_measured_df.index==device_number].squeeze()
	temp_df['Device name'] = f'#{device_number},{device_info["trench depth"]},{device_info["trenches"]},{device_info["trench process"]},{device_info["pixel border"]},{device_info["contact type"]}'
	data_df = data_df.append(temp_df, ignore_index=True)
data_df.reset_index()

data_df['Distance (m)'] -= data_df['Distance (m)'].mean()
GROUP_BY = ['#','n_pulse','n_channel','n_position','Pad', 'Device name']
mean_df = data_df.groupby(by=GROUP_BY).mean()
std_df = data_df.groupby(by=GROUP_BY).std()
for column in std_df:
	column_name = f"{column.split('(')[0]}std ({column.split('(')[-1]}" if '(' in column else f'{column} std'
	mean_df[column_name] = std_df[column]
mean_df.reset_index(inplace=True)

# Normalize collected charge ---
for device in set(mean_df['#']):
	for n_pulse in sorted(set(mean_df['n_pulse'])):
		for n_channel in sorted(set(mean_df['n_channel'])):
			mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] = mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Collected charge (V s)']
			mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] -= np.nanmin(mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'])
			division_factor = np.nanmax(mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'])
			mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge'] /= division_factor
			mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Normalized collected charge std'] = mean_df.loc[(mean_df['#']==device)&(mean_df['n_pulse']==n_pulse)&(mean_df['n_channel']==n_channel), 'Collected charge std (V s)'] / division_factor

def calculate_offset(mean_df):
	if len(set(mean_df['#'])) > 1:
		raise ValueError(f'`mean_df` must contain data from a single device, I have received a dataframe with data from {len(set(mean_df["#"]))} devices.')
	if len(mean_df) == 0:
		raise ValueError(f'`mean_df` is empty.')
	mean_df = mean_df.loc[mean_df['n_pulse']==1]
	metal_to_silicon_transition_distance = {}
	for pad in sorted(set(mean_df['Pad'])): # 'left' or 'right'
		if pad == 'left':
			distance_vs_normalized_collected_charge = interpolate.interp1d(
				x = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['n_pulse']==1)&(mean_df['Distance (m)']<-50e-6), 'Normalized collected charge'],
				y = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['n_pulse']==1)&(mean_df['Distance (m)']<-50e-6), 'Distance (m)'],
			)
		else:
			distance_vs_normalized_collected_charge = interpolate.interp1d(
				x = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['n_pulse']==1)&(mean_df['Distance (m)']>50e-6), 'Normalized collected charge'],
				y = mean_df.loc[(mean_df['Pad']==pad)&(mean_df['n_pulse']==1)&(mean_df['Distance (m)']>50e-6), 'Distance (m)'],
			)
		metal_to_silicon_transition_distance[pad] = distance_vs_normalized_collected_charge(.5) # It is the distance in which the normalized collected charge is 0.5
	return np.mean(list(metal_to_silicon_transition_distance.values()))

for device in set(mean_df['#']):
	mean_df.loc[mean_df['#']==device, 'Distance (m)'] -= calculate_offset(mean_df.loc[(mean_df['#']==device)])

figs = []
fig = utils.line(
	data_frame = mean_df.loc[mean_df['n_pulse']==1],
	x = 'Distance (m)',
	y = 'Collected charge (V s)',
	color = 'Device name',
	symbol = 'Pad',
	error_y = 'Collected charge std (V s)',
	error_y_mode = 'bands',
)
for data in fig.data:
	data['legendgroup'] = data['legendgroup'].split(',')[0]
# ~ fig.write_html('raw_collected_charge.html', include_plotlyjs='cdn')
figs.append(fig)

fig = utils.line(
	data_frame = mean_df.loc[mean_df['n_pulse']==1],
	x = 'Distance (m)',
	y = 't_50 std (s)',
	color = 'Device name',
	symbol = 'Pad',
	log_y = True,
)
for data in fig.data:
	data['legendgroup'] = data['legendgroup'].split(',')[0]
# ~ fig.write_html('t_50_std.html', include_plotlyjs='cdn')
figs.append(fig)

fig = utils.line(
	data_frame = mean_df.loc[mean_df['n_pulse']==1],
	x = 'Distance (m)',
	y = 'Normalized collected charge',
	color = 'Device name',
	symbol = 'Pad',
	error_y = 'Normalized collected charge std',
	error_y_mode = 'bands',
)
for data in fig.data:
	data['legendgroup'] = data['legendgroup'].split(',')[0]
# ~ fig.write_html('normalized_collected_charge.html', include_plotlyjs='cdn')
figs.append(fig)

fig = utils.line(
	data_frame = mean_df.loc[mean_df['n_pulse']==1].groupby(by=['Device name','n_position']).agg({'Normalized collected charge': np.sum, 'Normalized collected charge std': np.sum, 'Distance (m)': np.mean}).reset_index(),
	x = 'Distance (m)',
	y = 'Normalized collected charge',
	color = 'Device name',
	error_y = 'Normalized collected charge std',
	error_y_mode = 'bands',
	markers = '.',
)
for data in fig.data:
	data['legendgroup'] = data['legendgroup'].split(',')[0]
# ~ fig.write_html('sum_of_normalized_collected_charge.html', include_plotlyjs='cdn')
figs.append(fig)

for fig in figs:
	for x in [-70e-6, 0, 70e-6]:
		fig.add_shape(type="line",
			yref = "paper",
			x0 = x,
			y0 = 0,
			x1 = x,
			y1 = 1,
			line = dict(
				color = 'black',
				dash = 'dash',
			),
		)
	fig.show()
