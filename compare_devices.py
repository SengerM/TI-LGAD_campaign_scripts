import utils
import numpy as np
from pathlib import Path
import pandas
import plotly.express as px

devices_df = utils.read_devices_sheet()
devices_that_have_been_measured_df = devices_df.loc[~devices_df['Linear scan before irradiation'].isnull()]
data_df = pandas.DataFrame()
for device_number in devices_that_have_been_measured_df.index:
	temp_df = utils.read_measured_data_from(devices_that_have_been_measured_df.loc[device_number, 'Linear scan before irradiation'])
	temp_df['#'] = str(device_number)
	data_df = data_df.append(temp_df, ignore_index=True)

mean_df = data_df.groupby(by=['#','n_pulse','n_channel','n_position']).mean()
std_df = data_df.groupby(by=['#','n_pulse','n_channel','n_position']).std()
for column in std_df:
	column_name = f"{column.split('(')[0]}std ({column.split('(')[-1]}" if '(' in column else f'{column} std'
	mean_df[column_name] = std_df[column]
mean_df.reset_index(inplace=True)

fig = utils.line(
	data_frame = mean_df[mean_df['n_pulse']==1],
	x = 'n_position',
	y = 'Collected charge (V s)',
	color = '#',
	symbol = 'n_channel',
	error_y = 'Collected charge std (V s)',
	error_y_mode = 'bands',
)
for data in fig.data:
	data['legendgroup'] = data['legendgroup'].split(',')[0]
fig.show()
