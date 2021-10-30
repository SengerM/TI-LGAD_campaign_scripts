from pathlib import Path
from measurements_table import create_measurements_table
import utils
import pandas
import plotly.express as px

measurements_table_df = create_measurements_table()

IV_measurements_table_df = measurements_table_df.query('Type=="IV curve"')

data_df = pandas.DataFrame()
for measurement_name in IV_measurements_table_df.index:
	df = pandas.read_feather(utils.path_to_measurements_directory/Path(measurement_name)/Path('IV_curve/measured_data.fd'))
	df['Bias voltage (V)'] *= -1
	df['Bias current (A)'] *= -1
	df['Measurement name'] = measurement_name
	device_name = IV_measurements_table_df.loc[measurement_name, 'Measured device']
	df['Device name'] = device_name
	df['Device specs'] = utils.get_device_specs_string(device_name, humanize=False)
	data_df = data_df.append(df, ignore_index=True)

data_df = utils.mean_std(data_df, by=['Measurement name','Device name','Device specs','n_voltage'])

fig = utils.line(
	data_frame = data_df,
	x = 'Bias voltage (V) mean',
	y = 'Bias current (A) mean',
	labels = {
		'Bias voltage (V) mean': 'Bias voltage (V)',
		'Bias current (A) mean': 'Bias current (A)',
	},
	error_y = 'Bias current (A) std',
	error_y_mode = 'band',
	symbol = 'Device name',
	color = 'Device specs',
	markers = True,
	hover_name = 'Measurement name',
	log_y = True,
)

fig.show()
