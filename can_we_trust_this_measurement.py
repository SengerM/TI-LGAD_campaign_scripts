import pandas
import utils
from data_processing_bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import measurements_table as mt

def script_core(measurement_name: str, force=False):
	
	MEASUREMENTS_WITH_NOT_ALL_PADS_DC_GROUNDED = {
		'20210825124019_#22_LPW64%_1um_3300Trig_lineary',
		'20210828183326_#1_LPW64%_1um_3300Trig_linear_x',
		'20210901144436_#45_LPW64%_1um_3300Trig_linear_x',
		'20210903135445_#51_LPW64%_1um_3300Trigg_linear_x',
		'20210912161205_#65_LPW55%_1um_3300Trigg_linearx',
		'20210923205024_#77_LinearScan_3333Trig_LPW55_111V',
		'20210928113802_LinearScan_Device_#65_LPW_55_BiasVoltage_111_Comment_3333Triggers',
		'20211004172815_#57_111V_BetaScan',
		'20211004194105_#57_99V_BetaScan',
		'20211005105459_#57_88V_BetaScan',
		'20211005114157_#57_22V_BetaScan',
		'20211005125828_#57_55V_BetaScan',
		'20211006130249_#57_LaserDACScan',
		'20211007085602_#57_PIN_LaserDACScan',
		'20211007191936_#57_LinearScan_5MIP',
		'20211008170143_#45_1MIP',
		'20211009111833_#45_10MIP',
		'20211009194430_#33_LinearScan_1MIP',
		'20211010164616_#65_find_focus_great_detail',
		'20211010181916_#65_1MIP',
		'20211012032747_#1_1MIP',
		'20211012181944_#14_1MIP',
		'20211013012711_#14_0.25MIP',
		'20211013121218_#14_BetaScan_99V',
		'20211013144303_#14_BetaScan_99V_NoRFShield',
		'20211014124808_#77_2DScan',
		'20211014161540_#77_1MIP',
		'20211015001057_#77_1MIP',
		'20211015134203_#51_2DMap',
		'20211015164100_#51_find_focus_detail',
		'20211015182324_#51_1MIP',
		'20211018181528_#22_2DMap',
		'20211018205340_#22_1MIP',
		'20211019090359_#22_1MIP_122V',
		'20211020175106_#1_2D_map',
		'20211020205249_#1_z_scan_find_focus',
		'20211020232836_#1_1MIP_PadCenterScan',
		'20211021220330_#1_1MIP_55V',
		'20211022095225_#1_sweeping_bias_voltage',
		'20211022095227_#1_1DScan_66V',
		'20211022110858_#1_1DScan_88V',
		'20211022122254_#1_1DScan_111V',
		'20211022193606_#45_2D_map',
		'20211022212353_#45_z_scan_for_focus_detail',
		'20211023000338_#45_sweeping_bias_voltage',
		'20211023000339_#45_1DScan_55V',
		'20211023011256_#45_1DScan_66V',
		'20211023022052_#45_1DScan_77V',
		'20211023032841_#45_1DScan_88V',
		'20211023043631_#45_1DScan_97V',
		'20211023104738_#45_IVCurve',
		'20211023125955_#77_IVCurve',
		'20211023131124_#77_2D_map',
		'20211023142808_#77_find_focus',
		'20211023190308_#77_sweeping_bias_voltage',
		'20211023190309_#77_1DScan_55V',
		'20211023215712_#77_1DScan_66V',
		'20211024004813_#77_1DScan_77V',
		'20211024033857_#77_1DScan_88V',
		'20211024062930_#77_1DScan_99V',
		'20211024131707_#65_IV_curve',
		'20211024134543_#65_2D_map',
		'20211024152853_#65_focus',
		'20211024163128_#65_sweeping_bias_voltage',
		'20211024163129_#65_1DScan_55V',
		'20211024192714_#65_1DScan_66V',
		'20211024221940_#65_1DScan_77V',
		'20211025011141_#65_1DScan_88V',
		'20211025040241_#65_1DScan_99V',
		'20211104203325_#45_IV_curve_not_all_pads_grounded',
		'20211104225200_#45_z_scan_focus',
		'20211104233152_#45_sweeping_bias_voltage',
		'20211104233307_#45_1DScan_55V',
		'20211105013835_#45_1DScan_235V',
		'20211105034829_#45_1DScan_145V',
		'20211105055129_#45_1DScan_100V',
		'20211105075430_#45_1DScan_190V',
		'20211105095722_#45_1DScan_77V',
		'20211105120046_#45_1DScan_122V',
		'20211105140413_#45_1DScan_167V',
		'20211105160708_#45_1DScan_212V',
	}
	
	MEASUREMENTS_IN_WHICH_THE_DETECTOR_DIED = {
		'20220123045041_#77_7DaysAnnealing_1DScan_624V',
		'20220123075136_#77_7DaysAnnealing_1DScan_645V',
		'20220123105352_#77_7DaysAnnealing_1DScan_666V',
	}
	
	FORCE_TRUST = {
		'20211104233152_#45_sweeping_bias_voltage',
		'20211104233307_#45_1DScan_55V',
		'20211105034829_#45_1DScan_145V',
		'20211105055129_#45_1DScan_100V',
		'20211105075430_#45_1DScan_190V',
		'20211105095722_#45_1DScan_77V',
		'20211105120046_#45_1DScan_122V',
		'20211105140413_#45_1DScan_167V',
		'20211105160708_#45_1DScan_212V',
	}
	
	bureaucrat = Bureaucrat(
		utils.path_to_measurements_directory/Path(measurement_name),
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	with bureaucrat.verify_no_errors_context():
		can_we_trust = True
		reasons_not_to_trust = []

		if measurement_name in MEASUREMENTS_WITH_NOT_ALL_PADS_DC_GROUNDED:
			can_we_trust = False
			reasons_not_to_trust.append('Not all pads were grounded.')
		
		if measurement_name in MEASUREMENTS_IN_WHICH_THE_DETECTOR_DIED:
			can_we_trust = False
			reasons_not_to_trust.append('Detector was dead.')
		
		measured_data_df = utils.read_and_pre_process_1D_scan_data(measurement_name)
		
		# Check that amplifiers did not run into nonlinear mode ---
		DYNAMIC_RANGE = .9 # Volt
		AMMOUNT_OF_SIGNALS_WITHIN_DYNAMIC_RANGE = .95
		if len(measured_data_df.query(f'`Amplitude (V)` >= {DYNAMIC_RANGE}'))/len(measured_data_df) > 1-AMMOUNT_OF_SIGNALS_WITHIN_DYNAMIC_RANGE:
			can_we_trust = False
			reasons_not_to_trust.append(f'Amplitude is > {DYNAMIC_RANGE} V for (at least) the {(1-AMMOUNT_OF_SIGNALS_WITHIN_DYNAMIC_RANGE)*100:.2f} % of the events, amplifiers go into nonlinear regime.')
		
		# Check that there are not too many NaN values in the amplitude ---
		if measured_data_df['Amplitude (V)'].isna().sum()/len(measured_data_df) > .1:
			can_we_trust = False
			reasons_not_to_trust.append(f'Too many NaN points in the amplitude.')
		
		# Devices ---
		UNTRUSTABLE_DEVICES = {'1','2','88'}
		if mt.retrieve_device_name(measurement_name) in UNTRUSTABLE_DEVICES:
			can_we_trust = False
			reasons_not_to_trust.append(f'Measured device name is {repr(mt.retrieve_device_name(measurement_name))} which is in the listed of "untrustable devices".')
		
		if measurement_name in FORCE_TRUST:
			can_we_trust = True
		
		with open(bureaucrat.processed_data_dir_path/Path('result.txt'), 'w') as ofile:
			print(f'can_we_trust = {"yes" if can_we_trust else "no"}', file=ofile)
			if len(reasons_not_to_trust) > 0:
				print(f'\nReasons not to trust:', file=ofile)
				for reason in reasons_not_to_trust:
					print(f'- {reason}', file=ofile)
			if measurement_name in FORCE_TRUST:
				print('\nThis measurement is in the "FORCE_TRUST" set.', file=ofile)

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Creates a file saying whether some measurement is reliable or not. For example if the amplitude is > 1 V this means that the amplifiers went into the nonlinear regime so we cannot trust the measurement.')
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
		script_core(Path(args.directory).parts[-1], force=True)
	else:
		measurements_table_df = mt.create_measurements_table()
		for measurement_name in sorted(measurements_table_df.index)[::-1]:
			if mt.retrieve_measurement_type(measurement_name) == 'scan 1D':
				print(f'Processing {measurement_name}...')
				try:
					script_core(measurement_name)
				except Exception as e:
					print(f'Cannot process {measurement_name}, reason {repr(e)}...')
				
