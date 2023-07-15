#!/usr/bin/env python3
import re
import argparse
import pymlir

from calibration.kld_calibrator import ActivationCalibrator, ActivationCalibrator2
from calibration.data_selector import DataSelector


if __name__ == '__main__':
    print("MINI MLIR {}".format(pymlir.module().version))
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('mlir_file', metavar='mlir_file', help='mlir file')
    parser.add_argument('--dataset', type=str, help='dataset for calibration')
    parser.add_argument('--data_list', type=str, help='Input list file contain all input')
    parser.add_argument('--input_num', type=int, default=0, help='num of images for calibration')
    parser.add_argument('--tune_num', type=int, default=5, help='num of images for tune')
    parser.add_argument('--histogram_bin_num', type=int, default=2048,
                        help='Specify histogram bin numer for kld calculate')
    parser.add_argument('-o', '--calibration_table', type=str, help='output threshold table')
    parser.add_argument('--debug_cmd', type=str, default='', help='debug cmd')
    # yapf: enable
    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    selector = DataSelector(args.dataset, args.input_num, args.data_list)
    # calibration
    if 'use_old_cali' in args.debug_cmd:
        calibrator = ActivationCalibrator(args, selector)
    else:
        calibrator = ActivationCalibrator2(args, selector)
    calibrator.run()
