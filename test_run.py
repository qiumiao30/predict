# -*- coding: utf-8 -*-
# @Time    : 2021/10/31 11:29
# @File    : test_run.py
import sys
import os
from args import get_parser
from datetime import datetime
import torch
import time
import sys
parser = get_parser()
args = parser.parse_args()

# matplotlib.use("Agg")

if __name__ == '__main__':
    res = []
    flag = 0
    for headers in [1, 4, 8, 12, 16]:
        for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for seq_length in [10, 20, 30, 40, 50, 60, 70, 80]:
                for kernel_size in [3, 5, 7]:
                    for dropout_proba in [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
                        for init_lr in [0.001, 0.0001, 0.00001]:
                            for epochs in [5, 10, 15, 20]:
                                print(
                                    f'\n\n**** lookback: {seq_length}, lr: {init_lr}, kernel_size: {kernel_size}, epoch: {epochs}, depth: {depth}, dropout: {dropout_proba}, header: {headers} ****')
                                os.system(f"python train.py --dataset smd --model convformer --lookback {seq_length} --init_lr {init_lr} --kernel_size {kernel_size} --epochs {epochs} "
                                          f"--depth {depth} --dropout {dropout_proba} --header {headers}")
                                flag += 1
                                if flag % 8 == 0:
                                    time.sleep(600)