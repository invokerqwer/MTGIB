import sys
import os

# 添加当前脚本目录的父目录到sys.path
sys.path.append('/home/15509949926/project/MTGL-ADMET/Data')
sys.path.append('/home/15509949926/project/MTGL-ADMET')

from Data.data_prepare import *

args={}
args['input_csv'] = './Data/admet_demo.csv'
args['output_bin'] = './Data/admet_demo.bin'
args['output_csv'] = './Data/admet_group_demo.csv'

built_data_and_save_for_splited(
        origin_path=args['input_csv'],
        save_path=args['output_bin'],
        group_path=args['output_csv'],
        task_list_selected=None
         )
