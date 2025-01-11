import sys
import os
from Data.data_prepare import *
args={}
args['input_csv'] = './Data/admet.csv'
args['output_bin'] = './Data/admet.bin'
args['output_csv'] = './Data/admet_group.csv'

built_data_and_save_for_splited(
        origin_path=args['input_csv'],
        save_path=args['output_bin'],
        group_path=args['output_csv'],
        task_list_selected=None
         )






