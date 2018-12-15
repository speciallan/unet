#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import sys
from spe import *

# cmd = 'labelme_json_to_dataset'+ base_dir + 'tf/data/test/'
cmd = 'labelme_json_to_dataset'

dir = base_dir + 'tf/data/test/math/56/'

test_imgs = os.listdir(dir)
test_imgs.sort()

for i, img_name in enumerate(test_imgs):
    if img_name.lower().endswith('.json'):

        json_file_path = dir + img_name
        print(json_file_path + ' start...')
        cmd_new = cmd + ' ' + json_file_path
        os.system(cmd_new)

