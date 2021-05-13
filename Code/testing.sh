#!/bin/bash
################################ Testing ################################
cd pix2pixHD
python test.py --name imagetoinfluenced_512p --resize_or_crop none --label_nc 0 --no_instance --dataroot='./datasets/AMD/'