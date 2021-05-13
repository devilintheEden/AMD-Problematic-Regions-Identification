#!/bin/bash
################################ Training ################################
python im_energy.py
cd pix2pixHD
python train.py --name imagetoinfluenced_512p --continue_train --label_nc 0 --no_instance --dataroot='./datasets/AMD/'