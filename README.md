### AMD Problematic Regions Identification

This project trained a neural network to help identify the problematic regions in patients' field of view caused by wet AMD.  For details see the project report.

To run the project, download the **Code** folder. Make sure you have installed *pytorch + torchvision, dominate, numpy, scipy* and *opencv-python* before you run the code. You can run the training shell script to generate the dataset and train the model. You can also run the testing script to test the results. 

The folder **Results** contain the full results of the 38 test images from my previous training process. The latest weight of the model is to big for GitHub without LFS so I upload separately to google drive at this [link](https://drive.google.com/file/d/1d6unWgoU_voklBEQjj99AETYCccw0LZt/view?usp=sharing).

The dataset I uploaded is already the resized 1024*512 version so the threshold of the GRID_SIZE and THRESHOLD_IN_GRID_INFLUENCED are changed accordingly. 

