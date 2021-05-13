import numpy as np
import cv2,os
from scipy import ndimage as ndi

THRESHOLD_EDGE = 20
THRESHOLD_DIFFERENCE_RATIO = 0.8
GRID_SIZE = 32
THRESHOLD_IN_GRID_INFLUENCED = 6

def forward_energy(im):

    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
        
    return energy

if __name__ == '__main__':

    if not os.path.exists("./pix2pixHD/datasets/AMD/train_A/"):
        os.makedirs("./pix2pixHD/datasets/AMD/train_A/")
    if not os.path.exists("./pix2pixHD/datasets/AMD/train_B/"):
        os.makedirs("./pix2pixHD/datasets/AMD/train_B/")
    if not os.path.exists("./pix2pixHD/datasets/AMD/test_A/"):
        os.makedirs("./pix2pixHD/datasets/AMD/test_A/")
    if not os.path.exists("./pix2pixHD/datasets/AMD/test_B/"):
        os.makedirs("./pix2pixHD/datasets/AMD/test_B/")
    for i in range(1,439):
        _id = f"{i}"
        print(_id)
        IM_PATH_ORIGINAL = "./pix2pixHD/datasets/AMD/original/" + _id + ".png"
        IM_PATH_MODIFIED = "./pix2pixHD/datasets/AMD/AMD/" + _id + ".png"
        OUTPUT_ORIGINAL_1 = "./pix2pixHD/datasets/AMD/train_A/" + _id + ".png"
        OUTPUT_GENERATED_1 = "./pix2pixHD/datasets/AMD/train_B/" + _id + ".png"
        OUTPUT_ORIGINAL_2 = "./pix2pixHD/datasets/AMD/test_A/" + _id + ".png"
        OUTPUT_GENERATED_2 = "./pix2pixHD/datasets/AMD/test_B/" + _id + ".png"
        im_1 = cv2.imread(IM_PATH_ORIGINAL)
        assert im_1 is not None
        output_1 = forward_energy(im_1)
        #output_1 = np.clip(output_1 * (255 / np.amax(output_1)), 0, 255)
        h, w = im_1.shape[:2]

        im_2 = cv2.imread(IM_PATH_MODIFIED)
        assert im_2 is not None
        output_2 = forward_energy(im_2)
        #output_2 = np.clip(output_2 * (255 / np.amax(output_2)), 0, 255)
        output = np.divide(np.abs(output_1 - output_2), output_1, out=np.zeros_like(output_1), where=output_1!=0) 
        #the difference of the original and modified images
        output[np.where(output_1 < THRESHOLD_EDGE)] = 0 #there is an edge here in the original image
        output[output >= THRESHOLD_DIFFERENCE_RATIO] = 255 #the threshold of whether the difference is significant
        output[output < THRESHOLD_DIFFERENCE_RATIO] = 0

        for j in range(0, int(h / GRID_SIZE)):
            for k in range(0, int(w / GRID_SIZE)):
                if (output[GRID_SIZE * j : GRID_SIZE * (j + 1), GRID_SIZE * k : GRID_SIZE * (k + 1)] == 255).sum() > THRESHOLD_IN_GRID_INFLUENCED:
                    output[GRID_SIZE * j : GRID_SIZE * (j + 1), GRID_SIZE * k : GRID_SIZE* (k + 1)] = 255
                else:
                    output[GRID_SIZE * j : GRID_SIZE * (j + 1), GRID_SIZE * k : GRID_SIZE* (k + 1)] = 0

        if(i < 401):
            cv2.imwrite(OUTPUT_ORIGINAL_1, im_1)
            cv2.imwrite(OUTPUT_GENERATED_1, output)
        else:
            cv2.imwrite(OUTPUT_ORIGINAL_2, im_1)
            cv2.imwrite(OUTPUT_GENERATED_2, output)

