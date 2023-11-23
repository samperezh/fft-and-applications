import argparse # for parsing command line arguments
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2 as cv

class Fft:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=1, required=False)
    parser.add_argument('-i', type=str, default="moonlanding.png", required=False)

    args = parser.parse_args()

    img = ""

    # Mode 1
    def mode_1(self):
        print("mode 1")
        # simply perform the FFT and output a one by two subplot of the original image and next to it its Fourier transform. 

        # The Fourier transform should be log scaled. 
        # An easy way to do this with matplotlib is to import LogNorm from matplotlib.colors to produce a logarithmic colormap.

    # Mode 2
    def mode_2(self):
        print("mode 2")

    # Mode 3
    def mode_3(self):
        print("mode 3")

    # Mode 4
    def mode_4(self):
        print("mode 4")

    def convert_image_into_array(self):
        self.img = cv.imread(self.args.i, cv.IMREAD_GRAYSCALE)

        # If the image given doesn't have a length or width that is a power of 2 
        # resize it with cv2 otherwise the FFT algorithm might not work
        
        # get current width and height
        h, w = self.img.shape[:2]
        if (w & (w-1)) != 0  | (h & (h-1)) !=0 : 
            w = int(2 ** np.ceil(np.log2(w)))
            h = int(2 ** np.ceil(np.log2(h)))
            self.img = cv.resize(self.img, (w,h))

       # cv.imshow("Display window", self.img)
       # K = cv.waitKey(0) # Wait for a keystroke in the window


if __name__ == "__main__":
    # program starts running here
    fft = Fft()

    # convert image into a NumPy array
    fft.convert_image_into_array()

    if(fft.args.m == 1):
        fft.mode_1()
    elif(fft.args.m == 2):
        fft.mode_2()
    elif(fft.args.m == 3):
        fft.mode_3()
    elif(fft.args.m == 4):
        fft.mode_4()


