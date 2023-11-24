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
        # output a one by two subplot:
        # include the original image next to its denoised version. 

        # To denoise: take the FFT of the image and set all the high frequencies to zero 
        # before inverting to get back the filtered original. 
        # Where you choose to draw the distinction between a “high” and a “low” frequency is up to you to design and tune for to get the best result.

        # Note: The FFT plot you produce goes from 0 to 2π so any frequency close to zero can be considered low 
        # (even frequencies near 2π) since 2π is just zero shifted by a cycle. 
        # Your program should print in the command line the number of non-zeros you are using and 
        # the fraction they represent of the original Fourier coefficients.

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

        #temp = [[1, 2], [3, 4]]
        temp = [[5, 4, 6, 3, 7], [-1, -3, -4, -7, 0]]
        self.naive_dft_2d(temp)

        # cv.imshow("Display window", img)
        # K = cv.waitKey(0) # Wait for a keystroke in the window

    def naive_dft_1d(self, img_1D_array):
        # assign complex values to array
        N = len(img_1D_array)
        X = np.zeros(N, dtype=complex)
        for k in range(N): # RENAME VAR TO MATCH FORMULA
            for n in range(N):
                X[k] += img_1D_array[n] * np.exp((-2j * np.pi * k * n) / N)

        return X
    
    def naive_dft_1d_inverse(self, img_1D_array):
        N = len(img_1D_array)
        x = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                x[k] += img_1D_array[n] * (1/N) * (np.exp((2j * np.pi * k * n) / N))

        return x
    
    def naive_dft_2d(self, img_2D_array):
        complex_img_array = np.asarray(img_2D_array, dtype=complex)
        h, w = complex_img_array.shape[:2]

        F = np.zeros((h, w), dtype=complex)

        for row in range(h):
            F[row, :] = self.naive_dft_1d(complex_img_array[row, :])

        for column in range(w):
            F[:, column] = self.naive_dft_1d(F[:,column])

        # print(str(F))
        # print("fft2")
        # gfg = np.fft.fft2(img_2D_array) 
        # print(gfg)
        return F
    
    def naive_dft_2d_inverse(self, img_2D_array):
        complex_img_array = np.asarray(img_2D_array, dtype=complex)
        h, w = complex_img_array.shape[:2]

        f = np.zeros((h, w), dtype=complex)

        for row in range(h):
            f[row, :] = self.naive_dft_1d_inverse(complex_img_array[row, :])

        for column in range(w):
            f[:, column] = self.naive_dft_1d_inverse(f[:,column])

        return f

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


