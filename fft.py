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

        keep_fraction = 0.2
        transformed_img = self.fft_dft_2d(self.img)
        # transformed_img = self.naive_dft_2d(self.img)
        h, w = transformed_img.shape[:2]

        # https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html

        # transformed_img[int(h * keep_fraction) : int(h*(1-keep_fraction))] = 0
        # transformed_img[:, int(w*keep_fraction) : int(w * (1-keep_fraction))] = 0

        for row in range(int(h * keep_fraction), int(h * (1 - keep_fraction))):
            transformed_img[row, :] = 0

        for column in range(int(w * keep_fraction), int(w * (1 - keep_fraction))):
            transformed_img[:, column] = 0

        print("transformed" + str(transformed_img))
        number_of_non_zeros = np.count_nonzero(transformed_img)

        # Invert the denoised image
        # transformed_img = self.fft_dft_2d_inverse(transformed_img).real
        transformed_img = self.naive_dft_2d_inverse(transformed_img).real

        # Plot the original image and the denoised image
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(self.img, cmap='gray')
        axs[0].set_title('Original Image')
        axs[1].imshow(transformed_img.real, cmap='gray')
        axs[1].set_title('Denoised Image')
        plt.show()

        print("Number of non-zeros: " + str(number_of_non_zeros))
        print("Fraction of non-zeros: " + str(number_of_non_zeros / (h * w)))

    # Mode 3
    def mode_3(self):
        print("mode 3")

    # Mode 4
    def mode_4(self):
        print("mode 4")

    def convert_image_into_array(self):
        self.img = cv.imread(self.args.i, cv.IMREAD_GRAYSCALE)
        # self.img = np.array([
        #     [10, 20, 30],
        #     [40, 50, -60],
        #     [70, 80, 90]
        # ])

        # self.img = np.array([
        #     [5, 4, 6, 3, 7, 8, 10, 24],
        #     [-1, -3, -4, -7, 0, -1, 2, 4]
        # ])

        # self.img = np.array([
        #     [1, 2, 3, 4, 5, 6, 7, 8],
        #     [6, 7, 8, 9, 10, 11, 12, 13],
        #     [11, 12, 13, 14, 15, 16, 17, 18],
        #     [11, 12, 13, 14, 15, 16, 17, 18]
        # ])

        # self.img = np.array([
        #     [1, 2, 3, 4, 5, 6, 7],
        #     [6, 7, 8, 9, 10, 11, 12],
        #     [11, 12, 13, 14, 15, 16, 17],
        #     [11, 12, 13, 14, 15, 16, 17]
        # ])

        # If the image given doesn't have a length or width that is a power of 2 
        # resize it with cv2 otherwise the FFT algorithm might not work
        
        # get current width and height
        h, w = self.img.shape[:2]
        if (w & (w-1)) != 0  | (h & (h-1)) !=0 : 
            w = int(2 ** np.ceil(np.log2(w)))
            h = int(2 ** np.ceil(np.log2(h)))
            self.img = cv.resize(self.img, (w,h))

        # temp = [1, 2, 3, 4, 5, 6, 7, 8]
        # X = self.fft_dft_1d_inverse(temp)
        # print(str(X))
        # print("fft")
        # gfg = np.fft.ifft(temp) 
        # #gfg = np.fft.fft(temp) 
        # print(gfg)

        temp = [[5, 4, 6, 3, 7, 8, 10, 24], [-1, -3, -4, -7, 0, -1, 2, 4]]
        self.fft_dft_2d_inverse(temp)

        # cv.imshow("Display window", img)
        # K = cv.waitKey(0) # Wait for a keystroke in the window

    # Naive
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
    
    # FFT
    
    def fft_dft_1d(self, img_1D_array):
        # we can assume that the size of img_1D_array is a power of 2 as when 
        # converting the original image into a NumPy array, we resize it so that it is.
        N = len(img_1D_array)
        # TODO to CHOOSE
        if N <= 2: # stop splitting the problem and use naive method instead
            # We chose to stop splitting the problems at 64
            # We just want the runtime of your FFT to be in the same order of magnitude as what is theoretically expected 
            return self.naive_dft_1d(img_1D_array)
        else:
            # Split the sum in the even and odd indices which we sum separately and then put together
            X = np.zeros(N, dtype=complex)
            # list[start:end:step].
            even = self.fft_dft_1d(img_1D_array[0::2])
            odd = self.fft_dft_1d(img_1D_array[1::2])

            for k in range (N //2):
                X[k] = even[k] + np.exp((-2j * np.pi * k) / N) * odd[k]
                X[k + (N // 2)] = even[k] + np.exp((-2j * np.pi * (k + (N // 2))) / N) * odd[k]
            return X
        
    def fft_dft_1d_inverse(self, img_1D_array):
        # we can assume that the size of img_1D_array is a power of 2 as when 
        # converting the original image into a NumPy array, we resize it so that it is.
        N = len(img_1D_array)
        # TODO to CHOOSE
        if N <= 2:
            return self.naive_dft_1d_inverse(img_1D_array)
        else:
            # Split the sum in the even and odd indices which we sum separately and then put together
            x = np.zeros(N, dtype=complex)
            # list[start:end:step].
            even = self.fft_dft_1d_inverse(img_1D_array[0::2])
            odd = self.fft_dft_1d_inverse(img_1D_array[1::2])

            for k in range (N //2):
                x[k] = (1/2)* (even[k] + np.exp((2j * np.pi * k) / N) * odd[k])
                x[k + (N // 2)] = (1/2)* (even[k] + np.exp((2j * np.pi * (k + (N // 2))) / N) * odd[k])

            return x
        
    def fft_dft_2d_inverse(self, img_2D_array):
        complex_img_array = np.asarray(img_2D_array, dtype=complex)
        h, w = complex_img_array.shape[:2]

        F = np.zeros((h, w), dtype=complex)

        for row in range(h):
            F[row, :] = self.fft_dft_1d_inverse(complex_img_array[row, :])

        for column in range(w):
            F[:, column] = self.fft_dft_1d_inverse(F[:,column])

        # print(str(F))
        # print("fft2")
        # gfg = np.fft.ifft2(img_2D_array) 
        # print(gfg)
        return F   
    
    def fft_dft_2d(self, img_2D_array):
        complex_img_array = np.asarray(img_2D_array, dtype=complex)
        h, w = complex_img_array.shape[:2]

        F = np.zeros((h, w), dtype=complex)

        for row in range(h):
            F[row, :] = self.fft_dft_1d(complex_img_array[row, :])

        for column in range(w):
            F[:, column] = self.fft_dft_1d(F[:,column])

        # print(str(F))
        # print("fft2")
        # gfg = np.fft.fft2(img_2D_array) 
        # print(gfg)
        return F 


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


