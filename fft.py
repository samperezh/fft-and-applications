import argparse # for parsing command line arguments
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2 as cv
import time

class Fft:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=1, required=False)
    parser.add_argument('-i', type=str, default="moonlanding.png", required=False)

    args = parser.parse_args()

    img = ""

    # Mode 1
    def mode_1(self):
        print("mode 1")
        # simply perform the FFT and  
        transformed_img = self.fft_dft_2d(self.img)
        # Output a one by two subplot of the original image and next to it its Fourier transform.
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(self.img, cmap='gray')
        axs[0].set_title('Original Image')
        # np.abs(transformed_img) calculates the magnitude of the Fourier transform
        # norm=LogNorm() as the Fourier transform should be log scaled
        axs[1].imshow(np.abs(transformed_img), norm=LogNorm())
        axs[1].set_title('Fourier transform')
        plt.show()

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

        keep_fraction = 0.08
        transformed_img = self.fft_dft_2d(self.img)
        h, w = transformed_img.shape[:2]

        # https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html

        # transformed_img[int(h * keep_fraction) : int(h*(1-keep_fraction))] = 0
        # transformed_img[:, int(w*keep_fraction) : int(w * (1-keep_fraction))] = 0

        for row in range(int(h * keep_fraction), int(h * (1 - keep_fraction))):
            transformed_img[row, :] = 0

        for column in range(int(w * keep_fraction), int(w * (1 - keep_fraction))):
            transformed_img[:, column] = 0

        number_of_non_zeros = np.count_nonzero(transformed_img)

        # Invert the denoised image
        transformed_img = self.fft_dft_2d_inverse(transformed_img).real

        # Plot the original image and the denoised image
        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Mode 2 - Denoising')
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
        compression_levels = [0, 20, 45, 55, 75, 98] # in percentage

        fig, axs = plt.subplots(2, 3)
        fig.suptitle('Mode 3 - Compression')
        compression_level_index = 0
        for i in range(2):
            for j in range(3):
                compression_level = compression_levels[compression_level_index]
                number_of_non_zeros, compressed_image = self.compress_image_by_level(compression_level)
                print(f"Number of non-zeros for {compression_level}% compression: {number_of_non_zeros} out of {compressed_image.size}")
                axs[i, j].set_title(f"Compression level: {compression_level}%")
                axs[i, j].imshow(compressed_image.real, cmap='gray')
                compression_level_index += 1
        plt.show()

    def compress_image_by_level(self, compression_level):
        percent_of_img_to_keep = 100 - compression_level
        transformed_img = self.fft_dft_2d(self.img)

        high_frequencies = np.percentile(transformed_img.real, 100 - (percent_of_img_to_keep // 2))
        lower_frequencies = np.percentile(transformed_img.real, percent_of_img_to_keep // 2)

        transformed_img = transformed_img * np.logical_or(transformed_img >= high_frequencies, transformed_img <= lower_frequencies)

        number_of_non_zeros = np.count_nonzero(transformed_img)
        transformed_img = self.fft_dft_2d_inverse(transformed_img)

        return number_of_non_zeros, transformed_img

    # Mode 4
    # Produce plots that summarize the runtime of the naive and FFT algorithms
    def mode_4(self):
        print("=========================")
        print("***** Plotting Mode *****")
        print("=========================")
        # Create 2D arrays of random elements of various sizes (must be square and sizes must be powers of 2)
        # np.random.random ((N, N)) creates a 2D array of size NxN with random elements
        two_exp_5_arr = np.random.random((2**5,2**5))
        two_exp_6_arr = np.random.random((2**6,2**6))
        two_exp_7_arr = np.random.random((2**7,2**7))
        two_exp_8_arr = np.random.random((2**8,2**8))
        # not using 2^9x2^9 and 2^10x2^10 2D arrays as they take too long to run
        two_exp_9_arr = np.random.random((2**9,2**9))
        two_exp_10_arr = np.random.random((2**10,2**10))

        list_of_sizes = ["2^5 x 2^5", "2^6 x 2^6", "2^7 x 2^7", "2^8 x 2^8"]#, "2^9 x 2^9", "2^10 x 2^10"]
        list_of_arrays = [two_exp_5_arr, two_exp_6_arr, two_exp_7_arr, two_exp_8_arr]# two_exp_9_arr, two_exp_10_arr]

        naive_average_runtime = []
        naive_standard_deviations = []

        fft_average_runtime = []
        fft_standard_deviations = []

        # gather data fro the plot by re-running the experiment at least 10 times to obtain 
        # an average runtime for each problem size and a standard deviation
        for i in range(len(list_of_arrays)):
            # naive implementation
            runtime = []
            for j in range(10):
                start = time.time()
                self.naive_dft_2d(list_of_arrays[i])
                runtime.append(time.time() - start) # in seconds

            naive_average_runtime.append(np.mean(runtime))
            naive_standard_deviations.append(np.std(runtime))

            # fft implementation
            runtime = []
            for j in range(10):
                start = time.time()
                self.fft_dft_2d(list_of_arrays[i])
                runtime.append(time.time() - start) # in seconds

            fft_average_runtime.append(np.mean(runtime))
            fft_standard_deviations.append(np.std(runtime))

            # print in the command line the means and variances of the runtime of your algorithms vs the problem size
            print("*** Problem Size: "+ list_of_sizes[i] + " ***")
            print ("### Naive Implementation ###")
            print("Naive Average Runtime: " + str(naive_average_runtime[i]))
            print("Naive Standard Deviation: " + str(naive_standard_deviations[i]))
            print ("### FFT Implementation ###")
            print("FFT Average Runtime: " + str(fft_average_runtime[i]))
            print("FFT Standard Deviation: " + str(fft_standard_deviations[i]))
            print("=========================")

        # Plot:
        # problem size on x-axis & runtime in seconds on the y axis
        # two lines: one for the naive method and one for the FFT method
        # include error bars proportional to the standard deviation of the runtime 
        # representing confidence interval 
        # (97% confidence interval by making error bar length be twice the standard deviation)
        naive_error_bar_length = [i * 2 for i in naive_standard_deviations]
        fft_error_bar_length = [i * 2 for i in fft_standard_deviations]

        print(naive_error_bar_length)
        print(fft_error_bar_length)
        
        plt.title("Runtime of Naive and FFT Algorithms")
        plt.xlabel("Problem Size")
        plt.ylabel("Runtime (seconds)")

        # yerr adds error on y axis (algorithms runtime)
        # capsize: length of the error bar caps in points (default 0.0)
        # ecolor: colour of the error bars (default same as line)
        plt.errorbar(list_of_sizes, naive_average_runtime, yerr=naive_error_bar_length, capsize = 4, ecolor="lime",label="naive")
        plt.errorbar(list_of_sizes, fft_average_runtime, yerr=fft_error_bar_length, capsize = 4, ecolor="red", label="FFT")
        
        plt.legend()
        plt.show()
        


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

        # For testing purposes:

        # temp = [1, 2, 3, 4, 5, 6, 7, 8]
        # X = self.fft_dft_1d_inverse(temp)
        # print(str(X))
        # print("fft")
        # gfg = np.fft.ifft(temp) 
        # #gfg = np.fft.fft(temp) 
        # print(gfg)

        # temp = [[5, 4, 6, 3, 7, 8, 10, 24], [-1, -3, -4, -7, 0, -1, 2, 4]]
        # self.fft_dft_2d_inverse(temp)

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
        if N <= 64: # stop splitting the problem and use naive method instead
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
        if N <= 64:
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


