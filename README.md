# ECSE 316 Assignment 2: Fast Fourier Transform and Applications
## November 24th, 2023

Note that this code was written and tested on macOS 13.2.1 (M1 chip) through VSCode terminal. 

The FFT application should be invoked at the command line using the following syntax: 

``` python3 fft.py [-m mode] [-i image] ```

where the arguments are defined as follows:
- mode (optional) indicates one of four available modes: 
    - [1] (default) for fast mode where the image is converted into its FFT form and displayed
    - [2] for denoising where the image is denoised by applying an FFT, truncating high frequencies and then displayed
    - [3] for compressing and plot the image
    - [4] for plotting the runtime graphs for the report
- image (optional) filename of the image we wish to take the DFT of. (Default: the file name of the image given to you for the assignment)

Python Version used: Python 3.11.1

Libraries used: 
- argparse (for parsing command line arguments)
- numpy
- matplotlib.pyplot
- matplotlib.colors
- cv2
- time
