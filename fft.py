import argparse # for parsing command line arguments
import numpy as np
import matplotlib.pyplot as plt

class Fft:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=1, required=False)
    parser.add_argument('-i', type=str, default="moonlanding.png", required=False)

    args = parser.parse_args()

    def fast_mode(self):
        print("fast mode")

    def denoising(self):
        print("denoising")

    def compressing(self):
        print("compressing")

    def plotting(self):
        print("plotting")

if __name__ == "__main__":
    # program starts running here
    fft = Fft()

    if(fft.args.m == 1):
        fft.fast_mode()
    elif(fft.args.m == 2):
        fft.denoising()
    elif(fft.args.m == 3):
        fft.compressing()
    elif(fft.args.m == 4):
        fft.plotting()


