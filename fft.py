import argparse # for parsing command line arguments

class Fft:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=1, required=False)
    parser.add_argument('-i', type=str, default="moonlanding.png", required=False)

    args = parser.parse_args()


if __name__ == "__main__":
    # program starts running here
    fft = Fft()
    print("mode: " + str(fft.args.m))
    print("image: " + fft.args.i)