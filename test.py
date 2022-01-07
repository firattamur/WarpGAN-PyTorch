import argparse
import sys

def main(args):
    pass

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir", help="The path to the pretrained model",
                            type=str)

    parser.add_argument("input", help="The path to the aligned image",
                            type=str)

    parser.add_argument("output", help="The prefix path to the output file, subfix will be added for different styles.",
                            type=str, default=None)

    parser.add_argument("--num_styles", help="The number of images to generate with different styles",
                            type=int, default=5)

    parser.add_argument("--scale", help="The path to the input directory",
                            type=float, default=1.0)

    parser.add_argument("--aligned", help="Set true if the input face is already normalized",
                            action='store_true')
    

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
