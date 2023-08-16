__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Initiates an instance of knowit given some arguments.'

import os
import argparse


def main():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description='Run an experiment with Knowit.')
    parser.add_argument('--action', action='store', type=str, default=None,
                        help='what action should knowit take')
    parser.add_argument('--data', action='store', type=str, default=None,
                        help='what data should knowit perform the action on')
    parser.add_argument('--model', action='store', type=str, default=None,
                        help='what model should knowit perform the action on')
    parser.add_argument('--args_file', action='store', type=str, default=None,
                        help='additional arguments')
    args = parser.parse_args()

    # dummy(args.ref_image, args.target_image, args.mode_file)




if __name__ == '__main__':
    main()

