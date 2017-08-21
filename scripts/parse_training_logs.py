import argparse
import operator
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from src.utils.decorator_utils import *


@raise_exception
def get_best_params(file_name, dir, print_best=True):
    best_p, p = 0.0, 0.0
    best_r, r = 0.0, 0.0
    best_f, f = 0.0, 0.0
    with open(os.path.join(dir, file_name), 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Precision"):
                p_str = line.split(":")[1].rstrip("%").strip()
                p = 0.0 if len(p_str) == 0 else float(p_str)
            elif line.startswith("Recall"):
                r_str = line.split(":")[1].rstrip("%").strip()
                r = 0.0 if len(r_str) == 0 else float(r_str)
            elif line.startswith("F-score"):
                f_str = line.split(":")[1].rstrip("%").strip()
                f = 0.0 if len(f_str) == 0 else float(f_str)
                if f > best_f:
                    best_p = p
                    best_r = r
                    best_f = f
    if print_best:
        print ("Best Parameters for File : %s ---> Recall : %0.3f ; Precision : %0.3f ; F-score : %0.3f" % (
            file_name, best_r, best_p, best_f))
    return best_f, best_p, best_r


def get_best_params_file(logs_dir):
    files = os.listdir(logs_dir)
    params_dict = {file_name: get_best_params(file_name, logs_dir) for file_name in files}
    file_name, best_params = max(params_dict.iteritems(), key=operator.itemgetter(1))
    best_f, best_p, best_r = best_params
    print ("Overall Best Parameters for File : %s ---> Recall : %0.3f ; Precision : %0.3f ; F-score : %0.3f" % (
        file_name, best_r, best_p, best_f))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dir', help="Directory of logs")
    arg_parser.add_argument('--file', help="log file")
    args = arg_parser.parse_args()
    if args.dir is None and args.file is None:
        raise IOError("Missing arguments")
    if args.dir is not None:
        if not os.path.isdir(args.dir):
            raise IOError("Directory does not exists")
        else:
            get_best_params_file(args.dir)

    if args.file is not None:
        if not os.path.isfile(args.file):
            raise IOError("File does not exists")
        else:
            get_best_params(args.file)
