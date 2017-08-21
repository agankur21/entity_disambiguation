import os
import sys
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))


def get_line_count(data_file):
    count = 0
    with open(data_file, 'r') as f:
        for _ in f:
            count += 1
    return count


def get_train_valid_test_split(data_file, split_percentage=0.05, out_dir=None):
    total_indices = np.array(range(get_line_count(data_file)))
    random.shuffle(total_indices)
    print "Line Count : %d" % len(total_indices)
    valid_count = test_count = int(split_percentage * len(total_indices))
    if out_dir is None:
        out_dir = os.path.dirname(data_file)
    train_count = len(total_indices) - valid_count - test_count
    train_file = open(os.path.join(out_dir, 'train.txt'), 'w')
    valid_file = open(os.path.join(out_dir, 'valid.txt'), 'w')
    test_file = open(os.path.join(out_dir, 'test.txt'), 'w')
    train_indices = total_indices[0:train_count]
    valid_indices = total_indices[train_count:train_count + valid_count]
    test_indices = total_indices[train_count + valid_count:]
    print "Train Count : %d" % len(train_indices)
    print "Test Count : %d" % len(test_indices)
    print "Valid Count : %d" % len(valid_indices)
    total_lines = np.array(open(data_file, 'r').read().splitlines())
    train_data = total_lines[np.array(train_indices)]
    test_data = total_lines[np.array(test_indices)]
    valid_data = total_lines[np.array(valid_indices)]
    train_file.write("\n".join(train_data))
    test_file.write("\n".join(test_data))
    valid_file.write("\n".join(valid_data))
    test_file.close()
    valid_file.close()
    train_file.close()


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        print "No input arguments"
        sys.exit(1)
    elif len(args) == 2:
        if not os.path.isfile(args[1]):
            print("Incorrect file path")
            sys.exit(1)
        get_train_valid_test_split(args[1])
    elif len(args) == 3:
        get_train_valid_test_split(args[1], args[2])
    else:
        get_train_valid_test_split(args[1], args[2], args[3])
