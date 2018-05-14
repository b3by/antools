import ast
import os
import collections

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def get_files(ds_location, coordinates, exercises=None):
    """Collect file names for exercises

    In order to generate all the windows, a list of files for the target
    exercises is required. This method generates this list, by taking care of
    excluding all the exercises that are not required. In case the list of
    exclusions is empty, all the exercises will be included. The files to
    include are the segmented ones only, which are specified in the coordinates
    file.

    Arguments:

    ds_location -- the folder containing all the exercise folders
    coordinates -- the coordinate file with all the files to include
    exercises -- the exercises to include (defaulted to None, so all included)
    """
    c_df = pd.read_csv(coordinates)
    f_list = [(c[0], ast.literal_eval(c[1])) for c in list(zip(
        list(c_df['filename']),
        list(c_df['coordinates'])))]

    if exercises is not None:
        f_list = [f for f in f_list if f[0].split('.')[0] in exercises]

    f_list = [(os.path.join(ds_location, c[0].split('.')[0], c[0]), c[1]) for c
              in f_list]
    return [(c[0], list(zip(*[c[1][x::2] for x in (0, 1)]))) for c in f_list]


def next_window(signals, window_size, stride):
    """Rolling windows

    This method yields windows. That's how it rolls.
    """
    c_win = 0
    while c_win + window_size <= len(signals[0]):
        w = [list(signals[i][c_win:c_win + window_size])
             for i in range(len(signals))]
        yield [item for subsignal in w for item in subsignal]
        c_win += stride


def get_win(exercise_file, crds, sensors, window_size, stride,
            normalize=None, binary=False, win_t=0.25):
    """Extract windows

    This method returns all the windows contained in a single exercise file.
    The label assigned to each window depends on where the window is compared
    with the points in the coordinates file.

    Arguments:

    exercise_file -- the exercise file containing all the signals
    crds -- the coordinates for the current exercise
    sensors -- the sensors for which the signals should be included
    window_size -- the size of each window
    stride -- stride between subsequent windows
    """
    exr = pd.read_csv(exercise_file, sep=',')
    exr = exr.dropna(axis=0)

    cls = ['acc_x_', 'acc_y_', 'acc_z_', 'gyro_x_', 'gyro_y_', 'gyro_z_']

    col_list = list(exr.columns)

    if normalize == 'minmax':
        for col in col_list:
            exr[col] = (exr[col] -
                        exr[col].min()) / (exr[col].max() - exr[col].min())
    elif normalize == 'zscore':
        for col in col_list:
            exr[col] = (exr[col] - exr[col].mean()) / exr[col].std(ddof=0)

    all_signals = []

    for t in sensors:
        all_signals += [exr[i + t] for i in cls]

    cols = [f'{a}{i}' for a in cls for i in range(window_size * len(sensors))]
    cols.append('label')

    df = pd.DataFrame(columns=cols)
    cnt = collections.Counter()

    win_thresh = int(window_size * win_t) * 0

    for i, win in enumerate(next_window(all_signals, window_size, stride)):
        window_start = i * stride
        window_end = window_start + window_size

        if binary:
            if any(c[0] < (window_end - win_thresh) and c[1] > window_start +
                   win_thresh for c in crds):
                # movement
                df.loc[len(df)] = dict(zip(cols, win + [1]))
                cnt.update([1])
            else:
                # silent
                df.loc[len(df)] = dict(zip(cols, win + [0]))
                cnt.update([0])
        else:
            if any(c[0] < window_start and c[1] > window_end for c in crds):
                df.loc[len(df)] = dict(zip(cols, win + [2]))
                cnt.update([2])
            elif any((c[0] >= window_start and c[0] <= window_end) or
                     (c[1] >= window_start and c[1] <= window_end)
                     for c in
                     crds):
                df.loc[len(df)] = dict(zip(cols, win + [1]))
                cnt.update([1])
            else:
                df.loc[len(df)] = dict(zip(cols, win + [0]))
                cnt.update([0])

    df['label'] = df['label'].astype(int)

    return df, tuple(cnt.values())


def generate_input(dataset, train_dst, test_dst, crds, target_sensor,
                   window_size, stride, exercises=None, max_files=-1,
                   test_size=0.2, normalize=None, binary=False):
    """Generate train and test datasets, write them to file

    This method parses all the files in a given dataset, and aggregates them
    in two separate datasets, one for train and one for test.

    Arguments:

    dataset -- the location of the dataset
    train_dst -- destination for the train dataset
    test_dst -- destination for the test dataset
    crds -- coordinates file
    """
    all_files = get_files(dataset, crds, exercises)
    frames = []

    if max_files > 0:
        all_files = all_files[:max_files]

    for item in tqdm(all_files, desc='Extracting windows'):
        d, s = get_win(item[0], item[1], target_sensor, window_size, stride,
                       normalize=normalize, binary=binary)
        frames.append(d)

    final_frame = pd.concat(frames)
    train, test = train_test_split(final_frame, test_size=test_size)

    train.to_csv(train_dst, index=None, header=True)
    test.to_csv(test_dst, index=None, header=True)


def generate_datasets(Flags, exercises=None, max_files=-1, test_size=0.2,
                      normalize=None, binary=False):
    """Generate datasets from flags

    This method provides a shortcut to call the generate_input method, without
    passing all the arguments one by one.
    """
    generate_input(Flags.dataset_location, Flags.train, Flags.test,
                   Flags.coordinates, Flags.sensors, Flags.window_size,
                   Flags.stride, exercises=Flags.exercises,
                   max_files=max_files, test_size=test_size,
                   normalize=normalize, binary=binary)


def get_tf_train_test(train_file_loc, test_file_loc, height, width, depth):
    """Get training and testing datasets

    This method reads train and test files in, then reshapes them according to
    the dimensions provided in the input. The labels are returned in form of
    dummy vectors.

    Arguments:

    train_file_loc -- location of the training file
    test_file_loc -- location of the testing file
    height -- the desired height of the outuput
    width -- the desired width of the output
    depth -- the desired depth of the output
    """
    train_file = pd.read_csv(train_file_loc)
    test_file = pd.read_csv(test_file_loc)

    train_y = np.asarray(pd.get_dummies(train_file.label), dtype=np.int8)
    test_y = np.asarray(pd.get_dummies(test_file.label), dtype=np.int8)

    train_file = train_file.drop(['label'], axis=1)
    test_file = test_file.drop(['label'], axis=1)

    train_x = train_file.values
    test_x = test_file.values

    train_x = train_x.reshape(len(train_x), height, width, depth)
    test_x = test_x.reshape(len(test_x), height, width, depth)

    return train_x, train_y, test_x, test_y


def get_datasets(Flags):
    """Get training and testing datasets, but with Flags

    This method provides an interface to use the get_tf_train_test method by
    passing a namespace as argument.

    Arguments:

    Flags -- a SimpleNamespace containing all the fields for get_tf_train_test
    """
    return get_tf_train_test(Flags.train, Flags.test, Flags.input_height,
                             Flags.input_width, Flags.channels)
