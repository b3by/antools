import ast
import os
import multiprocessing
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

    Parameter
    ---------
    ds_location : str
        The path of the folder containing all the exercise folders.
    coordinates : str
        The path of the coordinate file with all the files to include.
    exercises : list
        The exercises to include (defaulted to None, so all included).

    Returns
    -------
    list
        A list of tuples, where each tuple contains the full path of a file in
        the first element, and a list of coordinates (tuples) in the second
        element.
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

    Parameters
    ----------
    signals : list
        A list of signals from which the windows should be extracted.
    window_size : int
        The size of the windows.
    stride : int
        The stride between consecutive windows.

    Yields
    ------
    list
        A list, where each element represents a window over one of the input
        signals. For each window, the yielded list will have as many elements
        as the signals provided in the input.
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

    Parameters
    ----------
    exercise_file : str
        The path of the exercise file containing all the signals.
    crds : list
        The coordinates for the current exercise.
    sensors : list
        A list of the sensors for which the signals should be included.
    window_size : int
        The size of each window.
    stride : int
        Stride between subsequent windows.

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe containing all the retrieved windows for the
        exercise provided as input.
    tuple
        A tuple containing the counting for the windows (silent and movement).
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

    cols = ['{}{}'.format(a, i)
            for a in cls
            for i in range(window_size * len(sensors))]

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
                   test_size=0.2, normalize=None, binary=False, tts_seed=42,
                   procs=None):
    """Generate train and test datasets, write them to file

    This method parses all the files in a given dataset, and aggregates them
    in two separate datasets, one for train and one for test.

    Parameters
    ----------
    dataset : str
        The location of the dataset.
    train_dst : str
        Destination for the train dataset (full path).
    test_dst : str
        Destination for the test dataset (full path).
    crds : str
        Coordinates file (full path).
    target_sensor : list
        A list of sensors that should be included in the dataset.
    window_size : int
        The size of the window to chunk the signal.
    stride : int
        The stride between consecutive windows.
    exercises : list
        The set of exercises to include in the dataset. It is defaulted to
        None, and the default value will include all the exercises.
    max_files : int
        The number of files to include in the dataset (only the first max_files
        files will be included). Defaulted to -1, will include all the files.
    test_size : float
        The proportion of the test split.
    normalize : str
        Specifies the normalization method. Can be either minmax or zscore.
    binary : bool
        Whether the dataset should be built with binary or ternary classes.
    tts_seed : int
        The random seed for the train-test split. Defaulted to 42.
    procs : int
        The number of processes to spawn for the data generation. If not
        provided, all the available processes will be used.

    Returns
    -------
    list:
        A list of the ids used for the training split.
    list
        A list of the ids used for the testing split.
    """
    all_files = get_files(dataset, crds, exercises)
    subjects = {}

    if max_files > 0:
        all_files = all_files[:max_files]

    for f in all_files:
        f_name = os.path.basename(f[0])
        s_id = int(f_name.split('.')[1])

        if subjects.get(s_id, None) is None:
            subjects[s_id] = [f]
        else:
            subjects[s_id].append(f)

    subs = list(subjects.keys())
    train_s, test_s = train_test_split(subs, random_state=tts_seed)

    train_files = []
    for train_sub_id in train_s:
        train_files += subjects[train_sub_id]

    test_files = []
    for test_sub_id in test_s:
        test_files += subjects[test_sub_id]

    print('Training subjects: {}'.format(train_s))
    print('Testing subjects: {}'.format(test_s))

    tqdm.monitor_interval = 0

    train_frames = []
    train_args = [[tf[0], tf[1], target_sensor, window_size, stride,
                   normalize, binary] for tf in train_files]

    print('Producing training set...')
    with multiprocessing.Pool(procs or multiprocessing.cpu_count()) as pool:
        train_frames = pool.starmap(get_win, train_args)

    train_frames = [tf[0] for tf in train_frames]
    final_train = pd.concat(train_frames, sort=True)
    final_train.to_csv(train_dst, index=None, header=True)

    test_frames = []
    test_args = [[tf[0], tf[1], target_sensor, window_size, stride,
                  normalize, binary] for tf in test_files]

    print('Producing test set...')
    with multiprocessing.Pool(procs or multiprocessing.cpu_count()) as pool:
        test_frames = pool.starmap(get_win, test_args)

    test_frames = [tf[0] for tf in test_frames]
    final_test = pd.concat(test_frames, sort=True)
    final_test.to_csv(test_dst, index=None, header=True)

    return train_s, test_s


def generate_datasets(Flags, exercises=None, max_files=-1, test_size=0.2,
                      normalize=None, binary=False, tts_seed=42):
    """Generate datasets from flags

    This method provides a shortcut to call the generate_input method, without
    passing all the arguments one by one.

    Parameters
    ----------
    Flags : types.SimpleNamespace
        The namespace containing all the required arguments for the
        generate_input method.
    exercise : list
        The list of the exercise to include in the generated dataset. It is
        defaulted to None, by which all exercises will be included.
    max_files : int
        Maximum number of files to include in the final dataset. It is
        defaulted to -1, which includes all the files.
    test_size : float
        The proportion of subjects to include in the test dataset (def 0.2).
    normalize : str
        Normalization method, can be either minmax or zscore. By default, no
        normalization is applied to the data.
    binary : bool
        The type of labels to generate in the output datasets. If True, only
        two classes will be generated (silence and movement), while False will
        result in a dataset annotated with three classes (silence, transition,
        and movement). Defaulted to False.
    tts_seed : int
        The random seed for the train-test split. Defaulted to 42.

    Returns
    -------
    list:
        A list of the ids used for the training split.
    list
        A list of the ids used for the testing split.
    """
    return generate_input(Flags.dataset_location, Flags.train, Flags.test,
                          Flags.coordinates, Flags.sensors, Flags.window_size,
                          Flags.stride, exercises=Flags.exercises,
                          max_files=max_files, test_size=test_size,
                          normalize=normalize, binary=binary,
                          tts_seed=tts_seed)


def get_tf_train_test(train_file_loc, test_file_loc, height, width, depth):
    """Get training and testing datasets

    This method reads train and test files in, then reshapes them according to
    the dimensions provided in the input. The labels are returned in form of
    dummy vectors.

    Parameters
    ----------
    train_file_loc : str
        Location of the training file.
    test_file_loc : str
        Location of the testing file.
    height : int
        The desired height of the outuput.
    width : int
        The desired width of the output.
    depth : int
        The desired depth of the output.

    Returns
    -------
    numpy.ndarray
        The training X dataset.
    numpy.ndarray
        The training Y dataset.
    numpy.ndarray
        The test X dataset.
    numpy.ndarray
        The test Y dataset.
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

    Parameter
    ---------
    Flags : types:SimpleNamespace
        A SimpleNamespace containing all the fields for get_tf_train_test.

    Returns
    -------
    numpy.ndarray
        The training X dataset.
    numpy.ndarray
        The training Y dataset.
    numpy.ndarray
        The test X dataset.
    numpy.ndarray
        The test Y dataset.
    """
    return get_tf_train_test(Flags.train, Flags.test, Flags.input_height,
                             Flags.input_width, Flags.channels)
