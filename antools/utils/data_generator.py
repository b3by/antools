import ast
import os
import multiprocessing as mp
import shutil
import glob

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from . import clprint


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


def get_win(exercise_file, crds, sensors, win_size, stride, dst,
            normalize=None, win_t=0.25):
    """Extract windows

    This method reads in an exercise file, and produces a data file containing
    all the linearized windows produced for that file, with a corresponding
    label for each window. A label of 1 represent a window where movement is
    detected, whilst a label of 0 is assigned to windows where no movement is
    detected. The method directly writes the produced windows in a file
    specified by the dst argument, without using pandas in the process.

    Parameters
    ----------
    exercise_file : str
        The path of the exercise file containing all the signals.
    crds : list
        The coordinates for the current exercise.
    sensors : list
        A list of the sensors for which the signals should be included.
    win_size : int
        The size of each window.
    stride : int
        Stride between subsequent windows.
    dst : str
        The destination file of the generated windows.
    normalize : str
        If a normalization policy is specified, the data will be normalized.
        Possible normalization methods are minmax and zscore. Def. to None.
    win_t : float
        Represents the threshold to use when assigning labels to the windows.
        The default is 0.25, and is passed as percentage value.

    Returns
    -------
    str
        If the process was successful, the method returns the full path of the
        produced file.

    Raises
    ------
    Exception
        A generic exception is meant to indicat the conversion failure of the
        passed exercise files. If that's the case, the method returns None.
        This exception will be further detailed in the future.

    """
    exr = pd.read_csv(exercise_file, sep=',')
    exr = exr.dropna(axis=0)

    try:
        cls = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        col_list = list(exr.columns)

        if normalize == 'minmax':
            for col in col_list:
                exr[col] = (exr[col] -
                            exr[col].min()) / (exr[col].max() - exr[col].min())
        elif normalize == 'zscore':
            for col in col_list:
                exr[col] = (exr[col] - exr[col].mean()) / exr[col].std(ddof=0)

        all_sig = []

        if 'NULL' in sensors:
            all_sig += [exr[i] for i in cls]
        else:
            for t in sensors:
                all_sig += [exr[i + '_' + t] for i in cls]

        cols = ['{}{}'.format(a, i) for a in cls for i in range(win_size *
                                                                len(sensors))]

        cols.append('label')

        win_thresh = int(win_size * win_t) * 0
        outfile_path = os.path.join(dst, os.path.basename(exercise_file))

        with open(outfile_path, 'w') as outfile:
            outfile.write(','.join(cols) + '\n')

            for i, win in enumerate(next_window(all_sig, win_size, stride)):
                win_start = i * stride
                win_end = win_start + win_size

                if any(c[0] < (win_end - win_thresh) and c[1] > win_start +
                       win_thresh for c in crds):
                    outfile.write(','.join(map(str, win + [1])) + '\n')
                else:
                    outfile.write(','.join(map(str, win + [0])) + '\n')

        return outfile_path
    except Exception:
        clprint.ptrun('  Exception for {}'.format(exercise_file))
        return None


def generate_input(dataset, train_dst, test_dst, crds, target_sensor,
                   window_size, stride, exercises=None, max_files=-1,
                   test_size=0.2, normalize=None, tts_seed=42, parallel=False,
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

    clprint.ptinfo('Training subjects: {}'.format(train_s))
    clprint.ptinfo('Testing subjects: {}'.format(test_s))

    train_temp_dst = os.path.join(dataset, 'temp_file_concat', 'train')
    test_temp_dst = os.path.join(dataset, 'temp_file_concat', 'test')
    os.makedirs(train_temp_dst, exist_ok=True)
    os.makedirs(test_temp_dst, exist_ok=True)

    clprint.ptinfo('  Train temp destination: {}'.format(train_temp_dst))
    clprint.ptinfo('  Test temp destination: {}'.format(test_temp_dst))

    train_args = [[tf[0], tf[1], target_sensor, window_size, stride,
                   train_temp_dst, normalize] for tf in train_files]

    test_args = [[tf[0], tf[1], target_sensor, window_size, stride,
                  test_temp_dst, normalize] for tf in test_files]

    parallelize_window_generation_imap(train_args, procs=4)
    concatenate_and_save(train_temp_dst, train_dst)

    parallelize_window_generation_imap(test_args, procs=4)
    concatenate_and_save(test_temp_dst, test_dst)

    shutil.rmtree(train_temp_dst)
    shutil.rmtree(test_temp_dst)

    return train_s, test_s


def concatenate_and_save(source, destination):
    """Read in many files, spit out one single file

    This method reads all the csv files specified in the source folder, and
    returns the concatenation of the files in the destination file. The method
    expects to find the header in all the files in the source folder: only
    the first header will be included in the produced csv, while all the other
    headers will be ignored. The method brutally concatenates the files, so no
    check is performed over the number of columns.

    Parameters
    ---------
    source : str
        A full path of the folder containing the files to concatenate.
    destination : str
        The full path of the destination file.

    """
    clprint.ptinfo('Concatenating from {} to {}'.format(source, destination))
    allFiles = glob.glob(source + '/*.csv')

    with open(destination, 'wb') as outfile:
        for i, fname in enumerate(allFiles):
            with open(fname, 'rb') as infile:
                if i != 0:
                    infile.readline()

                shutil.copyfileobj(infile, outfile)


def serialize_window_generation(passed_args):
    """Extract windows in a serial fashion

    This method takes a list of arguments, and calls the get_win function for
    each one of them. The function is called sequentially.

    Parameters
    ----------
    passed_args : list
        A list of lists to unpack and pass to the get_win function.

    Returns
    -------
    list
        A list of tuples, where each tuple is composed by a list of arguments
        and the returned value from the get_win function (either the
        destination path or None).

    """
    tqdm.monitor_interval = 0
    written = []
    for a in tqdm(passed_args, desc='Processing frames'):
        written.append((a, get_win(*a)))

    return written


def parallelize_window_generation_imap(passed_args, procs=None):
    """Produce window files, in a parallel fashion

    This method calls the get_win function as many times as sets of arguments
    specified in passed_args. starmap is used to pass the list of arguments to
    each invocation of get_win. The pool is created with either the number of
    provided processors, or half the number of the available processors (be
    kind, don't allocate everything).

    Parameters
    ----------
    passed_args : list
        A list of lists, each one containing all the arguments to pass to an
        invocation of the get_win function.
    procs : int
        The number of processors to use. Defaulted to None, will use half of
        the available cores.

    Returns
    -------
    list
        A list containing the paths of all the results from the get_win calls.

    """
    pool = mp.Pool(procs or int(mp.cpu_count() / 2))
    results = pool.starmap(get_win, passed_args)

    pool.close()
    pool.join()

    return results


def generate_datasets(Flags, exercises=None, max_files=-1, test_size=0.2,
                      normalize=None, tts_seed=42):
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
                          normalize=normalize, tts_seed=tts_seed)


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
