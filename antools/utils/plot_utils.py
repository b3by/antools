import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_windows(df_location, win_labels, window_size, stride, crds=None):
    if crds is None:
        crds = []

    plt.figure(figsize=(20, 10))

    exr = pd.read_csv(df_location, sep=',')
    exr = exr.dropna(axis=0)
    exr['acc_y_arm'].plot()

    for idx, lb in enumerate(win_labels):
        start_interval = idx * stride
        end_interval = start_interval + window_size
        x_interval = [x for x in range(start_interval, end_interval)]

        if lb == 0:
            color = 'C2'
            level = 1
        elif lb == 1:
            color = 'C1'
            level = 2
        else:
            color = 'C3'
            level = 3

        plt.plot(x_interval, [level for _ in range(window_size)],
                 color=color)

    orange_patch = mpatches.Patch(color='C1', label='Class 1')
    green_patch = mpatches.Patch(color='C2', label='Class 0')
    red_patch = mpatches.Patch(color='C3', label='Class 2')

    for xc in crds:
        plt.axvline(x=xc, color='k', linestyle='--')

    plt.xlabel('Timestamp')
    plt.ylabel('Acceleration y')
    plt.legend(handles=[green_patch, red_patch, orange_patch])
    plt.savefig('./yoooooolo.png', bbox_inches='tight')
