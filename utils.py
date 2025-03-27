from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import math

def process_request(request):
    request['messages'][0]['content'][1]['image_url'] = 'ignore'

    return request

def id_to_vector(ids, len_dim=9):
    # not perfect inversion of function vector_to_id
    anomalies = np.zeros(len_dim)
    for id in ids:
        try:
            anomalies[int(id)] = 1
        except Exception:
            continue

    return anomalies

def vector_to_id(multi_vector):
    ids = []

    for i in range(multi_vector.shape[1]):
        vector = multi_vector[:, i]
        # Ignore NaN values and check if all remaining elements are 1
        if np.all(vector[np.isnan(vector) == False] == 1):
            ids.append(i)

    return ids

def point_to_vector(points, len_vector=400):
    anomalies = np.zeros(len_vector)
    for point in points:
        try:
            anomalies[int(point)] = 1
        except Exception:
            continue

    return anomalies

def vector_to_point(vector):
    points = [i for i, x in enumerate(vector) if x == 1]

    return points

def interval_to_vector(interval, start=0, end=400, pred=False):
    anomalies = np.zeros(end - start)
    for entry in interval:
        if len(entry) !=2 :
            continue
        try:
            entry = {'start': int(entry[0]), 'end': int(entry[1])}
            entry['end'] = entry['end'] + 1 if pred else entry['end']
            entry['start'] = np.clip(entry['start'], start, end)
            entry['end'] = np.clip(entry['end'], entry['start'], end)
            anomalies[entry['start']:entry['end']] = 1
        except (ValueError, IndexError, TypeError) as e:
            continue  # Skip the current entry and move to the next
    
    return anomalies

def vector_to_interval(vector):
    intervals = []
    in_interval = False
    start = 0
    for i, value in enumerate(vector):
        if value == 1 and not in_interval:
            start = i
            in_interval = True
        elif value == 0 and in_interval:
            intervals.append((start, i))
            in_interval = False
    if in_interval:
        intervals.append((start, len(vector)))
    
    return intervals

def nearest_square_root(n):
    lower_sqrt = math.floor(math.sqrt(n))
    upper_sqrt = math.ceil(math.sqrt(n))

    lower_square = lower_sqrt ** 2
    upper_square = upper_sqrt ** 2

    return lower_sqrt if abs(lower_square - n) <= abs(upper_square - n) else upper_sqrt

def create_color_generator(exclude_color='blue'):
    # Get the default color list
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
    # Filter out the excluded color
    filtered_colors = [color for color in default_colors if color != exclude_color]
    # Create a generator that yields colors in order
    return (color for color in filtered_colors)

def plot_rectangle_stack_series(
    series,
    gt_anomaly,
    single_series_figsize: tuple[int, int] = (10, 10),
    gt_color: str = 'steelblue',
    train_eval: str = 'train'
) -> None:
    stream_length, dim = series.shape

    # Calculate the optimal number of rows and columns for a rectangular layout
    rows = int(math.sqrt(dim))
    cols = math.ceil(dim / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(single_series_figsize[0] * cols / rows, single_series_figsize[1]))
    fig.subplots_adjust(hspace=0, wspace=0)

    # Plot each univariate time series in its subplot
    for idx in range(rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        if idx < dim:
            ax.plot(series[:, idx], color=gt_color)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Turn off unused subplots
            ax.axis('off')

        if train_eval == 'train' and gt_anomaly is not None:
            if isinstance(gt_anomaly[0], int) and idx in gt_anomaly:
                ax.lines[-1].set_color('red')

    plt.tight_layout()
    return plt.gcf()

def plot_series(
    series,
    gt_anomaly,
    single_series_figsize: tuple[int, int] = (10, 1.5),
    # gt_ylim: tuple[int, int] = (-1, 1),
    gt_color: str = 'steelblue',
    train_eval: str = 'train'
) -> None:
    plt.figure(figsize=single_series_figsize)

    # plt.ylim(gt_ylim)
    plt.plot(series, color=gt_color)

    if train_eval == 'train':
        if gt_anomaly is not None:
            if isinstance(gt_anomaly[0], tuple):
                for start, end in gt_anomaly:
                    plt.axvspan(start, end-1, alpha=0.2, color=gt_color)
            elif isinstance(gt_anomaly[0], int):
                for point in gt_anomaly:
                    plt.axvline(x=point, color=gt_color, alpha=0.5, linestyle='--')

    plt.tight_layout()
    return plt.gcf()

def view_base64_image(base64_string):
    import base64
    from io import BytesIO
    from PIL import Image
    import matplotlib.pyplot as plt

    # Decode the base64 string to binary data
    image_data = base64.b64decode(base64_string)
    
    # Convert binary data to an image
    image = Image.open(BytesIO(image_data))
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()
