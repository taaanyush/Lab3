import sys
from tkinter import Tk, simpledialog, filedialog, messagebox

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Tk().withdraw()
FILE_PATH = filedialog.askopenfilename()
if not FILE_PATH:
    sys.exit()

fileValues = pd.read_csv(FILE_PATH, delimiter='\s+', header=None, names=["points", "y"])
points = fileValues.iloc[:, [0, 1]].values
N = len(points)
y = np.zeros(N)
num_of_clusters = simpledialog.askinteger("Input", "Write amount of clusters!")
if not num_of_clusters:
    sys.exit()

if num_of_clusters < 1:
    messagebox.showerror("Error", "Input more than 0 clusters!")
    sys.exit()


def k_means(num_cluster, x_point, y_point, points_number):
    is_last_step = True
    is_first_step = True
    prev_centroid = []
    centroid = []
    avg_arr = []
    avg = 0
    while is_last_step:
        if is_first_step:
            is_first_step = False
            start_point = np.random.choice(range(points_number), num_cluster, replace=False)
            centroid = x_point[start_point]
        else:
            prev_centroid = np.copy(centroid)
            for i in range(num_cluster):
                centroid[i] = np.mean(x_point[y_point == i], axis=0)
        for i in range(points_number):
            dist = np.sum((centroid - x_point[i]) ** 2, axis=1)
            avg_arr.append(min(dist))
            min_ind = np.argmin(dist)
            y_point[i] = min_ind
        if np.array_equiv(centroid, prev_centroid):
            avg = np.mean(avg_arr)
            is_last_step = False
    avg_arr.clear()
    return avg, x_point, y_point


sse = []
means, x, y = k_means(num_of_clusters, points, y, N)
sse.append(means)
matplotlib.rc('figure', figsize=(10, 10))
for k in range(num_of_clusters):
    fig = plt.scatter(x[y == k, 0], x[y == k, 1])
plt.show()

for i in range(1, num_of_clusters):
    y = np.copy(y)
    means, x, y = k_means(i, x, y, N)
    sse.append(means)

matplotlib.rc('figure', figsize=(10, 5))
sse.sort(reverse=True)
plt.plot(list(range(1, num_of_clusters + 1)), sse)
plt.xticks(list(range(1, num_of_clusters + 1)))
plt.scatter(list(range(1, num_of_clusters + 1)), sse)
plt.xlabel("Amount of Clusters")
plt.ylabel("SSE")
plt.show()
