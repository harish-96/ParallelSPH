import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import random
import os
import sys

outdir = sys.argv[1]
geom_file = sys.argv[2]

path, dirs, files = next(os.walk(outdir))
file_count = len(files) - 1
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
line, = axes.plot(xdata, ydata, 'o', markersize=0.5)

df = pd.read_csv(outdir + "/0.csv")
npts = len(df['X pos'])

x = np.zeros((file_count, npts))
y = np.zeros((file_count, npts))

for i in range(file_count):
    stt = outdir + "/" + str(i) + '.csv'
    df = pd.read_csv(stt)
    x[i] = df['X pos']
    y[i] = df['Y pos']
    
df = pd.read_csv(geom_file, header=None)
x_geom = df[0]
y_geom = df[1]

for i in range (file_count):

    line.set_xdata(np.append(x[i], x_geom))
    line.set_ydata(np.append(y[i], y_geom))
    plt.draw()
    plt.xlim(-3,5)
    plt.ylim(-2,3)
    plt.pause(1e-17)
    time.sleep(0.001)

