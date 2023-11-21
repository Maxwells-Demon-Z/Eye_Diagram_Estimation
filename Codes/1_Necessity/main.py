
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import interpolate
from scipy import fft
import random
import time


# Parameters of signal.
T = 15e-9
PI = np.pi
ACC = 100
LEN = 15
MARGIN = 0.2

# Ideal gaussian distribution.
SIGMA = 0.025
jit_dist_t = np.linspace(-4*SIGMA, 4*SIGMA,
                         np.round(8*SIGMA*ACC).astype(int)+1)
jit_dist_y = 1 / SIGMA / np.sqrt(2*PI) * \
             np.exp(- np.square(jit_dist_t) / 2 / SIGMA**2)

# Plot.
plt.figure()
plt.plot(jit_dist_t, jit_dist_y)
plt.show()

# Uniform distributed time axis.
tim = np.linspace(0.0, LEN*T, LEN*ACC, endpoint=False)


# Extract data.
file = open('Test_Nonlinear/Without_Jitter.txt', 'r')
file_data = file.readlines()[1: -1]
wave = np.array([np.array(data.split('\t')) for data in file_data])
wave[:, -1] = np.array([data.replace('\n', '') for data in wave[:, -1]])
wave = wave.astype(float)
tim_sim = wave[:, 0]
sig_sim = wave[:, 1]

# Intepolate data.
f = interpolate.splrep(tim_sim, sig_sim)
sig = interpolate.splev(tim, f)

# Normalize data.
tim_norm = np.array(range(len(tim)))
sig_norm = np.round((sig - np.min(sig)) / (np.max(sig) - np.min(sig)) * ACC).astype(int)

# Create canvas.
margin = np.round(MARGIN * ACC).astype(int)
canvas = np.zeros([2*margin+ACC, len(tim)])
canvas[sig_norm+margin, tim_norm] = 100.0

# Plot.
c_map = cm.get_cmap('plasma')
c_map.set_bad('k')
canvas_cache = np.copy(canvas)
canvas_cache[canvas_cache == 0] = np.nan
plt.imshow(canvas_cache, interpolation='none', cmap=c_map, origin='lower')
plt.show()

# Convolve by jitter distribution.
canvas_con = [0.0] * (len(tim)+len(jit_dist_y)-1)
for row in list(canvas):
    cache = np.convolve(row, jit_dist_y)
    canvas_con = np.vstack((canvas_con, cache))
canvas_con = canvas_con[1:, :]

# Plot.
c_map = cm.get_cmap('plasma')
c_map.set_bad('k')
canvas_cache = np.copy(canvas_con)
canvas_cache[canvas_cache == 0] = np.nan
plt.imshow(canvas_cache, interpolation='none', cmap=c_map, origin='lower')
plt.show()


# Extract data.
file = open('Test_Nonlinear/With_Jitter.txt', 'r')
file_data = file.readlines()[1:]

# How many times to simulate.
head_loc = []
for row_num in range(len(file_data)):
    if file_data[row_num][0: 4] == 'Step':
        head_loc.append(row_num)

# Calculate weight.
param = np.copy(jit_dist_t)
param = np.vstack((np.tile(param, len(param)),
                   np.repeat(param, len(param))))
param = param.T
param = np.round((param+4*SIGMA)*ACC).astype(int)
weight = np.multiply(jit_dist_y[param[:, 0]],
                     jit_dist_y[param[:, 1]])

# Extract signal.
sig = []
for n in range(len(head_loc)):
    if n == len(head_loc) - 1:
        wave = file_data[head_loc[n] + 1:]
    else:
        wave = file_data[head_loc[n] + 1: head_loc[n + 1]]
    wave = np.array([np.array(data.split('\t')) for data in wave])
    wave[:, -1] = np.array([data.replace('\n', '') for data in wave[:, -1]])
    wave = wave.astype(float)
    tim_cache = wave[:, 0]
    sig_cache = wave[:, 1]
    f = interpolate.splrep(tim_cache, sig_cache, k=1)
    sig.append(interpolate.splev(tim, f))
sig = np.array(sig)

sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig))
sig = np.round(sig * ACC).astype(int)

margin = np.round(MARGIN * ACC).astype(int)
canvas = np.zeros([2*margin+ACC, len(tim)])
for n in range(len(weight)):
    canvas[sig[n] + margin, tim_norm] += weight[n]

# Plot.
c_map = cm.get_cmap('plasma')
c_map.set_bad('k')
canvas_cache = np.copy(canvas)
canvas_cache[canvas_cache == 0.0] = np.nan
plt.imshow(canvas_cache, interpolation='none', cmap=c_map, origin='lower')
plt.show()


#
# if _EN_DETERMINE_DELAY_:
#
#     # The delay of transmission line can be determined in the SPICE program.
#     # PYTHON is not required.
#     pass
#
#
# if _EN_TEST_JITTER_:
#
#     # Sequence charactors.
#     T = 20e-9
#     ACC = 1000
#     HIGH = 5.0
#     LEN = 100000
#     PI = np.pi
#
#     # Parameters of jitters.
#     TPJ = (4*PI) * T
#     APJ = 0.10
#     ARJ = 0.05
#
#     # Extract data.
#     file = open('Test_Jitter/Test_Jitter.txt', 'r')
#     file_data = file.readlines()[1:]
#
#     # Transform types.
#     wave = np.array([np.array(data.split('\t')) for data in file_data])
#     wave[:, -1] = np.array([data.replace('\n', '') for data in wave[:, -1]])
#     wave = wave.astype(float)
#
#     # The standard time axis.
#     tim = np.linspace(0.0, T * LEN, ACC * LEN, endpoint=False)
#     # Extract raw time.
#     tim_cache = wave[:, 0]
#
#     # Extract raw random jitters.
#     rand_jit_cache = wave[:, 2]
#     # Intepolate random jitters.
#     f = interpolate.splrep(tim_cache, rand_jit_cache, k=1)
#     rand_jit = interpolate.splev(tim, f)
#     # Mean and variance of random jitters.
#     rand_jit_mean = np.mean(rand_jit)
#     rand_jit_var = np.sqrt(np.var(rand_jit))
#     print('Mean of random jitter: %.3g*T' % rand_jit_mean)
#     print('Variance of random jitter: %.3g*T' % rand_jit_var)
#     # Normalize random jitters.
#     rand_jit = np.round(rand_jit * ACC).astype(int)
#     [rand_jit, times] = np.unique(rand_jit, return_counts=True)
#     rand_jit = rand_jit / ACC
#     # Plot.
#     plt.figure(figsize=(20, 15))
#     plt.plot(rand_jit, times)
#     plt.show()
#
#     # Extract raw period jitters.
#     peri_jit_cache = wave[:, 1]
#     # Intepolate period jitters.
#     f = interpolate.splrep(tim_cache, peri_jit_cache, k=1)
#     peri_jit = interpolate.splev(tim, f)
#     # Get distribution.
#     peri_jit = np.round(peri_jit*ACC).astype(int)
#     [peri_jit, times] = np.unique(peri_jit, return_counts=True)
#     peri_jit = peri_jit / ACC
#     # Plot.
#     plt.figure(figsize=(20, 15))
#     plt.plot(peri_jit, times)
#     plt.show()
#
#     # Extract overall jitters.
#     jit_cache = wave[:, 3]
#     # Intepolate jitters.
#     f = interpolate.splrep(tim_cache, jit_cache, k=1)
#     jit = interpolate.splev(tim, f)
#     # Find zero points.
#     neg_next = np.where(jit < 0.0)[0][: -1] + 1
#     loc = np.where(jit[neg_next] >= 0.0)[0]
#     zero_points = neg_next[loc]
#     zero_points = np.mod(zero_points, ACC)
#     # Counting.
#     [jit, times] = np.unique(zero_points, return_counts=True)
#     jit = jit / ACC
#     # Reversing.
#     index = 0
#     for n in range(len(jit)):
#         if jit[n] > 0.5:
#             index = n
#             break
#     jit[index:] -= 1.0
#     jit = np.array(list(jit[index:]) + list(jit[: index]))
#     times = np.array(list(times[index:]) + list(times[: index]))
#     # Plot.
#     plt.figure(figsize=(20, 15))
#     plt.plot(jit, times)
#     plt.show()
#
#
# if _EN_TEST_SAMPLE_RATE_:
#
#     # Sequence charactors.
#     T = 20e-9
#     ACC = 100
#     HIGH = 5.0
#
#     # The driver is a invertor.
#     # The signal input is a positive pulse.
#     # Vary the width of the pulse.
#     DIST_STEP = 0.01
#     DIST_START = 0.01
#     DIST_STOP = 10.0
#     section = np.array(range(int(DIST_STOP))) * ACC
#     num = np.round((DIST_STOP - DIST_START) / DIST_STEP).astype(int) + 1
#     dist = np.linspace(DIST_START, DIST_STOP, num)
#
#     # Extract data.
#     file = open('Test_Sample_Rate/Test_Sample_Rate.txt', 'r')
#     file_data = file.readlines()[1:]
#
#     # How many times to simulate.
#     head = []
#     for row_num in range(len(file_data)):
#         if file_data[row_num][0: 4] == 'Step':
#             head.append(row_num)
#
#     # The standard time axis.
#     tim = np.linspace(0.0, T, ACC, endpoint=False)
#
#     # Extract signal.
#     sig = []
#     for n in range(num):
#         if n == len(head) - 1:
#             wave = file_data[head[n] + 1:]
#         else:
#             wave = file_data[head[n] + 1: head[n + 1]]
#         wave = np.array([np.array(data.split('\t')) for data in wave])
#         wave[:, -1] = np.array([data.replace('\n', '') for data in wave[:, -1]])
#         wave = wave.astype(float)
#         tim_cache = wave[:, 0]
#         sig_cache = wave[:, 1]
#         f = interpolate.splrep(tim_cache, sig_cache, k=1)
#         sig.append(interpolate.splev(tim, f))
#     tim = np.array(tim)
#     sig = np.array(sig)
#
#     # Normalize the signal.
#     sig_mean = np.mean(sig, axis=0)
#     sig_mean = np.tile(sig_mean, num).reshape(num, -1)
#     sig_dif = sig - sig_mean
#
#     # Plot.
#     plt.figure(figsize=(20, 15))
#     sc = plt.imshow(sig_dif)
#     plt.colorbar(sc)
#     plt.show()
#
#     # Definee a threshold.
#     threshold = HIGH / ACC / 10
#
#     # Loop.
#     for n in range(len(section)):
#
#         # Get the index to calculate.
#         head = section[n]
#
#         # FFT for ACC times.
#         seq_x = fft.rfftfreq(num - head, T * DIST_STEP)
#         seq_y = np.abs(fft.rfft(sig_dif[head:, 0]))
#         for s in range(1, ACC):
#             seq_y_cache = np.abs(fft.rfft(sig_dif[head:, s]))
#             seq_y_cache = np.vstack((seq_y, seq_y_cache))
#             seq_y = np.max(seq_y_cache, axis=0)
#         seq_y /= num - head
#         seq_x_new = np.linspace(seq_x[0], seq_x[-1], len(seq_x) * ACC)
#         f = interpolate.splrep(seq_x, seq_y, k=3)
#         seq_y_new = interpolate.splev(seq_x_new, f)
#
#         # Calculate the ratio.
#         loc = np.argmin(np.abs(seq_y_new - threshold))
#         ratio = 1 / seq_x_new[loc] / T
#         print('No.' + str(n + 1) + ': %.4g' % ratio)
#
#         # Plot.
#         plt.figure()
#         plt.plot(seq_x_new, seq_y_new)
#         plt.scatter(seq_x_new[loc], seq_y_new[loc], marker='x')
#         plt.show()
#
#
# if _EN_EDGE_RESPONSE_2_:
#
#     # Sequence charactors.
#     T = 20e-9
#     ACC = 1000
#     HIGH = 5.0
#     PI = np.pi
#
#     # Parameters of jitters.
#     TPJ = (4*PI) * T
#     APJ = 0.10
#     ARJ = 0.05
#
#     # Linear section.
#     # Jitter distribution.
#     # Period jitter.
#     tim = np.linspace(0.0, 2*PI, ACC*ACC, endpoint=False)
#     sin_func = (APJ*ACC + 0.5 - 1/ACC) * np.sin(tim)
#     sin_func = np.round(sin_func).astype(int)
#     [peri_dist, peri_times] = np.unique(sin_func, return_counts=True)
#     # Random jitter.
#     t = np.linspace(-0.5, 0.5, ACC+1)
#     f = 1 / ARJ / np.sqrt(2*PI) * np.exp(- np.square(t) / 2 / ARJ / ARJ)
#     t = np.round(t * ACC).astype(int)
#     # Total jitter.
#     jit = np.convolve(peri_times, f)
#     threshold = np.max(jit) / ACC
#     index = 0
#     for n in range(len(jit)):
#         if threshold < jit[n]:
#             index = n
#             break
#     jit = jit[index: len(jit)-index]
#     tim = np.linspace(-(len(jit)//2), len(jit)//2, len(jit)) / ACC
#     jit = jit / np.max(jit)
#     # Plot.
#     plt.figure()
#     plt.plot(tim, jit)
#     plt.show()
#
