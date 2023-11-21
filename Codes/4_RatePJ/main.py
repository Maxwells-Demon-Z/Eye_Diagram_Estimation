
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import interpolate
from scipy import fft
import time

import process
import simulate
import create_circuit


# Parameters of signal.
AMP = 5.0
DIO_OFFSET = 0.0
DELAY = 2.24e-9
T = 20e-9
T_PJ = np.pi * np.pi
AMP_PJ = 0.1
SIGMA_RJ = 0.06

# Invoke the class.
circuit = simulate.CircuitStruct()
# Generate circuit.
circuit = create_circuit.create_circuit(circuit, AMP, DIO_OFFSET)
circuit.circuit.gene_circuit(circuit.parameters)

# Enable.
_EN_BE_ = True
_EN_JE_ = False
_EN_RATE_PJ_ = False
_EN_RATE_RJ_ = False
_EN_GEN_ = False

dif = []

acc = 100
sig = []
tim = np.linspace(5*T+DELAY, 6*T+DELAY, acc, endpoint=False)

data = [0, 1, 0, 1]
T0_list = np.linspace(0.0, TPJ, acc, endpoint=False)

for T0 in T0_list:

    jit = []
    for n in range(6):
        jit.append(APJ*np.sin(2*np.pi/TPJ*(n*T+T0)))

    circuit.signal.reset_signal()
    circuit.signal.gene_sig(1, data, jit, AMP, T)

    # Simulate.
    OPT = '.OPTIONS GMIN=2e-12 ABSTOL=1e-12 RELTOL=0.002 CHGTOL=1e-14 TRTOL=1 VOLTTOL=1e-6 METHOD=GEAR'
    TEMP = 25
    STEP = T
    circuit.spice_simulation.reset_circuit()
    circuit.spice_simulation.sim_setting(circuit.circuit,
                                         circuit.signal,
                                         circuit.models,
                                         OPT, TEMP, STEP, 7*T+DELAY*2)

    # Extract data.
    tim_sim = circuit.spice_simulation.result['time']
    sig_sim = circuit.spice_simulation.result['v_out'][0]

    f = interpolate.splrep(tim_sim, sig_sim, k=1)
    sig.append(interpolate.splev(tim, f))

sig = np.array(sig)

# SECTION = 20
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = tim
y = T0_list
x, y = np.meshgrid(x, y)
surf = ax.plot_surface(x, y, sig, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
# ax.plot(xs=[tim[SECTION]]*acc, ys=T0_list, zs=sig[:, SECTION], c='black')
plt.show()

seq_x = fft.rfftfreq(acc, TPJ/acc)
freq = 0.0
for n in range(1, acc):
    seq_y = np.abs(fft.rfft(sig[:, n]))
    area = [0.0]
    for seq_y_m in seq_y:
        area.append(area[-1]+seq_y_m)
    area = np.array(area[1:])
    loc = np.argmin(np.abs(area-0.95*area[-1]))
    freq = freq if (seq_x[loc] < freq) else seq_x[loc]
print(str(freq))

index = np.linspace(0, 99, 100)
index_new = np.linspace(0, 99, 20).astype(int)
sig_new = []
for n in index_new:
    sig_new.append(sig[n, :])
sig_new = np.array(sig_new)
sig_fit = []
for n in range(acc):
    f = interpolate.splrep(index_new, sig_new[:, n], k=3)
    sig_fit.append(interpolate.splev(index, f))
sig_fit = np.array(sig_fit).T

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x, y, sig_fit, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
plt.show()

# sig_fit = np.copy(sig)
# rand = np.random.normal(loc=0.0, scale=AMP/acc/20, size=np.shape(sig))
# sig_fit = sig_fit + rand

# plt.figure(figsize=(15, 10))
# for n in range(acc):
#     plt.plot(tim, sig[n, :], 'b-', alpha=0.5)
#     plt.plot(tim, sig_fit[n, :], 'r--', alpha=0.5)
# plt.ylim(2, 6)
# plt.show()

