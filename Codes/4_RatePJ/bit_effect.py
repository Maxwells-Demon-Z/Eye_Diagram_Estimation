
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


def bit_effect()

    WIDTH = 1
    acc = 100
    sig = []
    tim = np.linspace(WIDTH*T+DELAY, (WIDTH+10)*T+DELAY, 10*acc, endpoint=False)

    for n in range(200):

        data = np.random.randint(2**WIDTH, 2**(WIDTH+1))
        data = list(bin(data)[3:])
        data = list(np.array(data).astype(int))
        jit = [0.0] * (len(data))

        circuit.signal.reset_signal()
        circuit.signal.gene_sig(0, data, jit, AMP, T)
        circuit.signal.gene_sig(1, data, jit, AMP, T)

        # Simulate.
        OPT = '.OPTIONS GMIN=1e-11 ABSTOL=1e-12 RELTOL=0.005 CHGTOL=1e-14 TRTOL=1 VOLTTOL=1e-6 METHOD=GEAR'
        TEMP = 25
        STEP = T
        circuit.spice_simulation.reset_circuit()
        circuit.spice_simulation.sim_setting(circuit.circuit,
                                             circuit.signal,
                                             circuit.models,
                                             OPT, TEMP, STEP, (WIDTH+10)*T+DELAY*2)

        # Extract data.
        tim_sim = circuit.spice_simulation.result['time']
        sig_sim = circuit.spice_simulation.result['v_out'][0] - \
                  circuit.spice_simulation.result['v_out'][1]

        # Unify.
        f = interpolate.splrep(tim_sim, sig_sim, k=1)
        sig.append(interpolate.splev(tim, f))

        print(str(n))

    sig_m = np.abs(np.array(sig))
    sig_m = np.max(sig_m, axis=0)

    # plt.figure(figsize=(10, 10))
    # for n in range(500):
    #     plt.plot(tim, sig[n], alpha=0.5)
    # plt.show()

    print(max(sig_m))

    plt.figure(figsize=(10, 10))
    plt.plot(tim, sig_m)
    plt.plot([tim[0], tim[-1]], [AMP/acc, AMP/acc])
    plt.show()


threshold = AMP/100
effect_x = [2, 3, 4, 5,
            6, 7, 8, 9]
effect_y = [2.76, 1.46, 0.633, 0.279,
            0.177, 0.0452, 0.0192, 0.00768]
line_fit = np.polyfit(effect_x, np.log10(effect_y), 1)
line_x = np.array([effect_x[0], effect_x[-1]])
line_y = np.array([line_x[0]*line_fit[0]+line_fit[1],
                   line_x[1]*line_fit[0]+line_fit[1]])
line_y = np.power(10, line_y)

plt.figure()
plt.scatter(effect_x, effect_y, marker='x')
plt.plot(line_x, line_y)
plt.plot(line_x, [threshold, threshold])
plt.yscale('log')
plt.grid(True)
plt.show()

