
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


# Invoke the class.
circuit = simulate.CircuitStruct()

# Signal.
AMP = 5.0
DIO_OFFSET = 0.3
T = 12e-9
PI = np.pi
DELAY = 2.20e-9
TPJ = PI*PI*T
APJ = 0.1

# Add models.
# Anolog models.
circuit.models.add_model('.model NMOS VDMOS(Rg=3 Rd=15m Rs=9m Vto= 1 Kp=100 Cgdmax=1n Cgdmin=.44n Cgs=2n Cjo=.1n Is=40p Rb=15m ksubthres=.1 mfg=Siliconix Vds= 20 Ron=30m Qg=15n)')
circuit.models.add_model('.model PMOS VDMOS(pchan Rg=3 Rd=15m Rs=9m Vto=-1 Kp=100 Cgdmax=1n Cgdmin=.44n Cgs=2n Cjo=.1n Is=40p Rb=15m ksubthres=.1 mfg=Siliconix Vds=-20 Ron=30m Qg=15n)')
circuit.models.add_model('.model DIODE D(Is=2.52n Rs=.568 N=1.752 Cjo=4p M=.4 tt=20n Iave=200m Vpk=75 mfg=OnSemi)')
# Bridges of analog and digital.
circuit.models.add_model('.model ADC_BUF adc_bridge(in_low={low} in_high={high} rise_delay=1.0e-12 fall_delay=1.0e-12)'
                         .replace('{low}', str(AMP/2-DIO_OFFSET)).replace('{high}', str(AMP/2+DIO_OFFSET)))
circuit.models.add_model('.model DAC_BUF dac_bridge(out_low=0.0 out_high={high} t_rise=0.0 t_fall=0.0)'
                         .replace('{high}', str(AMP)))

# Set parameters.
# Set parameters of sender.
TYPE = 'PP'
MODEL = ['NMOS', 'PMOS', 'DIODE']
COMP = [30e-15]
PKG = [0.01, 1e-12, 20e-15]
VOLT = [AMP, AMP-DIO_OFFSET, DIO_OFFSET]
PARAM = [TYPE, MODEL, COMP, PKG, VOLT]
circuit.parameters.tx_param = PARAM
# Set parameters of channel.
LENGTH = 1
RLGC = [0.01, 100e-9, 1e-12, 50e-12]
ACC = 10
PARAM = [LENGTH, RLGC, ACC]
circuit.parameters.ch_param = PARAM
# Set parameters of receiver.
MODEL = ['DIODE']
COMP = [0.0]
LOAD = [100]
VOLT = [AMP, AMP-DIO_OFFSET, DIO_OFFSET]
PARAM = [MODEL, COMP, LOAD, VOLT]
circuit.parameters.rx_param = PARAM

# Generate circuit.
circuit.circuit.gene_circuit(circuit.parameters)

for k in range(6):

    dif = []

    acc = 100
    sig = []
    tim = np.linspace(5*T+DELAY, 6*T+DELAY, acc, endpoint=False)

    data = [1, 0, 1, 0, 1, 0]
    sigma = 0.02
    delta_list = np.linspace(-5*sigma, 5*sigma, acc)

    for delta in delta_list:

        jit = [0.0] * 6
        jit[k] = delta

        circuit.signal.reset_signal()
        circuit.signal.gene_sig(0, data, jit, AMP, T)

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

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = tim
    y = delta_list
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, sig, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    plt.show()

    # seq_x = fft.rfftfreq(acc, 10*sigma*T/acc)
    # seq_y = np.abs(fft.rfft(sig[:, 0]))
    # for n in range(1, acc):
    #     seq_y_cache = np.abs(fft.rfft(sig[:, n]))
    #     seq_y = np.vstack((seq_y, seq_y_cache))
    # seq_y = np.max(seq_y, axis=0)

    # area = [0.0]
    # for seq_y_m in seq_y:
    #     area.append(area[-1]+seq_y_m)
    # area = np.array(area[1:])
    # loc = np.argmin(np.abs(area-0.95*area[-1]))
    #
    # print(seq_x[loc]/10e9)

    index = np.linspace(0, 99, 100)
    index_new = np.linspace(0, 99, 10).astype(int)
    sig_new = []
    for n in index_new:
        sig_new.append(sig[n, :])
    sig_new = np.array(sig_new)
    sig_fit = []
    for n in range(acc):
        f = interpolate.splrep(index_new, sig_new[:, n], k=3)
        sig_fit.append(interpolate.splev(index, f))
    sig_fit = np.array(sig_fit).T

    # plt.figure(figsize=(15, 10))
    # for n in range(acc):
    #     plt.plot(tim, sig[n, :], 'b-', alpha=0.5)
    #     plt.plot(tim, sig_fit[n, :], 'r--', alpha=0.5)
    # # plt.ylim(0, 6)
    # plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, y, sig_fit, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    plt.show()

