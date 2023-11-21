
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from scipy import interpolate
import time

import process
import simulate


# Enable.
_EN_EFFECT_ = True

# Invoke the class.
circuit = simulate.CircuitStruct()

# Signal.
AMP = 5.0
DIO_OFFSET = 0.0
T = 20e-9
PI = np.pi
DELAY = 2.24e-9

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
COMP = [3e-15]
PKG = [0.001, 0.1e-12, 2e-15]
VOLT = [AMP, AMP-DIO_OFFSET, DIO_OFFSET]
PARAM = [TYPE, MODEL, COMP, PKG, VOLT]
circuit.parameters.tx_param = PARAM
# Set parameters of channel.
LENGTH = 1
RLGC = [0.01, 100e-9, 1e-6, 50e-12]
ACC = 10
PARAM = [LENGTH, RLGC, ACC]
circuit.parameters.ch_param = PARAM
# Set parameters of receiver.
MODEL = ['DIODE']
COMP = [1e-15]
LOAD = [200]
VOLT = [AMP, AMP-DIO_OFFSET, DIO_OFFSET]
PARAM = [MODEL, COMP, LOAD, VOLT]
circuit.parameters.rx_param = PARAM

# Generate circuit.
circuit.circuit.gene_circuit(circuit.parameters)

if _EN_EFFECT_:

    dif = []

    for M in range(1, 6):

        acc = 100
        sig = []
        tim = np.linspace((M+1)*T+DELAY, (M+10)*T+DELAY, 9*acc, endpoint=False)

        data = [0] * M + [1]
        jit = [0.0] * (M+1)
        jit_x = np.copy(jit)
        jit_range = np.linspace(-0.3, 0.3, 61)

        circuit.signal.reset_signal()

        for jit_0 in jit_range:
            jit_x[0] = jit_0
            circuit.signal.gene_sig(1, data, jit_x, AMP, T)

        # Simulate.
        OPT = '.OPTIONS GMIN=1e-11 ABSTOL=1e-12 RELTOL=0.005 CHGTOL=1e-14 TRTOL=1 VOLTTOL=1e-6 METHOD=GEAR'
        TEMP = 25
        STEP = T
        circuit.spice_simulation.reset_circuit()
        circuit.spice_simulation.sim_setting(circuit.circuit,
                                             circuit.signal,
                                             circuit.models,
                                             OPT, TEMP, STEP, (M+10)*T+DELAY*2)

        # Extract data.
        tim_sim = circuit.spice_simulation.result['time']
        sig_sim = circuit.spice_simulation.result['v_out']

        for sig_m in sig_sim:
            f = interpolate.splrep(tim_sim, sig_m, k=1)
            sig.append(interpolate.splev(tim, f))

        sig_m = np.array(sig)
        sig_max = np.max(sig_m, axis=0)
        sig_min = np.min(sig_m, axis=0)
        sig_dif = sig_max - sig_min

        plt.figure(figsize=(10, 10))
        sig_ref = sig[30]
        for sig_m in sig:
            plt.plot(tim, sig_m-sig_ref)
        plt.show()

        # plt.figure()
        # plt.plot(tim, sig_dif)
        # plt.show()

        dif.append(max(sig_dif))
        print('M = ' + str(M))

    threshold = AMP/100
    effect_x = [1, 2, 3, 4, 5]
    line_fit = np.polyfit(effect_x, np.log10(dif), 1)
    line_x = np.array([effect_x[0], effect_x[-1]])
    line_y = np.array([line_x[0]*line_fit[0]+line_fit[1],
                       line_x[1]*line_fit[0]+line_fit[1]])
    line_y = np.power(10, line_y)

    plt.figure()
    plt.scatter(effect_x, dif, marker='x')
    plt.plot(line_x, line_y)
    plt.plot(line_x, [threshold, threshold])
    plt.yscale('log')
    plt.grid(True)
    plt.show()

