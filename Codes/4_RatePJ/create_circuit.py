
def create_circuit(circuit, amp, dio_offset):

    # Add models.
    # Anolog models.
    circuit.models.add_model('.model NMOS VDMOS(Rg=3 Rd=15m Rs=9m Vto= 1 Kp=100 Cgdmax=1n Cgdmin=.44n Cgs=2n Cjo=.1n Is=40p Rb=15m ksubthres=.1 mfg=Siliconix Vds= 20 Ron=30m Qg=15n)')
    circuit.models.add_model('.model PMOS VDMOS(pchan Rg=3 Rd=15m Rs=9m Vto=-1 Kp=100 Cgdmax=1n Cgdmin=.44n Cgs=2n Cjo=.1n Is=40p Rb=15m ksubthres=.1 mfg=Siliconix Vds=-20 Ron=30m Qg=15n)')
    circuit.models.add_model('.model DIODE D(Is=2.52n Rs=.568 N=1.752 Cjo=4p M=.4 tt=20n Iave=200m Vpk=75 mfg=OnSemi)')
    # Bridges of analog and digital.
    circuit.models.add_model('.model ADC_BUF adc_bridge(in_low={low} in_high={high} rise_delay=1.0e-12 fall_delay=1.0e-12)'
                             .replace('{low}', str(amp/2-dio_offset)).replace('{high}', str(amp/2+dio_offset)))
    circuit.models.add_model('.model DAC_BUF dac_bridge(out_low=0.0 out_high={high} t_rise=0.0 t_fall=0.0)'
                             .replace('{high}', str(amp)))

    # Set parameters.
    # Set parameters of sender.
    TYPE = 'PP'
    MODEL = ['NMOS', 'PMOS', 'DIODE']
    COMP = [3e-15]
    PKG = [0.001, 0.1e-12, 2e-15]
    VOLT = [amp, amp-dio_offset, dio_offset]
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
    VOLT = [amp, amp-dio_offset, dio_offset]
    PARAM = [MODEL, COMP, LOAD, VOLT]
    circuit.parameters.rx_param = PARAM

    return circuit
