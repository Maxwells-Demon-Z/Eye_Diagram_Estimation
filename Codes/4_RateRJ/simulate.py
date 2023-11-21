
import numpy as np
from PySpice.Spice.Netlist import Circuit
import PySpice.Logging.Logging as Logging
import matplotlib.pyplot as plt

logger = Logging.setup_logging()


class CircuitStruct:
    def __init__(self):
        self.models = self.Models()
        self.signal = self.Signal()
        self.parameters = self.Parameters()
        self.circuit = self.Circuit()
        self.spice_simulation = self.SpiceSimulation()

    class Models:
        def __init__(self):
            self.model_list = []

        def add_model(self, model):
            self.model_list.append(model)

    class Signal:
        def __init__(self):
            self.b_sig = []

        def gene_sig(self, init, seq, jit, amp, period):

            current_state = init
            edge_r = []
            edge_f = []
            b_sig = '0.0' if (init == 0) else '{high}'.replace('{high}', str(amp))

            for n in range(len(seq)):
                if (seq[n] == 1) & (current_state == 0):
                    edge_r.append(n)
                    current_state = 1
                elif (seq[n] == 0) & (current_state == 1):
                    edge_f.append(n)
                    current_state = 0

            for n in range(len(edge_r)):
                b_sig += ' + (time>({tn})?{amp}:0.0)'
                b_sig = b_sig.replace('{tn}', str((edge_r[n]+1+jit[edge_r[n]])*period))

            for n in range(len(edge_f)):
                b_sig += ' - (time>({tn})?{amp}:0.0)'
                b_sig = b_sig.replace('{tn}', str((edge_f[n]+1+jit[edge_f[n]])*period))

            b_sig = b_sig.replace('{amp}', str(amp))
            b_sig = '\nB' + str(len(self.b_sig)) + ' I' + str(len(self.b_sig)) + ' 0 V=' + b_sig

            self.b_sig.append(b_sig)

        def reset_signal(self):
            self.b_sig = []

    class Parameters:
        def __init__(self):
            self.tx_param = None
            self.ch_param = None
            self.rx_param = None

    class Circuit:

        def __init__(self):
            self.circuit = ''

        def gene_circuit(self, parameters):

            # Add title.
            self.circuit += '\n* CIRCUIT WITHOUT SOURCE MODEL\n.SUBCKT ORI IN OUT SGND\n'

            # Tx terminal.
            # Parameters of sender.
            param = parameters.tx_param
            model = param[1]
            comp = param[2]
            pkg = param[3]
            volt = param[4]
            nmos = model[0]
            pmos = model[1]
            diode = model[2]
            C_comp = comp[0]
            R_pkg = pkg[0]
            L_pkg = pkg[1]
            C_pkg = pkg[2]
            Vcc = volt[0]
            Vmax = volt[1]
            Vmin = volt[2]
            # Title.
            TX = ''
            # Netlist
            TX += '\nM1 2 IN SGND SGND ' + nmos
            TX += '\nM2 2 IN VCCT VCCT ' + pmos
            # TX += '\nM3 3 2 SGND SGND ' + nmos
            # TX += '\nM4 3 2 VCCT VCCT ' + pmos
            TX += '\nVCCT VCCT SGND ' + str(Vcc)
            TX += '\nVMAXT VMAXT SGND ' + str(Vmax)
            TX += '\nVMINT VMINT SGND ' + str(Vmin)
            TX += '\nD1T 2 VMAXT ' + diode
            TX += '\nD2T VMINT 2 ' + diode
            TX += '\nC10T 2 SGND ' + str(C_comp)
            TX += '\nC11T VCCT 2 ' + str(C_comp)
            TX += '\nR1T 4 2 ' + str(R_pkg)
            TX += '\nL1T NTX 4 ' + str(L_pkg)
            TX += '\nC1T NTX SGND ' + str(C_pkg)
            # Add TX into raw spice.
            self.circuit += TX + '\n'

            # Channel.
            # Parameters of Channel.
            param = parameters.ch_param
            length = param[0]
            RLGC = param[1]
            acc = param[2]
            R = length / acc * RLGC[0]
            L = length / acc * RLGC[1]
            G = length / acc * RLGC[2]
            C = length / acc * RLGC[3]
            # Title.
            CH = ''
            # Netlist.
            for i in range(acc):
                CH += '\nR' + str(i) + '0 N' + str(i) + '0 N' + str(i) + '1 ' + str(R)
                CH += '\nL' + str(i) + ' N' + str(i) + '1 N' + str(i + 1) + '0 ' + str(L)
                CH += '\nR' + str(i) + '1 N' + str(i + 1) + '0 SGND ' + str(1 / G)
                CH += '\nC' + str(i) + ' N' + str(i + 1) + '0 SGND ' + str(C)
            CH = CH.replace('N00', 'NTX')
            CH = CH.replace('N' + str(acc) + '0', 'OUT')
            # Add CH into raw spice.
            self.circuit += CH + '\n'

            # Rx terminal.
            # Parameters of Receiver.
            param = parameters.rx_param
            model = param[0]
            comp = param[1]
            load = param[2]
            volt = param[3]
            diode = model[0]
            C_comp = comp[0]
            R_load = load[0]
            Vcc = volt[0]
            Vmax = volt[1]
            Vmin = volt[2]
            # Title.
            RX = ''
            # Netlist.
            RX += '\nRL OUT SGND ' + str(R_load)
            RX += '\nVCCR VCCR SGND ' + str(Vcc)
            RX += '\nVMAXR VMAXR SGND ' + str(Vmax)
            RX += '\nVMINR VMINR SGND ' + str(Vmin)
            RX += '\nC10R OUT SGND ' + str(C_comp)
            RX += '\nC11R VCCR OUT ' + str(C_comp)
            RX += '\nD1R OUT VMAXR ' + diode
            RX += '\nD2R VMINR OUT ' + diode
            # Add RX into raw spice.
            self.circuit += RX + '\n'

            # End subcircuit.
            self.circuit += '\n.ENDS ORI\n'

    class SpiceSimulation:
        def __init__(self):
            self.circuit = Circuit('Sim')
            self.result = None

        def sim_setting(self, circuit, signal, models, opt, temp, step, end):

            # Add device models.
            # Number of device models.
            num = len(models.model_list)
            model = ''
            # Convert device models to strings.
            for n in range(num):
                model += '\n' + models.model_list[n]
            # Add device models into raw spice.
            self.circuit.raw_spice += model

            # Add subcircuit model into raw spice.
            self.circuit.raw_spice += '\n' + circuit.circuit

            # Number of signals.
            num = len(signal.b_sig)
            cir = ''
            # Multiply subcircuits.
            for n in range(num):
                cir += '\n' + 'X' + str(n) + ' I' + str(n) + ' O' + str(n) + ' 0 ORI'
                cir += signal.b_sig[n]
                # cir += '\n' + 'B' + str(n) + ' I' + str(n) + ' 0 V=' + signal.b_sig[n]
            # Add subcircuits into raw spice.
            self.circuit.raw_spice += cir

            # Simulate.
            self.circuit.raw_spice += '\n' + opt + '\n'
            simulator = self.circuit.simulator(temperature=temp, nominal_temperator=temp)
            res = simulator.transient(step_time=step, end_time=end)

            # Extract results.
            tim = np.array(res.time)
            # ran = []
            v_in = []
            v_out = []
            for n in range(num):
                # ran.append(np.array(res.nodes['vs4']))
                v_in.append(np.array(res.nodes['i' + str(n)]))
                v_out.append(np.array(res.nodes['o' + str(n)]))

            self.result = {'time': tim,
                           # 'ran': ran,
                           'v_in': v_in,
                           'v_out': v_out
                           }

        def reset_circuit(self):
            self.circuit = Circuit('Sim')
            self.result = None

        def plot_node_voltage(self):

            for n in range(len(self.result['v_in'])):
                plt.figure(figsize=(20, 15))
                # plt.plot(self.result['time'][: 1000], self.result['ran'][n][: 1000])
                # plt.plot(self.result['time'], self.result['v_in'][n])
                plt.plot(self.result['time'], self.result['v_out'][n])
                plt.show()

