
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class EyeDiagramStruct:
    def __init__(self):
        # Eye Diagram.
        self.eye_diagram = np.array([]).astype(int)  # Canvas of the eye diagram.
        self.volt_per_dot = None  # Value of the voltage per dot.
        self.time_per_dot = None  # Value of the time per dot.
        self.zero_volt_location = None  # Orientation of zero voltage.
        self.eye_amplitude = None  # Amplitude of the eye.
        self.eye_height = None  # Height of the eye.
        self.eye_width = None  # Width of the eye.
        self.eye_jitter = None  # Jitter of the eye.

    def plot_eye_from_signal(self, signal, time, length, period, acc):

        """
        ______________________________________________________________________
        Function: Plot the eye diagram from signal.
        ______________________________________________________________________
        :param signal: The original list of signal data.
        :param time: The original list of time data.
        :param length: Length of the sequence.
        :param period: Period of the signal.
        :param acc: The accuracy of eye diagram.
        ______________________________________________________________________
        :return: The eye diagram.
        ______________________________________________________________________
        """

        # Margin.
        margin = acc // 5

        # Fix time steps.
        time_ori = time
        time_new = np.linspace(0.0, (length-1/acc)*period, length*acc)
        f = interpolate.splrep(time_ori, signal, k=3)
        signal = interpolate.splev(time_new, f)

        # Normalize.
        signal_h = max(signal)
        signal_l = min(signal)
        self.volt_per_dot = (signal_h - signal_l) / acc
        self.time_per_dot = period / acc
        self.zero_volt_location = margin + np.round(-signal_l/self.volt_per_dot).astype(int)
        sample = np.round((signal - signal_l) / self.volt_per_dot).astype(int) + margin

        # Get orientations.
        time = np.tile(np.array(range(acc)), length)
        [orientation, times] = np.unique(np.vstack((time, sample)), axis=1, return_counts=True)

        # Create canvas.
        self.eye_diagram = np.zeros([acc+2*margin, acc], dtype=int)
        self.eye_diagram[orientation[1, :], orientation[0, :]] = times

        # # Normalize.
        # eye_max = np.max(self.eye_diagram)
        # self.eye_diagram = np.round(self.eye_diagram / eye_max * 255).astype(int)

        # Plot the eye diagram.
        c_map = cm.get_cmap('plasma')
        c_map.set_bad('k')
        canvas = np.round(self.eye_diagram).astype(float)
        canvas[canvas == 0] = np.nan
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas, interpolation='none', cmap=c_map, origin='lower')
        # plt.axis('off')
        plt.show()

        # Output
        return self.eye_diagram

    # def eye_parameter(self):
    #
    #     """
    #     ______________________________________________________________________
    #     Function: Calculate parameters from the eye pattern.
    #     ______________________________________________________________________
    #     :return: Amplitude, height, width, and jitter of the eye pattern.
    #     ______________________________________________________________________
    #     """
    #
    #     # Accuracy.
    #     acc = self.accuracy
    #     # Set margin.
    #     margin = np.round(0.1*acc).astype(int)
    #
    #     # Calculate eye width and eye jitter.
    #     # Initialize parameters.
    #     section_t = [0.0, 0.0]
    #     jitter = [0.0, 0.0]
    #     # Find the two section.
    #     for n in range(2):
    #         # Get sample
    #         sample = self.eye_pattern[self.zero_volt_location+margin: self.zero_volt_location+acc-margin,
    #                                   n*acc: (n+1)*acc]
    #         weight = np.tile(np.array(range(acc)), acc-2*margin)
    #         weight = weight.reshape(-1, acc)
    #         centre = np.sum(np.multiply(sample, weight), axis=1) / np.sum(sample, axis=1)
    #         dist2 = np.power(weight - np.repeat(centre, acc).reshape(-1, acc), 2)
    #         var = np.power(np.sum(np.multiply(sample, dist2), axis=1) / np.sum(sample, axis=1), 0.5)
    #         section_t[n] = centre[np.nanargmin(var)]
    #         jitter[n] = 3 * var[np.nanargmin(var)]
    #     # Calculate.
    #     self.eye_jitter = sum(jitter) / 2 * self.time_per_dot
    #     self.eye_width = self.period - self.eye_jitter
    #
    #     # Calculate eye amplitude.
    #     # Get centre of the pattern.
    #     centre_t = np.round((section_t[0]+section_t[1]+acc)/2).astype(int)
    #     mid_height = np.shape(self.eye_pattern)[0] // 2
    #     # Get sample
    #     sample_bt = np.sum(self.eye_pattern[:mid_height, centre_t-margin: centre_t+margin], axis=1)
    #     sample_tp = np.sum(self.eye_pattern[mid_height:, centre_t-margin: centre_t+margin], axis=1)
    #     weight_bt = np.array(range(len(sample_bt)))
    #     weight_tp = np.array(range(len(sample_tp))) + mid_height
    #     centre_bt = np.sum(np.multiply(sample_bt, weight_bt)) / np.sum(sample_bt)
    #     centre_tp = np.sum(np.multiply(sample_tp, weight_tp)) / np.sum(sample_tp)
    #     # Calculate.
    #     self.eye_amplitude = (centre_tp-centre_bt) * self.volt_per_dot
    #
    #     # Calculate eye height.
    #     dist2_bt = np.power(weight_bt-centre_bt, 2)
    #     dist2_tp = np.power(weight_tp-centre_tp, 2)
    #     var_bt = np.power(np.sum(np.multiply(sample_bt, dist2_bt)) / np.sum(sample_bt), 0.5)
    #     var_tp = np.power(np.sum(np.multiply(sample_tp, dist2_tp)) / np.sum(sample_tp), 0.5)
    #     self.eye_height = self.eye_amplitude - 3*(var_bt+var_tp)/2 * self.volt_per_dot
    #
    #     # Output.
    #     return [self.eye_amplitude, self.eye_height, self.eye_width, self.eye_jitter]
    #
