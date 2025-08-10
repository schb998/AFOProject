import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from yatpkg.util.data import TRC as YatTRC

class CustomTRC:
    def __init__(self, data, marker_set, column_labels, dt):
        self.data = data
        self.marker_set = marker_set
        self.column_labels = column_labels
        self.dt = dt

    def update(self):
        """
        Updates the TRC marker data based on the current dataset (for each gait cycle).
        """
        if len(self.marker_set.keys()) > 0:
            backup = deepcopy(self.data)
            marker_temp = {}
            for m in self.marker_set:
                for j in range(len(self.column_labels)):
                    label = self.column_labels[j]
                    if m in label and "x" in label.lower():
                        st = j
                        en = j + 3
                        mx = deepcopy(self.data[:, st:en])
                        marker_temp[m] = pd.DataFrame(data=mx, columns=self.marker_set[m].columns)
                        break

            self.marker_set = marker_temp
            self.data = backup
        else:
            markers = {
                self.column_labels[c].split('_')[0]: pd.DataFrame(
                    data=np.zeros([self.data.shape[0], 3]),
                    columns=["X", "Y", "Z"]
                )
                for c in range(1, len(self.column_labels))
                if 'time' not in self.column_labels[c]
            }
            marker_names = list(markers.keys())
            self.data[:, 0] = self.data[:, 0] - self.data[0, 0]
            for m in range(len(marker_names)):
                markers[marker_names[m]].columns = [f'X{m+1}', f'Y{m+1}', f'Z{m+1}']

            self.marker_set = markers
            temp = np.zeros([self.data.shape[0], self.data.shape[1]+1])
            temp[:, 0] = np.arange(1, self.data.shape[0]+1)
            temp[:, 1:] = self.data
            self.data = temp
            self.column_labels.insert(0, "frame")

    def write(self, filename):
        """
        Writes the TRC data to a file.
        """
        lines = []
        c0 = "Frame#\tTime\t"
        c1 = "\t\t"
        for c in self.marker_set.keys():
            c0 += f"{c}\t\t\t"
            c1 += "X\tY\tZ\t"
        lines.append(c0.strip() + "\t\t\n")
        lines.append("\t\t" + c1.strip() + "\n")

        for si in range(self.data.shape[0]):
            line = f"{int(self.data[si, 0])}\t{self.data[si, 1]}"
            for m in self.marker_set.keys():
                marker = self.marker_set[m]
                x = marker.iloc[si, 0]
                y = marker.iloc[si, 1]
                z = marker.iloc[si, 2]

                xo = str(x) if not np.isnan(x) else ''
                yo = str(y) if not np.isnan(y) else ''
                zo = str(z) if not np.isnan(z) else ''
                line += f"\t{xo}\t{yo}\t{zo}"
            line += '\n'
            lines.append(line)

        with open(filename, 'w') as writer:
            writer.writelines(lines)

def read_mot_files(data_path_mot):
    motion_data_list = []
    file_list = sorted(f for f in os.listdir(data_path_mot) if f.endswith('.mot'))

    for filename in file_list:
        file_path = os.path.join(data_path_mot, filename)
        print(f"Reading MOT file: {file_path}")
        try:
            with open(file_path, 'r') as file:
                for _ in range(6):  # Skip the first 6 header rows
                    next(file)
                data = pd.read_csv(file, sep=r'\s+')
            motion_data_list.append({
                'fileName': filename,
                'rawData': data.to_numpy(),
                'colNames': data.columns.to_list()
            })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return motion_data_list

def read_trc_files(data_path_trc):
    if not os.path.isdir(data_path_trc):
        print(f"The provided path '{data_path_trc}' is not a directory.")
        return []

    trc_data_list = []
    for filename in os.listdir(data_path_trc):
        if filename.endswith('.trc'):
            file_path = os.path.join(data_path_trc, filename)
            print(f"Reading TRC file: {file_path}")
            try:
                trc_data = YatTRC.read(file_path)  # Use the aliased YatTRC
                trc_data_list.append(trc_data)
                print(f"Successfully read {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return trc_data_list

def detect_heel_strikes(vertical_force, threshold=20):
    """
    Detect heel strikes from vertical grf.

    Returns:
        list: Indices of detected heel strikes.
    """
    heel_strikes = []
    for i in range(1, len(vertical_force)):
        if vertical_force[i] > threshold and vertical_force[i - 1] <= threshold:
            heel_strikes.append(i)
    return heel_strikes

def get_samples(trc_data, time_points):
    """
    Retrieve the accompanying data in TRC file based on given time points.

    Parameters:
        trc_data (YatTRC): TRC data object.
        time_points (list): List of time points to extract data.

    Returns:
        pd.DataFrame: Extracted marker data corresponding to the time points.
    """
    samples = trc_data.get_samples(time_points, as_pandas=True)
    return samples


def visualize_heel_strikes(mot_data, heel_strikes_left, heel_strikes_right):

    time = mot_data['time']
    ground_force1_vy = mot_data['ground_force1_vy']
    ground_force2_vy = mot_data['ground_force2_vy']

    plt.figure(figsize=(12, 6))

    # Plot ground reaction forces
    plt.plot(time, ground_force1_vy, label="Left vy (ground_force1_vy)", alpha=0.8)
    plt.plot(time, ground_force2_vy, label="Right vy (ground_force2_vy)", alpha=0.8)

    # Mark heel strikes
    plt.scatter(time[heel_strikes_left], ground_force1_vy[heel_strikes_left], color='red', label="Left Heel Strikes",
                zorder=5)
    plt.scatter(time[heel_strikes_right], ground_force2_vy[heel_strikes_right], color='blue',
                label="Right Heel Strikes", zorder=5)

    plt.xlabel("Time (s)")
    plt.ylabel("Vertical Force (N)")
    plt.title("Heel Strike Detection")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    data_path_trc = r"C:\Users\schb998\MyData\Pilot_Ella\Incline_Trials\walking_incline trial NOAFO\org_trc"
    data_path_mot = r"C:\Users\schb998\MyData\Pilot_Ella\Incline_Trials\walking_incline trial NOAFO\mot.corrected"

    # Load TRC and MOT files
    trc_data_list = read_trc_files(data_path_trc)
    motion_data_list = read_mot_files(data_path_mot)

    if motion_data_list and trc_data_list:
        trc_data = trc_data_list[0]
        mot_data = pd.DataFrame(
            motion_data_list[0]['rawData'],
            columns=motion_data_list[0]['colNames']
        )

        # Detect left and right heel strikes
        heel_strikes_left = detect_heel_strikes(mot_data['ground_force1_vy'], threshold=20)
        heel_strikes_right = detect_heel_strikes(mot_data['ground_force2_vy'], threshold=20)
        visualize_heel_strikes(mot_data, heel_strikes_left, heel_strikes_right)
        # trc_org = YatTRC.read(r"C:\Users\schb998\MyData\Pilot_Ella\Day2\Day2\walking_speed.trc")
        # Process left gait cycles
        for i in range(1, len(heel_strikes_left)):
            period = mot_data['time'][heel_strikes_left[i]] - mot_data['time'][heel_strikes_left[i - 1]]
            num_frames = int(period / (1 / 1000.0))
            timepoints = [t * (1 / 1000.0) + mot_data['time'][heel_strikes_left[i - 1]] for t in range(num_frames)]
            trc_subset = trc_data.get_samples_as_trc(timepoints)
            trc_subset.write(rf"C:\Users\schb998\MyData\Pilot_Ella\Incline_Trials\walking_incline trial NOAFO\segmented_trc\Left\left_cycle_{i}.trc")

        # Process right gait cycles
        for i in range(1, len(heel_strikes_right)):
            period = mot_data['time'][heel_strikes_right[i]] - mot_data['time'][heel_strikes_right[i - 1]]
            num_frames = int(period / (1 / 1000.0))
            timepoints = [t * (1 / 1000.0) + mot_data['time'][heel_strikes_right[i - 1]] for t in range(num_frames)]
            trc_subset = trc_data.get_samples_as_trc(timepoints)
            trc_subset.write(rf"C:\Users\schb998\MyData\Pilot_Ella\Incline_Trials\walking_incline trial NOAFO\segmented_trc\Right\right_cycle_{i}.trc")

        print("TRC files for left and right gait cycles written successfully.")
