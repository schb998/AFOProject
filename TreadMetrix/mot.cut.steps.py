import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CustomMOT:
    def __init__(self, data, column_labels, metadata):
        """
        Initialize a CustomMOT object.

        Parameters:
            data (pd.DataFrame): The motion data for a gait cycle.
            column_labels (list): Column headers from the original MOT file.
            metadata (dict): Metadata extracted from the MOT file.
        """
        self.data = data
        self.column_labels = column_labels
        self.metadata = metadata

    def write(self, filename):
        """
        Writes the MOT data to a file.

        Parameters:
            filename (str): The path to the output MOT file.
        """
        with open(filename, 'w') as file:
            # Write the metadata
            file.write("walking_incline\n")  # Change
            for key, value in self.metadata.items():
                file.write(f"{key} = {value}\n")
            file.write("\n")  # Blank line separating metadata and data

            # Write column headers
            file.write("\t".join(self.column_labels) + "\n")

            # Write the data
            self.data.to_csv(file, sep='\t', index=False, header=False)


def write_mot_files(samples, column_labels, metadata, output_dir):
    """
    Writes MOT files for each gait cycle.

    Parameters:
        samples (dict): Dictionary of samples for each gait cycle.
        column_labels (list): Column headers from the original MOT file.
        metadata (dict): Metadata extracted from the MOT file.
        output_dir (str): Directory to save the MOT files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cycle, data in samples.items():
        filename = os.path.join(output_dir, f"gait_cycle_{cycle}.mot")
        with open(filename, 'w') as file:
            # Write metadata
            file.write("walking_incline\n")  # Example descriptive line
            for key, value in metadata.items():
                if key not in ["heel_strikes_left", "heel_strikes_right"]:
                    file.write(f"{key} = {value}\n")
            file.write("\n")  # Blank line separating metadata and data

            # Write column headers
            file.write("\t".join(column_labels) + "\n")

            # Write data without extra blank lines
            data.to_csv(file, sep='\t', index=False, header=False, lineterminator='\n')

        print(f"Wrote MOT file: {filename}")

def detect_heel_strikes(vertical_force, threshold=20):
    """
    Detect heel strikes from vertical ground reaction forces.

    Returns:
        list: Indices of detected heel strikes.
    """
    heel_strikes = []
    for i in range(1, len(vertical_force)):
        if vertical_force[i] > threshold and vertical_force[i - 1] <= threshold:
            heel_strikes.append(i)
    return heel_strikes

def read_mot_files(data_path_mot, threshold=20):
    """
    Reads MOT files from a directory, extracts metadata and data, and detects heel strikes separately for left and right.
    """
    motion_data_list = []

    file_list = sorted(f for f in os.listdir(data_path_mot) if f.endswith('.mot'))

    for filename in file_list:
        file_path = os.path.join(data_path_mot, filename)
        print(f"Reading MOT file: {file_path}")

        try:
            with open(file_path, 'r') as file:
                # Read metadata
                metadata = {}
                descriptive_line = file.readline().strip()  # Ignore descriptive line (e.g., 'walking_speed')

                for _ in range(5):  # Skip the remaining metadata rows until 'endheader'
                    line = file.readline().strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        metadata[key.strip()] = value.strip()

                # Skip the blank line after the header
                file.readline()

                # Read data into a DataFrame
                data = pd.read_csv(file, sep=r'\s+')

                # Detect heel strikes separately for left and right
                if 'ground_force1_vy' in data.columns and 'ground_force2_vy' in data.columns:
                    heel_strikes_left = detect_heel_strikes(data['ground_force1_vy'], threshold)
                    heel_strikes_right = detect_heel_strikes(data['ground_force2_vy'], threshold)
                    metadata['heel_strikes_left'] = heel_strikes_left
                    metadata['heel_strikes_right'] = heel_strikes_right

            # Append the parsed data to the list
            motion_data_list.append({
                'fileName': filename,
                'rawData': data,
                'colNames': data.columns.to_list(),
                'metadata': metadata
            })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return motion_data_list

def get_samples(mot_data, heel_strikes):
    """
    Retrieve MOT data samples based on detected heel strikes.

    Parameters:
        mot_data (pd.DataFrame): The motion data.
        heel_strikes (list): Indices of detected heel strikes.

    Returns:
        dict: Dictionary with heel strike indices and corresponding MOT data.
    """
    samples = {}
    for i in range(1, len(heel_strikes)):
        start = heel_strikes[i - 1]
        end = heel_strikes[i]
        samples[i] = mot_data.iloc[start:end]
    return samples

def visualize_heel_strikes(mot_data, heel_strikes_left, heel_strikes_right, threshold=20):
    """
    Visualizes the detected heel strikes on the vertical ground reaction force data.

    Parameters:
        mot_data (pd.DataFrame): The motion data containing ground reaction forces.
        heel_strikes_left (list): Indices of detected left heel strikes.
        heel_strikes_right (list): Indices of detected right heel strikes.
        threshold (float): The threshold used for detection.
    """
    time = mot_data['time']
    ground_force1_vy = mot_data['ground_force1_vy']
    ground_force2_vy = mot_data['ground_force2_vy']

    plt.figure(figsize=(12, 6))

    # Plot left and right vertical ground reaction forces
    plt.plot(time, ground_force1_vy, label="Left Vertical Force (ground_force1_vy)", alpha=0.8)
    plt.plot(time, ground_force2_vy, label="Right Vertical Force (ground_force2_vy)", alpha=0.8)

    # Mark left and right heel strikes
    plt.scatter(time[heel_strikes_left], ground_force1_vy.iloc[heel_strikes_left], color='red', label="Left Heel Strikes", zorder=5)
    plt.scatter(time[heel_strikes_right], ground_force2_vy.iloc[heel_strikes_right], color='blue', label="Right Heel Strikes", zorder=5)

    # Add threshold line
    plt.axhline(y=threshold, color='gray', linestyle='--', label=f"Threshold ({threshold})")

    plt.xlabel("Time (s)")
    plt.ylabel("Vertical Force (N)")
    plt.title("Heel Strike Detection")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data_path_mot = r"C:\Users\schb998\MyData\Pilot_Ella\Incline_Trials\walking_incline trial NOAFO\mot.corrected"
    output_dir_left = r"C:\Users\schb998\MyData\Pilot_Ella\Incline_Trials\walking_incline trial NOAFO\chopped_mot\Left"
    output_dir_right = r"C:\Users\schb998\MyData\Pilot_Ella\Incline_Trials\walking_incline trial NOAFO\chopped_mot\Right"
    threshold = 20

    # Call the updated function
    motion_data = read_mot_files(data_path_mot, threshold)

    if motion_data:
        # Process each MOT file
        for file_data in motion_data:
            mot_data = file_data['rawData']
            column_labels = file_data['colNames']
            metadata = file_data['metadata']

            # Get samples for left and right gait cycles
            heel_strikes_left = metadata.get('heel_strikes_left', [])
            heel_strikes_right = metadata.get('heel_strikes_right', [])
            left_samples = get_samples(mot_data, heel_strikes_left)
            right_samples = get_samples(mot_data, heel_strikes_right)

            # Write MOT files for left gait cycles
            write_mot_files(left_samples, column_labels, metadata, output_dir_left)

            # Write MOT files for right gait cycles
            write_mot_files(right_samples, column_labels, metadata, output_dir_right)
    else:
        print("No MOT files found in the directory.")

