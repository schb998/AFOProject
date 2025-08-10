import os
import sys

import pandas as pd

# Setup OpenSim
opensim_path = r"C:/OpenSim 4.4/bin"
os.environ['OPENSIM_HOME'] = opensim_path
sys.path.append(os.path.join(opensim_path, 'Bindings', 'Python'))
os.environ['PATH'] += os.pathsep + os.path.join(opensim_path, 'bin')

import numpy as np
from scipy.signal import butter, filtfilt
from ptb.util.io.mocap.file_formats import TRC
from ptb.util.osim.osim_store import OSIMStorage, HeadersLabels
import opensim as osim

# Paths
base_path = r"D:\MyData\Testingworkflow\Test"
model_file = os.path.join(base_path, "scaled_model_Ella.osim")
trc_path = os.path.join(base_path, "segmented_trc")
ik_results_path = os.path.join(base_path, "IK_Results")

# Ensure output folders exist
for side in ["Right", "Left"]:
    os.makedirs(os.path.join(ik_results_path, side), exist_ok=True)

# Marker setup
marker_names = [
    'Sternum', 'LShoulder', 'RShoulder', 'LASIS', 'RASIS', 'RPSIS', 'LPSIS',
    'RFibula', 'RShank', 'RAnkleLateral', 'RToe', 'LToe', 'RMT5', 'RMT2', 'RHeel',
    'LFibula', 'LShank', 'LAnkleLateral', 'LMT5', 'LMT2', 'LHeel', 'RKneeLateral',
    'LAnkleMedial', 'LKneeLateral', 'RAnkleMedial', 'LKneeMedial', 'RKneeMedial'
]
do_not_include = ['RKneeMedial', 'RAnkleMedial', 'RToe', 'LKneeMedial', 'LAnkleMedial', 'LToe']

# Butterworth filter
def filter_signals(data, fs=100, cutoff=6, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

# Read .mot using OpenSim Storage
def read_mot_storage(filepath):
    storage = osim.Storage(filepath)
    label_array = storage.getColumnLabels()
    labels = [label_array.get(i) for i in range(label_array.getSize())]

    time_vec = []
    data_vec = []
    for i in range(storage.getSize()):
        row = storage.getStateVector(i)
        time_vec.append(row.getTime())
        data_array = row.getData()
        data_row = [data_array.get(j) for j in range(data_array.getSize())]
        data_vec.append(data_row)

    data = np.array(data_vec)
    time_vec = np.array(time_vec).reshape(-1, 1)
    return labels, np.hstack((time_vec, data))

# Write .mot file with OpenSim-compatible header
# def write_filtered_mot(filename, data, headers):
#     lines = []
#     lines.append("name " + os.path.basename(filename).replace(".mot", "") + "\n")
#     lines.append("version={}\n".format(1))
#     lines.append("nRow={}\n".format(data.shape[0]))
#     lines.append("nColumns={}\n".format(data.shape[1]))
#     lines.append("inDegrees={}\n".format("yes"))
#     # lines.append("range {} {}\n".format(data[0, 0], data[-1, 0]))
#     lines.append("endheader\n")
#     lines.append("\t".join(headers) + "\n")
#     for row in data:
#         lines.append("\t".join([f"{x:.8f}" for x in row]) + "\n")
#     with open(filename, "w") as f:
#         f.writelines(lines)

# Main processing loop
# for side in [""]:

for side in ["Right", "Left"]:
    trc_side_path = os.path.join(trc_path, side)
    out_side_path = os.path.join(ik_results_path, side)
    trc_files = sorted([f for f in os.listdir(trc_side_path) if f.endswith(".trc")])
    # trc_files = [trc_path]
    for i, trc_file in enumerate(trc_files, 1):
        print(f"Processing {side}/{trc_file}...")

        trc_full_path = os.path.join(trc_side_path, trc_file)
        trc = TRC.read(trc_full_path)
        trc_data = trc.to_panda_data_frame()
        t0 = float(trc_data['Time'].iloc[0])
        t1 = float(trc_data['Time'].iloc[-1])

        # Setup IK Tool
        ik_tool = osim.InverseKinematicsTool()
        ik_tool.set_model_file(model_file)
        ik_tool.setMarkerDataFileName(trc_full_path)
        ik_tool.setStartTime(t0)
        ik_tool.setEndTime(t1)

        # Name format
        cycle_name = f"{side.lower()}_cycle_{i}"
        mot_path_temp = os.path.join(base_path, f"{cycle_name}_temp.mot")
        mot_path_final = os.path.join(out_side_path, f"{cycle_name}.mot")
        # mot_path_final = mot_path_temp+".mot"
        ik_tool.setOutputMotionFileName(mot_path_temp)

        # Add marker tasks
        task_set = ik_tool.getIKTaskSet()
        for m in marker_names:
            task = osim.IKMarkerTask()
            task.setName(m)
            task.setApply(m not in do_not_include)
            task.setWeight(1)
            task_set.cloneAndAppend(task)

        ik_tool.run()

        if not os.path.exists(mot_path_temp):
            print(f"IK failed for {trc_file}")
            continue

        # Read and filter
        headers, raw_data = read_mot_storage(mot_path_temp)
        time = raw_data[:, 0]
        signals = raw_data[:, 1:]
        filtered = filter_signals(signals)
        filtered_data = np.column_stack((time, filtered))
        h = OSIMStorage.simple_header_template()
        filename = os.path.split(mot_path_temp)[1]

        h[HeadersLabels.trial] = filename[:filename.rindex('.')]
        h[HeadersLabels.version] = 1
        h[HeadersLabels.nRows] = filtered_data.shape[0]
        h[HeadersLabels.nColumns] = filtered_data.shape[1]
        h[HeadersLabels.inDegrees] = True
        mot = OSIMStorage.create(data=pd.DataFrame(data=filtered_data, columns=headers), header=h, filename=filename)
        mot.write(mot_path_final)
        # Save .mot
        # write_filtered_mot(mot_path_final, filtered_data, headers)

        os.remove(mot_path_temp)

print("\nAll IK trials processed and saved as filtered .mot files.")
