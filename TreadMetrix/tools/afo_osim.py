import os

import numpy as np
from bokeh.layouts import column
from ptb.util.gait.helpers import OsimHelper
from ptb.util.osim.osim_store import OSIMStorage
import opensim as osim
from ptb.util.data import MocapDO, TRC

import pandas as pd
import copy


class Util:

    @staticmethod
    def add_geo_search_path(custom_geometry_path):
        # todo need to add to ptb
        osim.ModelVisualizer.addDirToGeometrySearchPaths(custom_geometry_path)

    @staticmethod
    def marker_data_from_mot(model_file_path, mot_path, geometry_path=r"C:\OpenSim 4.5\Geometry"):
        Util.add_geo_search_path(geometry_path)
        opens_model = OsimHelper(model_file_path)
        p = [c for c in opens_model.state_variable_names_processed if 'N_A' not in c]
        k0 = copy.deepcopy(opens_model.markerset)
        idx = 1
        index_m = {}
        columns = ['Frame#', 'Time']
        for c in k0.columns:
            index_m[c] = idx
            columns.append("{1}_X{0}".format(index_m[c], c))
            columns.append("{1}_Y{0}".format(index_m[c], c))
            columns.append("{1}_Z{0}".format(index_m[c], c))
            idx+=1
        
        m = OSIMStorage.read(mot_path)
        frame_rate = int(1/m.store.dt)

        frames = []
        for i in range(0, m.store.data.shape[0]):
            joint = pd.Series(data=m.store.data[i, :], index=m.store.column_labels)
            opens_model.set_joints(joint)
            frames.append(copy.deepcopy(opens_model.markerset))

        marker_df = np.zeros([len(frames), len(columns)])
        for i in range(0, len(frames)):
            marker_df[i, 0] = i + 1
            marker_df[i, 1] = i * 1.0 / frame_rate
            frame = frames[i]
            for j in range(0, frame.shape[1]):
                st = j*3+2
                en = st+3
                marker_df[i, st: en] = frame.iloc[:, j].to_numpy()
                pass
            pass

        df = pd.DataFrame(data=marker_df, columns=columns)
        trc = TRC(df)
        trc.headers['DataRate'] = frame_rate
        trc.headers['CameraRate'] = frame_rate
        trc.headers['OrigDataRate'] = frame_rate
        trc.headers['NumFrames'] = len(frames)
        trc.headers['OrigNumFrames'] = len(frames)
        trc.update()
        out_file = "{0}.trc".format(mot_path[:mot_path.rindex('.')])
        trc.write(out_file)
        pass



if __name__ == '__main__':
    Util.marker_data_from_mot(r"C:\Users\ty8on\test_data\S17_scaledmodelIM.osim",
                              r"C:\Users\ty8on\test_data\Inverse Kinematics\walk08 IK.mot")
    pass
