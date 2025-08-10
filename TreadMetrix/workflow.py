import copy
import os.path
import pandas as pd
import numpy as np
from ptb.util.osim.osim_store import OSIMStorage
from ptb.util.math.filters import Butterworth


class AFOMinions:
    def __init__(self, name=None):
        self.name = name
        self.parcel = None
        pass

    def run(self):
        return Bapple()

class Kevin(AFOMinions):
    def __init__(self, filepath):
        """
        This minion process the force data: butterworth filtering,
        thresholding and zeroing forces in swing phases.
        """
        super().__init__("Kevin")
        self.filepath = filepath
        self.batch = False
        if os.path.isdir(self.filepath):
            self.batch = True
        self.data = {'original_mot': None}

    def butter(self, data:pd.DataFrame):
        cols = [c for c in data.columns if 'time' not in c.lower()]
        time_col = [c for c in data.columns if 'time' in c.lower()][0]
        tx = data[time_col].to_numpy()
        data_to_butter = data[cols].to_numpy()
        butter_data = copy.deepcopy(data_to_butter)
        dt = np.mean((tx[1:]-tx[:-1]))
        for x in range(0, data_to_butter.shape[1]):
            b = Butterworth.butter_low_filter(data=data_to_butter[:, x],
                                              cut=6,
                                              fs=1/dt,
                                              order=4)
            butter_data[:, x] = b
            pass
        data.iloc[:, 1:] = butter_data
        return data

    def get_parcel(self, b):
        self.parcel = b

    def run_batch(self):
        return Bapple()

    def run_single(self, filepath):
        # TreadMetrix
        if filepath is not None:
            m = OSIMStorage.read(filepath)
            c = self.butter(m.to_pandas())
            pass
        return Bapple()

    def run(self):
        if self.batch:
            return self.run_batch()
        else:
            return self.run_single(self.filepath)


class Bapple:
    def parcel(self):
        pass

if __name__ == "__main__":
    workflow = [Kevin("I:/walking_speed_NoAFO.mot")]
    for w in workflow:
        w.run()
    pass