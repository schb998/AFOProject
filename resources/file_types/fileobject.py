from __future__ import annotations
import os.path
try:
    import opensim as osim
except ImportError:
    class osim:
        ExternalLoads = type('ExternalLoads', (), {})
import pandas as pd


class FileObject:
    def __init__(self,
                 filename: str,
                 data,
                 filepath: str = None) \
            -> None:
        self.data = data
        self.filename = filename
        self.filepath = filepath

class CustomExternalLoads:
    extension = ".xml"

    def __init__(self, external_loads: osim.ExternalLoads, path: str = None):
        self.external_loads = external_loads
        path = path if path is not None else external_loads.getAbsolutePathString()
        self.filepath = path
        self.filename = os.path.basename(path) if path is not None else None


class CustomJointPower:
    extension = ".csv"

    def __init__(self, joint_power: pd.DataFrame, path: str = None):
        self.joint_power = joint_power
        self.filepath = path
        self.filename = os.path.basename(path) if path is not None else None

    def save(self, path: str = None):
        self.joint_power.to_csv(path if path is not None else self.filepath, index=False)
