import os.path
import opensim as osim
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

class CustomExternalLoads(FileObject):
    extension = ".xml"

    def __init__(self, data: osim.ExternalLoads, filename: str, path: str = None):
        super().__init__(filename, data, os.path.basename(path) if path is not None else None)


class CustomJointPower(FileObject):
    extension = ".csv"

    def __init__(self, data: pd.DataFrame, filename: str, path: str = None):
        super().__init__(filename, data, os.path.basename(path) if path is not None else None)

    def save(self, path: str = None):
        self.data.to_csv(path if path is not None else self.filepath, index=False)
