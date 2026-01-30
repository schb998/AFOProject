import os.path
import opensim as osim
import pandas as pd


class FileObject:
    """FileObject. Links the data of a file to its filepath.

    Attributes:
        data:         object containing the data.
        filename:     String indicating the name of the originating file
        filepath:     String pointing to the file associated with the object, if existing.
    """
    def __init__(self, filename: str, data, filepath: str = None) -> None:
        self.data = data
        self.filename = filename
        self.filepath = filepath


class CustomExternalLoads(FileObject):
    """CustomExternalLoads. Links the data of an OpenSim's ExternalLoads file, to its XML filepath.

        Attributes:
        data:         osim.ExternalLoads, external loads object containing the data.
        filename:     String indicating the name of the originating file
        filepath:     String pointing to the XML file associated with the object, if existing
    """
    extension = ".xml"

    def __init__(self, data: osim.ExternalLoads, filename: str, path: str = None) -> None:
        super().__init__(filename, data, os.path.basename(path) if path is not None else None)


class CustomJointPower(FileObject):
    """CustomJointPower. Links the pd.DataFrame data of the joint power computation, to its CSV filepath.

        Attributes:
        data:         pd.DataFrame, external loads object containing the data.
        filename:     String indicating the name of the originating file
        filepath:     String pointing to the CSV file associated with the object, if existing
    """
    extension = ".csv"

    def __init__(self, data: pd.DataFrame, filename: str, path: str = None) -> None:
        super().__init__(filename, data, os.path.basename(path) if path is not None else None)

    def save(self, path: str = None) -> None:
        """Saves the object into a CSV file.

        Args:
            path: full path to the CSV file to save.

        Returns:
            None

        """
        self.data.to_csv(path if path is not None else self.filepath, index=False)
