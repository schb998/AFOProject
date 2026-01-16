import bisect
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import ast
import random
from typing import Self
import logging
import re
from ptb.util.io.mocap.low_lvl.c3d import Reader
from resources.custom_exceptions import *

# todo: double-check operations when int/float/double difference

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing_files")
output = os.path.join(path, "test_output")

# working files:
_filename_standard = "TRC_standard.trc"
_filename_nan = "TRC_nan.trc"  # missing values should be handled
_filename_c3d = "C3D_standard.c3d"
# error management files:
_filename_missing_z7 = "TRC_missing_z7.trc"  # error : missing marker coordinate z7

coordinates_names = ['X', 'Y', 'Z', 'T', 'N']

class TRCMetadata(object):
    _string_data_rate: str = "DataRate"
    _string_camera_rate : str = "CameraRate"
    _string_num_frames: str = 'NumFrames'
    _string_num_markers: str = 'NumMarkers'
    _string_units: str = 'Units'
    _string_og_data_rate: str = 'OrigDataRate'
    _string_og_start_frame: str = 'OrigDataStartFrame'
    _string_og_num_frames: str = 'OrigNumFrames'


    def __init__(self, metadata: dict[str, str | int | float]):
        """Creates a TRCMetadata object."""

        self.data_rate = metadata.pop(self._string_data_rate) if self._string_data_rate in metadata else None
        self.camera_rate = metadata.pop(self._string_camera_rate) if self._string_camera_rate in metadata else None
        self.num_frames = metadata.pop(self._string_num_frames) if self._string_num_frames in metadata else None
        self.num_markers = metadata.pop(self._string_num_markers) if self._string_num_markers in metadata else None
        self.units = metadata.pop(self._string_units) if self._string_units in metadata else None
        self.og_data_rate = metadata.pop(self._string_og_data_rate) if self._string_og_data_rate in metadata else None
        self.og_start_frame = metadata.pop(self._string_og_start_frame) if self._string_og_start_frame in metadata else None
        self.og_num_frames = metadata.pop(self._string_og_num_frames) if self._string_og_num_frames in metadata else None
        self.additional_metadata = {}
        for key in metadata.keys():
            self.additional_metadata[key] = metadata[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TRCMetadata):
            return False
        if self.data_rate != other.data_rate:
            return False
        if self.camera_rate != other.camera_rate:
            return False
        if self.num_frames != other.num_frames:
            return False
        if self.num_markers != other.num_markers:
            return False
        if self.units != other.units:
            return False
        if self.og_data_rate != other.og_data_rate:
            return False
        if self.og_start_frame != other.og_start_frame:
            return False
        if self.og_num_frames != other.og_num_frames:
            return False
        return self.additional_metadata == other.additional_metadata

    def __str__(self):
        str_names = ""
        str_values = ""

        if self.data_rate is not None:
            str_names = str_names + TRCMetadata._string_data_rate + "\t"
            str_values = str_values + str(self.data_rate) + "\t"
        if self.camera_rate is not None:
            str_names = str_names + TRCMetadata._string_camera_rate + "\t"
            str_values = str_values + str(self.camera_rate) + "\t"
        if self.num_frames is not None:
            str_names = str_names + TRCMetadata._string_num_frames + "\t"
            str_values = str_values + str(self.num_frames) + "\t"
        if self.num_markers is not None:
            str_names = str_names + TRCMetadata._string_num_markers + "\t"
            str_values = str_values + str(self.num_markers) + "\t"
        if self.units is not None:
            str_names = str_names + TRCMetadata._string_units + "\t"
            str_values = str_values + str(self.units) + "\t"
        if self.og_data_rate is not None:
            str_names = str_names + TRCMetadata._string_og_data_rate + "\t"
            str_values = str_values + str(self.og_data_rate) + "\t"
        if self.og_start_frame is not None:
            str_names = str_names + TRCMetadata._string_og_start_frame + "\t"
            str_values = str_values + str(self.og_start_frame) + "\t"
        if self.og_num_frames is not None:
            str_names = str_names + TRCMetadata._string_og_num_frames + "\t"
            str_values = str_values + str(self.og_num_frames) + "\t"
        for key in self.additional_metadata.keys():
            str_names = str_names + key + "\t"
            str_values = str_values + str(key) + "\t"
        return str_names + "\n" + str_values + "\n"



class TRC(object):
    """TRC object.

    Attributes:
        filename:    String indicating the name of the originating file.
        metadata:    Dictionary with the TRC metadata.
        marker_set:  List of the markers used.
        col_names:   List of the names of the data columns.
        marker_dict: Dictionary of the columns associated with each marker.
        data:        Dataframe containing the data. The frames are used as index.
        first_frame: Integer, first frame of the data.
        num_coordinates: Integer, number of coordinates by marker
        file_header: List of string, content of the TRC file's header line. Optional.
        filepath: String of the path to the matching TRC file, if existing

    """

    extension = ".trc"

    def __init__(self, filename: str, meta_data: TRCMetadata, marker_set: list[str],
                 col_names: list[str], marker_dict: dict[str, list[str]], data: pd.DataFrame, num_coordinates: int,
                 file_header: list[str] = None, filepath : str = None) \
            -> None:
        """Creates a TRC object.

        Args:
            filename: name of the TRC file associated with the object
            meta_data: metadata of the dataset
            marker_set: markers of the data
            col_names: (ordered) list of the markers' data coordinates from the file
            marker_dict: build in the form of {marker: list of associated coordinate columns}
            data: data
            num_coordinates: number of coordinates by marker
            file_header: header line of the file
            filepath: str, path to the matching TRC file, if existing
        """
        self.filename = filename
        self.metadata = meta_data
        self.marker_set = marker_set
        self.col_names = col_names
        self.marker_dict = marker_dict
        self.data = data
        self.num_coordinates = num_coordinates
        self.first_frame = data.index[0]
        self.file_header = file_header if file_header is not None else []
        self.filepath = filepath

    def __eq__(self, other: object) -> bool:
        """Overrides the default implementation of equality operation.

        TRC objects are compared on data content. Filename and file_header attributes are not considered.

        Args:
            other: object to compare

        Returns:
            bool
        """
        if not isinstance(other, TRC):
            return False
        if (self.metadata != other.metadata) \
                or (self.marker_set != other.marker_set) \
                or (self.col_names != other.col_names) \
                or (self.marker_dict != other.marker_dict) \
                or (self.first_frame != other.first_frame) \
                or not (self.data.equals(other.data)):
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """Overrides the default implementation of inequality operation.

        TRC objects are compared on data content. Filename and file_header attributes are not considered.

        Args:
            other: object to compare

        Returns:
            bool
        """
        return not self.__eq__(other)

    def __gt__(self, other: Self) -> bool:
        """Overrides the default implementation of "strictly greater than" operation.

        TRC objects are compared on the lexical order of their filenames.

        Args:
            other: TRC object to compare

        Returns:
            bool
        """
        return self.filename.lower() > other.filename.lower()

    def __lt__(self, other: Self) -> bool:
        """Overrides the default implementation of "strictly lower than" operation.

        TRC objects are compared on the lexical order of their filenames.

        Args:
            other: TRC object to compare

        Returns:
            bool
        """
        return self.filename.lower() < other.filename.lower()

    def __le__(self, other: Self) -> bool:
        """Overrides the default implementation of "equal or lower than" operation.

        TRC objects are compared on the lexical order of their filenames.

        Args:
            other: TRC object to compare

        Returns:
            bool
        """
        return self.filename.lower() <= other.filename.lower()

    def __ge__(self, other: Self) -> bool:
        """Overrides the default implementation of "equal or greater than" operation.

        TRC objects are compared on the lexical order of their filenames.

        Args:
            other: TRC object to compare

        Returns:
            bool
        """
        return self.filename.lower() >= other.filename.lower()

    @classmethod
    def load_from_trc(cls, filepath: str, filename: str = None, header: bool = True, delimiter: str = "\t",
                      num_coordinates=None) -> Self:
        """Reads data from a TRC file.

        Args:

            filepath: path to the TRC file.
            filename:  name of the TRC file. \
                Should be filled if path does not include filename, optional otherwise.
            header: whether the TRC file includes a header. Default value is True.
            delimiter: delimiter of the TRC file. Default value is "\t".
            num_coordinates: number of coordinates by marker

        Returns:
            TRC object

        Raises:
            OSError: if the file cannot be read.
        """
        # clean up paths if needed:
        if filename is None:
            filename = os.path.basename(filepath)
        else:
            if os.path.basename(filepath) != filename:
                filepath = os.path.join(filepath, filename)

        error_message = f"TRC object could not be loaded from file {filepath}: "

        # test that given path is valid :
        if (not os.path.isfile(filepath)) or (not filepath.endswith(".trc")):
            error_message = error_message + " given path does not lead to a TRC file."
            logging.warning(error_message)
            raise OSError(error_message)

        # read the file:
        try:
            with open(filepath, 'r') as file:
                # header of the files:
                if header:
                    file_header = next(file).strip().split(delimiter)

                    if num_coordinates is None:
                        for element in file_header:
                            if re.search(r'^\(.*\)$', element) is not None:
                                num_coordinates = len(element.split("/"))

                if num_coordinates is None:
                    num_coordinates = 3

                # meta_data
                meta_data = {}
                meta_data_keys = next(file).strip().split(delimiter)
                meta_data_values = next(file).strip().split(delimiter)
                for i in range(0, len(meta_data_keys)):
                    md = meta_data_values[i].strip()
                    try:
                        meta_data[meta_data_keys[i]] = ast.literal_eval(md)
                    except ValueError:
                        meta_data[meta_data_keys[i]] = md
                meta_data = TRCMetadata(meta_data)

                # data headers:
                headers = next(file).strip().split(delimiter)
                headers = [headers[i] for i in range(0, len(headers)) if len(headers[i]) > 0]
                sub_headers = next(file).strip().split(delimiter)
                if len(sub_headers) != (len(headers) - 2) * num_coordinates:
                    raise Exception(f"Issue reading {filepath}: wrong number of columns")
                sub_headers.insert(0, headers[0])
                sub_headers.insert(1, headers[1])

                # data:
                data = pd.read_csv(file, sep=r'\s', names=sub_headers, engine='python', index_col=headers[0])

                # close the file:
                file.close()

                # marker management:
                marker_set = headers[2:]
                sub_headers = data.columns.tolist()
                marker_dictionary = {}
                i = 1
                for m in range(len(marker_set)):
                    marker_dictionary[marker_set[m]] = sub_headers[i:i + num_coordinates]
                    i += num_coordinates

                res = cls(filename, meta_data, marker_set, sub_headers, marker_dictionary, data, num_coordinates,
                          file_header if header else None, filepath = filepath)
                logging.info(f'TRC object successfully loaded from file {filepath}.')
                return res
        except Exception as e:
            error_message = error_message + getattr(e, 'message', repr(e))
            logging.warning(error_message)
            raise OSError(error_message)

    @classmethod
    def load_from_c3d(cls, c3d: str, filename: str = None) -> Self:
        """Reads data from a C3D file.

        Args:
            c3d:  path to a c3d file.
            filename: name of the TRC file. Optional. if not given, filename will be the same as the c3d file.

        Returns:
            TRC object

        Raises:
            OSError: if the file cannot be read.
        """

        error_message = f"TRC object could not be loaded from file {c3d}: "

        # test that given path is valid :
        if (not os.path.isfile(c3d)) or (not c3d.endswith(".c3d")):
            error_message = error_message + "given path does not lead to a C3D file."
            logging.warning(error_message)
            raise OSError(error_message)

        with open(c3d, 'rb') as file:
            reader = Reader(file)

            if filename is None:
                filename = os.path.basename(c3d).replace(".c3d", ".trc")
            first_frame = reader.first_frame()

            # get the number of frames and markers:
            for _, points, _ in reader.read_frames():
                num_markers = len(points)
                num_coordinates = len(points[0])
                break

            # read metadata :
            meta_data = {}
            rate = reader.header.frame_rate
            frames = reader.header.last_frame - reader.header.first_frame + 1
            meta_data['CameraRate'] = rate
            meta_data['DataRate'] = rate
            meta_data['NumFrames'] = frames
            meta_data['OrigDataRate'] = rate
            meta_data['OrigDataStartFrame'] = reader.header.first_frame
            meta_data['OrigNumFrames'] = frames
            meta_data['Units'] = reader.groups['POINT'].params['UNITS'].bytes.decode("utf-8")
            meta_data['NumMarkers'] = num_markers
            meta_data = TRCMetadata(meta_data)

            # organize markers and their coordinates:
            marker_set = [point.strip() for point in reader.point_labels]
            marker_dictionary = {}
            columns = []
            for i in range(len(marker_set)):
                label = marker_set[i]
                str_i = str(i+1)
                coo_list = []
                global coordinates_names
                if num_coordinates > len(coordinates_names):
                    raise Exception("Too many coordinates to unpack.")
                for j in range(num_coordinates):
                    coo_list.append(coordinates_names[j] + str_i)
                columns.extend(coo_list)
                marker_dictionary[label] = coo_list

            # data:
            data = pd.DataFrame(columns=np.array(columns), index=range(first_frame, reader.last_frame()))
            for frame_no, points, analog in reader.read_frames():
                temp = []
                for marker in points:
                    temp.extend(marker)
                data.loc[frame_no] = temp
            num_coordinates = len(points[0])

            res = cls(filename, meta_data, marker_set, columns, marker_dictionary, data, num_coordinates,
                      file_header=None)
            logging.info(f'TRC object successfully loaded from C3D.')

            # close the file:
            file.close()

            return res

    def save(self, filepath: str = None, filename: str = None) -> None:
        """Saves data into a TRC file.

        Args:
            filepath (string): path to the directory in which to save the TRC file.
            filename (string): name of the save file. Optional. If not filled, attribute filename will be used.

        Raises:
            OSError: if the file cannot be saved.
        """
        if filename is None:
            filename = self.filename

        if filepath is None :
            if self.filepath is not None:
                filepath = os.path.dirname(self.filepath)
            else:
                raise MissingPathException("path to directory",
                                           f"no path provided to save TRC object {self.filename}")

        error_message = f"TRC object {filename} couldn't be saved in {filepath}: "

        # check if valid path:
        try:
            os.makedirs(filepath, exist_ok=True)
        except Exception as e:
            error_message = error_message + getattr(e, 'message', repr(e))
            logging.warning(error_message)
            raise OSError(error_message)

        content = []

        if self.file_header is not None:
            line = ""
            for header in self.file_header:
                line += f"{header}\t"
            content.append(line.strip() + "\n")
        content.append(str(self.metadata))
        c0 = "Frame#\tTime\t"
        c1 = "\t\t"
        for marker_data in self.marker_set:
            c0 += f"{marker_data}\t\t\t"
            c1 += (f"{self.marker_dict[marker_data][0]}\t"
                   + f"{self.marker_dict[marker_data][1]}\t"
                   + f"{self.marker_dict[marker_data][2]}\t")
        content.append(c0.strip() + "\t\t\n")
        content.append("\t\t" + c1.strip() + "\n")

        frames = self.data.index

        for line in range(self.data.shape[0]):
            c0 = str(frames[line]) + "\t"
            for col in self.data.columns.tolist():
                d = self.data[col][line + self.first_frame]
                d0 = str(d) if not np.isnan(d) else ""
                c0 += f"{d0}\t"
            c0 += '\n'
            content.append(c0)

        full_path = os.path.join(filepath, filename)
        with open(full_path, 'w') as writer:
            writer.writelines(content)
        logging.info(f"File {filename} saved in directory {filepath}.")
        self.filepath = full_path

    @classmethod
    def adapt_to_opensim_use(cls, filepath: str, filename: str = None, header: bool = True,
                             delimiter: str = "\t") -> None:
        """Overwrites a TRC file with a copy of data with added marker ZERO located at position (0,0,0) at all
        frames, as last marker column.

        Used to arrange TRC files to use in OpenSim.

        Args:
            filepath: path to the TRC file.
            filename:  name of the TRC file. \
                Should be filled if path does not include filename, optional otherwise.
            header: whether the TRC file includes a header. Default value is True.
            delimiter: delimiter of the TRC file. Default value is "\t".

        Returns:
            None

        Raises:
            OSError: if the file cannot be read.

        """
        if filename is None:
            temp = os.path.split(filepath)
            filename = temp[1]
            filepath = temp[0]
        trc = cls.load_from_trc(filepath, filename, header, delimiter)
        old_name = deepcopy(trc.filename)
        num_frames = trc.data.shape[0]
        if 'ZERO' in trc.marker_set and trc.col_names[-1] == trc.marker_dict['ZERO'][-1]:
            return
        trc.add_marker('ZERO', {'X': np.zeros(num_frames),
                                'Y': np.zeros(num_frames),
                                'Z': np.zeros(num_frames)})
        trc.rename(old_name)
        trc.save(filepath)

    def rename(self, filename: str):
        """This method updates the TRC object's name and/or file_name.

        Args:
            filename (str): The new filename of the TRC object.
        """
        if not filename.endswith(".trc"):
            self.filename = filename + ".trc"
        else:
            self.filename = filename

    def add_marker(self, marker_name: str, data: np.ndarray | dict[str, np.ndarray]) -> None:
        """Adds a marker to the data.

        Args:
            marker_name: name of the marker
            data: marker's trajectory. Can be either a numpy array or a directory.
            Raises an exception if it does not contain data for exactly 3 coordinates.

        Returns:
            None

        Raises:
            Exception if given data does not contain exactly 3 coordinates.
        """
        if ((isinstance(data, dict) and len(data) != 3)
                or (isinstance(data, np.ndarray) and data.shape[1] != 3)):
            raise Exception("Markers require three coordinates in order to be added.")

        # manage marker name
        name = marker_name
        i = 2
        while name in self.marker_set:
            logging.info(f"Marker {name} already exists, changing name to {marker_name + str(i)}")
            name = marker_name + str(i)
            i = i + 1
        marker_name = name
        self.marker_set.append(marker_name)
        self.metadata.num_markers = self.metadata.num_markers + 1

        num = str(len(self.marker_set))
        new_x_column_name, new_y_column_name, new_z_column_name = 'X' + num, 'Y' + num, 'Z' + num

        if isinstance(data, np.ndarray):
            result = {new_x_column_name: data[:, 0:1],
                      new_y_column_name: data[:, 1:2],
                      new_z_column_name: data[:, 2:3]}

        else:
            # manage marker coordinates:
            content = deepcopy(data)
            columns = list(content.keys())
            try:
                x_column_name = [x for x in columns if re.search("^([Xx])|([Xx])$", x) is not None][0]
                y_column_name = [y for y in columns if re.search("^([Yy])|([Yy])$", y) is not None][0]
                z_column_name = [z for z in columns if re.search("^([Zz])|([Zz])$", z) is not None][0]

            except KeyError:
                logging.info(f"Given columns do not match expected X/Y/Z name formulation "
                             f"of starting or ending by X/Y/Z. "
                             f"Assigning them to coordinates X, Y, Z in this order.")
                x_column_name = columns[0]
                y_column_name = columns[1]
                z_column_name = columns[2]

            result = {new_x_column_name: content[x_column_name],
                      new_y_column_name: content[y_column_name],
                      new_z_column_name: content[z_column_name]}

        new_cols = list(result.keys())
        self.marker_dict[marker_name] = new_cols
        self.col_names.extend(new_cols)
        for coo in new_cols:
            self.data[coo] = result[coo]
        self.rename(self.filename.replace('.trc', f'added_{marker_name}'))

    def copy(self) -> Self:
        """Returns a copy of the object.

        Returns:
            TRC object
        """
        copy = deepcopy(self)
        copy.filename = copy.filename.replace(".trc", "_copy.trc")
        return copy

    def sample(self, first_point: int | float, last_point: int | float, force_time: bool = False) -> Self:
        """Samples the current TRC file between the given points.

        Object will be sampled at frames if both points are integers and force_time is False, and at time if not.

        Args:
            first_point: int or float, the index or the time of the first frame, included.
            last_point: int or float, the index or the time of the last frame, included.
            force_time: bool, whether the previous are to be read as timestamps even if they're integers

        Returns:
            TRC: sampled TRC object.

        Raises:
            IndexError: if the given points are out of bound for the data.
        """
        frames = sorted((first_point, last_point))
        first_point = frames[0]
        last_point = frames[1]

        ff = self.first_frame

        if isinstance(first_point, int) and isinstance(last_point, int) and not force_time:
            if (first_point < ff) or (last_point > ff + self.data.shape[0]):
                raise IndexError("Cannot cut at given frames: out of bound index.")

        else:
            time_scale = self.data['Time']
            if first_point < time_scale[ff] or last_point > time_scale[ff + self.data.shape[0] - 1]:
                raise IndexError("Cannot cut at given times: out of bound index.")

            first_point = bisect.bisect_left(time_scale, first_point)
            last_point = bisect.bisect_right(time_scale, last_point)

        file_name = self.filename.replace('.trc', "_segmented_" + str(first_point) + "-" + str(last_point - 1) + '.trc')
        metadata = deepcopy(self.metadata)
        metadata.num_frames = last_point - first_point

        d = {}
        for col in self.data.columns.to_list():
            d[col] = self.data[col][first_point - ff:last_point - ff]
        return TRC(file_name, metadata, deepcopy(self.marker_set), deepcopy(self.col_names), deepcopy(self.marker_dict),
                   pd.DataFrame(data=d), self.num_coordinates, file_header=deepcopy(self.file_header))

    def segment(self, points: list[int], index: bool = False) -> list[Self]:
        """Segments the data frame according to the given frames point.

         Each fragment contains frames from points[i-1] (included) to points[i] (excluded),
          with added segments: first segment from first_frame_of_data (included) to points[0] (excluded)
          and last segment from points[-1] (included) to last_frame_of_data (included).

        Parameters:
            points: list of integer, frames to segment the object at.
            index (bool): whether to rename the segments by their index.
                If False, segment name will include their starting and ending frame.

        Returns:
            List of the segmented TRC objects.

        Raises:
            IndexError if given points out of index.

        """
        # sort the frames at which to segment the object:
        points = sorted(points)
        ff = self.first_frame
        lf = ff + self.data.shape[0]
        if (points[0] < ff) or (points[-1] > lf):
            message = f"Cannot cut {self.filename} at given frames: out of bound index."
            logging.warning(message)
            raise IndexError(message)
        points.append(lf)
        points.insert(0, ff)

        resulting_trcs = []

        # segment the file:
        for i in range(len(points) - 1):
            start = points[i] + 1 if i != 0 else points[i]
            end = points[i + 1] + 1 if i != len(points) - 1 else points[i + 1]

            file_name = self.filename.replace(".trc", "_segmented_" + str(start) + "-" + str(end - 1) + ".trc") \
                if not index else self.filename.replace(".trc", "_cycle" + str(i) + ".trc")

            d = {}
            for col in self.data.columns.to_list():
                d[col] = self.data[col][start-ff:end-ff]
            metadata = deepcopy(self.metadata)
            metadata.num_frames = len(d["Time"])
            resulting_trcs.append(TRC(file_name, metadata, deepcopy(self.marker_set), deepcopy(self.col_names),
                                      deepcopy(self.marker_dict), pd.DataFrame(data=d), self.num_coordinates,
                                      file_header=deepcopy(self.file_header)))

        return resulting_trcs

    @classmethod
    def load_all(cls, dir_path: str) -> list[Self]:
        """Loads TRC objects from TRC files.

        Args:
            dir_path (string): path of the directory containing the TRC files.

        Returns:
            List of the loadable TRC objects

        Raises:
            OSError: if the given path is not a directory.
        """
        if not os.path.isdir(dir_path):
            raise OSError("Given path is not a directory.")
        resulting_trcs = []
        file_list = sorted(f for f in os.listdir(dir_path) if f.endswith('.trc'))
        for file in file_list:
            try:
                resulting_trcs.append(TRC.load_from_trc(os.path.join(dir_path, file)))
            except OSError:
                pass
        return resulting_trcs

    @classmethod
    def save_multiple(cls, trcs: list[Self], dir_path: str) -> None:
        """Recursively writes TRC object into files.

        Args:
            trcs       (list): list of TRC objects.
            dir_path (string): output directory.

        Raises:
            OSError: if a file could not be written.
        """
        os.makedirs(dir_path, exist_ok=True)
        for trc in trcs:
            try:
                trc.save(dir_path)
            except OSError:
                pass

    @classmethod
    def adapt_all_to_opensim_use(cls, dir_path: str) -> None:
        """Adapts all TRC file of given folder to be used with OpenSim. Sees method "adapt_to_opensim_use" for details.

        Args:
            dir_path: path to directory in which to process the TRc files.

        Returns:
            None
        """
        if not os.path.isdir(dir_path):
            raise OSError("Given path is not a directory.")
        file_list = [f for f in os.listdir(dir_path) if f.endswith('.trc')]
        for file in file_list:
            TRC.adapt_to_opensim_use(file)


class _TRCCleanup:
    """Static class to clean up test files."""

    @staticmethod
    def delete_trc_file(path_to_trc: str, force_delete: bool = False) -> None:
        """Deletes TRC file from given path.

        Args:
            path_to_trc: path to the TRC file to be deleted.
            force_delete: whever to skip asking for confirmation before deletion.

        Raises:
            OSError: if a file could not be deleted.
        """

        def delete(file: str) -> None:
            try:
                os.remove(file)
                logging.info(f"File {file} has been deleted.")
            except OSError:
                logging.error(f"File {file} has been deleted.")

        if not os.path.basename(path_to_trc).endswith('.trc'):
            message = f"Could not delete {path_to_trc}: invalid path."
            logging.warning(message)
            raise OSError(message)

        if force_delete:
            delete(path_to_trc)

        else:
            print(f"Confirm deletion of file {path_to_trc} (y/N):\n")
            confirmation = input().lower().strip()
            if confirmation == 'y' or confirmation == 'yes':
                delete(path_to_trc)
            else:
                logging.info(f"File {path_to_trc} has not been deleted.")

    @staticmethod
    def delete_all_files(path_to_directory: str, force_delete: bool = False) -> None:
        """Deletes all TRC files from given path.

        Args:
            path_to_directory: path to the directory where all TRC files are to be deleted.
            force_delete: whever to skip asking for confirmation before deletion.
        """

        def delete(files: list[str]) -> None:
            for file in files:
                try:
                    os.remove(os.path.join(path_to_directory, file))
                    logging.info(f"File {file} has been deleted.")
                except OSError:
                    logging.error(f"File {file} has been deleted.")

        if not os.path.isdir(path_to_directory):
            message = f"Could not delete {path_to_directory}: invalid path."
            logging.warning(message)
            raise OSError(message)

        file_list = sorted(f for f in os.listdir(path_to_directory) if f.endswith('.trc'))

        if force_delete:
            delete(file_list)

        else:
            print(f"This directory contains: " + str(file_list))
            print(f"Confirm deletion of all TRC files from {path_to_directory} (y/N):")
            confirmation = input().lower().strip()
            if confirmation == 'y' or confirmation == 'yes':
                delete(file_list)
            else:
                logging.info(f"Files in {path_to_directory} have not been deleted.")


class _Test:
    """Regression tests for the TRC class methods.

    Those tests are used to ensure working methods are not compromised by new code.
    """

    @staticmethod
    def main() -> None:
        _Test._test_load()
        _Test._test_nestled_loads()
        _Test._test_operations()
        _Test._test_copy()
        for i in range(10):
            _Test._test_sample()
            _Test._test_segmentation()
        _Test._test_save()
        _Test._test_load_all()
        _Test._test_save_all()
        _Test._test_add_marker()
        _Test._test_arrange()
        print("All tests passed. Deleting testing files...")
        _TRCCleanup.delete_all_files(output, True)
        logging.info('All tests passed.')

    @staticmethod
    def _test_load() -> None:
        try:
            TRC.load_from_trc(path, _filename_standard)
            TRC.load_from_trc(path, _filename_nan)
            TRC.load_from_trc(os.path.join(path, _filename_standard))
            TRC.load_from_trc(os.path.join(path, _filename_nan))
            TRC.load_from_c3d(os.path.join(path, _filename_c3d))
            assert True
        except OSError:
            assert False, "File couldn't be loaded."
        try:
            TRC.load_from_trc(os.path.join(path, _filename_missing_z7))
            assert False, f"File {_filename_missing_z7} should raise an exception upon loading."
        except OSError:
            assert True

    @staticmethod
    def _test_nestled_loads() -> None:
        try:
            trc = TRC.load_from_trc(os.path.join(path, _filename_nan))
            trc.save(output, "first_save.trc")
            trc_first_save = TRC.load_from_trc(os.path.join(output, "first_save.trc"))
            trc_first_save.save(output, "second_save.trc")
            trc_second_save = TRC.load_from_trc(os.path.join(output, "second_save.trc"))
            assert True
        except OSError:
            assert False, "Couldn't load and save files in a loop."
        assert trc == trc_first_save == trc_second_save, "Nestled loaded files should be equal."

    @staticmethod
    def _test_operations() -> None:
        trc1 = TRC.load_from_trc(path, _filename_standard)
        trc2 = TRC.load_from_trc(path, _filename_nan)
        assert trc1 == trc1 and trc2 == trc2, \
            "Equality operation is not working."
        assert trc1 != trc2 and trc2 != trc1, \
            "Inequality operation is not working."
        assert trc1 == TRC.load_from_trc(path, _filename_standard) and trc2 == TRC.load_from_trc(path, _filename_nan), \
            "Objects loaded from same file should be equal."
        trc3 = trc1.copy()
        trc3.rename('foo')
        assert trc1 > trc3 and trc1 >= trc3 and trc3 < trc1 and trc3 <= trc1, "Comparison operations are not working."

    @staticmethod
    def _test_copy() -> None:
        trc = TRC.load_from_trc(path, _filename_standard)
        assert trc.copy() == trc, \
            "Objects should be equal to copy."

    @staticmethod
    def _test_sample() -> None:
        trc = TRC.load_from_trc(os.path.join(path, _filename_standard))
        length = trc.data.shape[0]

        # test on frame sampling:
        rands = sorted((random.randint(0, length - 1), random.randint(0, length - 1)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Sampling method (frame) is not working with values {rand1, rand2}: "
        sample = trc.sample(rand1, rand2)
        assert sample.data.shape[1] == trc.data.shape[1], \
            error_message + "wrong number of columns."
        assert sample.data.shape[0] == rand2 - rand1 \
               and trc.data.shape[0] == sample.data.shape[0] + rand1 + (length - rand2), (
                error_message + "sampling at wrong frames.")
        assert trc != sample, \
            error_message + "original TRC object should not equal sampled objects."
        sample2 = trc.sample(rand1, rand2)
        assert sample == sample2, \
            error_message + "calls on object with same parameters should be equal."

        # test on time sampling
        time_scale = trc.data['Time']
        time_firstframe = time_scale[trc.first_frame]
        time_lastframe = time_scale[trc.first_frame + trc.data.shape[0] - 1]

        rands = sorted([random.uniform(time_firstframe, time_lastframe-1),
                        random.uniform(time_firstframe, time_lastframe-1)])
        rand1, rand2 = rands[0], rands[1]
        frame1, frame2 = bisect.bisect_left(time_scale, rand1), bisect.bisect_right(time_scale, rand2)

        error_message = f"Sampling method (time) is not working with values {rand1, rand2}: "
        sample = trc.sample(rand1, rand2)
        assert sample.data.shape[1] == trc.data.shape[1], \
            error_message + "wrong number of columns."
        assert sample.data.shape[0] == frame2 - frame1 \
               and trc.data.shape[0] == sample.data.shape[0] + frame1 + (length - frame2), (
                error_message + "sampling at wrong frames.")
        assert trc != sample, \
            error_message + "original TRC object should not equal sampled objects."
        sample2 = trc.sample(rand1, rand2)
        assert sample == sample2, \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_segmentation() -> None:
        trc = TRC.load_from_trc(os.path.join(path, _filename_standard))
        length = trc.data.shape[0]
        ff = trc.first_frame
        lf = length + ff - 1

        rands = sorted((random.randint(ff, lf), random.randint(ff, lf)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Segmentation method is not working with values {rand1, rand2}: "
        trcs = trc.segment(rands)
        assert len(trcs) == 3, \
            error_message + "wrong number of segments."
        assert trcs[0].data.shape[1] == trc.data.shape[1] \
               and trcs[1].data.shape[1] == trc.data.shape[1] \
               and trcs[2].data.shape[1] == trc.data.shape[1], error_message + "wrong number of columns."
        assert trcs[0].data.shape[0] + trcs[1].data.shape[0] + trcs[2].data.shape[0] == trc.data.shape[0], \
            error_message + "data lost in segmentation."

        assert trcs[0].data.shape[0] == trcs[0].metadata.num_frames == rand1 + 1 - ff \
               and trcs[1].data.shape[0] == trcs[1].metadata.num_frames == rand2 - rand1 \
               and trcs[2].data.shape[0] == trcs[2].metadata.num_frames == lf - rand2, (
                error_message + "segmentation at wrong frames.")

        assert trc != trcs[0] and trc != trcs[1] and trc != trcs[2], \
            error_message + "original TRC object should not equal to segmented objects."
        assert trcs == trc.segment([rand1, rand2]), \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_load_all() -> None:
        trcs = TRC.load_all(path)
        for f in trcs:
            assert f == TRC.load_from_trc(
                os.path.join(path, f.filename)), "Mass loaded objects should match object loaded from the same file."

    @staticmethod
    def _test_save() -> None:
        trc1 = TRC.load_from_trc(os.path.join(path, _filename_standard))
        try:
            trc1.save(output)
            assert True
        except OSError:
            assert False, "File not written."
        try:
            trc2 = TRC.load_from_trc(os.path.join(output, _filename_standard))
            assert True
        except OSError:
            assert False, "Written file could not be read."
        assert trc1 == trc2, \
            "Write method is not working."

    @staticmethod
    def _test_save_all() -> None:
        _TRCCleanup.delete_all_files(output, True)
        trc = TRC.load_from_trc(os.path.join(path, _filename_standard))
        ff = trc.first_frame
        length = trc.data.shape[0]
        rands = sorted((random.randint(ff, length + ff), random.randint(ff, length + ff)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Segmentation + save all method is not working with values {rand1, rand2}: "
        trcs = trc.segment(rands)
        try:
            TRC.save_multiple(trcs, output)
        except Exception as e:
            assert False, error_message + f"Segmented files couldn't be saved: {getattr(e, 'message', repr(e))}"
        try:
            trcs_copied = TRC.load_all(output)
        except Exception as e:
            assert False, error_message + f"Segmented files couldn't be reloaded: {getattr(e, 'message', repr(e))}"
        assert len(trcs) == len(trcs_copied), error_message + "Some file have not been saved"
        trcs_copied.sort()
        trcs.sort()
        for i in range(len(trcs)):
            assert trcs[i] == trcs_copied[i], (error_message + f"File {trcs[i].filename} should be equal to its saved "
                                                               f"and loaded version.")

    @staticmethod
    def _test_add_marker() -> None:
        trc1 = TRC.load_from_trc(path, _filename_standard)
        trc2 = trc1.copy()
        num_frames = trc2.data.shape[0]
        try:
            trc2.add_marker('TEST', {'X': np.zeros(num_frames),
                                     'Y': np.zeros(num_frames),
                                     'Z': np.zeros(num_frames)})
            assert True
        except Exception as e:
            assert False, "Adding marker method should work: " + getattr(e, 'message', repr(e))

        assert trc2.metadata.num_markers == trc1.metadata.num_markers + 1 == len(trc2.marker_set), \
            "Wrong number of markers."

        assert len(trc2.data.columns) == len(trc1.data.columns) + 3, "Wrong number of columns."

    @staticmethod
    def _test_arrange() -> None:
        trc = TRC.load_from_trc(path, _filename_standard).copy()
        trc.rename("test")
        trc.save(output)

        try:
            TRC.adapt_to_opensim_use(output, "test.trc")
            assert True
        except Exception as e:
            assert False, "Arrange method should not raise error: " + getattr(e, 'message', repr(e))
        trc_arranged1 = TRC.load_from_trc(output, "test.trc")
        assert trc != trc_arranged1, "Raw file and arrange file should be different."

        try:
            TRC.adapt_to_opensim_use(output, "test.trc")
            assert True
        except Exception as e:
            assert False, "Arrange method should not raise error: " + getattr(e, 'message', repr(e))
        trc_arranged2 = TRC.load_from_trc(output, "test.trc")
        assert trc_arranged1 == trc_arranged2, "Using arrange on already arranged TRC file should not have effect."


if __name__ == "__main__":
    logging.basicConfig(filename='test.log', level=logging.INFO)
    _Test.main()
