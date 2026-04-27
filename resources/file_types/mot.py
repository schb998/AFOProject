from __future__ import annotations
import bisect
import os
import unittest
from copy import deepcopy
import pandas as pd
import numpy as np
import ast
import random
from typing import Any
Self = Any
import logging
try:
    from ptb.util.data import Yac3do
except ImportError:
    pass
from resources.custom_exceptions import MissingPathException
from resources.file_types.fileobject import FileObject

# todo: double-check operations when int/float/double difference
# todo: check c3d load-write issue

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing_files")
output = os.path.join(path, "test_output")

# working files:
_filename_standard = "MOT_standard.mot"
_filename_nan = "MOT_nan.mot"  # missing data should be handled
_filename_c3d = "C3D_standard.c3d"

class MOTMetadata:
    _string_version: str = "version"
    _string_number_rows: str = "nRows"
    _string_number_columns: str = "nColumns"
    _string_in_degrees: str = "inDegrees"

    def __init__(self, version: int = None, n_rows: int = None, n_columns: int = None, in_degrees: bool = None,
                 additional_metadata: dict = None) -> None:
        """Creates a MOT object.

        Args:
            version: int, version number of the MOT object
            n_rows: int, number of rows of the MOT object's data
            n_columns: int, number of columns of the MOT object's data
            in_degrees: bool, whether the MOT object data is in degrees
        """
        self.version = version
        self.number_rows = n_rows
        self.number_columns = n_columns
        self.in_degrees = in_degrees
        self.additional_metadata = additional_metadata if additional_metadata is not None else {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MOTMetadata):
            return False
        if self.version != other.version:
            return False
        if self.number_rows != other.number_rows:
            return False
        if self.number_columns != other.number_columns:
            return False
        if self.in_degrees != other.in_degrees:
            return False
        return self.additional_metadata == other.additional_metadata


    def __str__(self):
        string = ""
        string = string if self.version is None else string + MOTMetadata._string_version + "=" + str(self.version) + "\n"
        string = string if self.number_rows is None else string + MOTMetadata._string_number_rows + "=" + str(self.number_rows) + "\n"
        string = string if self.number_columns is None else string + MOTMetadata._string_number_columns + "=" + str(self.number_columns) + "\n"
        if self.in_degrees is not None:
            addition = "yes" if self.in_degrees else "no"
            string = string + MOTMetadata._string_in_degrees + "=" + addition + "\n"
        for key in self.additional_metadata.keys():
            string = string + key + "=" + str(self.additional_metadata[key]) + "\n"
        return string

    @classmethod
    def from_dict(cls, dictionary):
        new = MOTMetadata()
        new.version = dictionary.pop(MOTMetadata._string_version) if MOTMetadata._string_version in dictionary else None
        new.number_rows = dictionary.pop(MOTMetadata._string_number_rows) if MOTMetadata._string_number_rows in dictionary else None
        new.number_columns = dictionary.pop(MOTMetadata._string_number_columns) if MOTMetadata._string_number_columns in dictionary else None
        new.in_degrees = dictionary.pop(MOTMetadata._string_in_degrees).strip().lower() == "yes" if MOTMetadata._string_in_degrees in dictionary else None
        for key in dictionary.keys():
            new.additional_metadata[key] = dictionary[key]
        return new


class MOT(FileObject):
    """MOT object.

    Attributes:
        name:         String indicating the name given to the data set
        filename:     String indicating the name of the originating file
        header_lines: MOTMetadata object containing the header lines
        data:         DataFrame containing the data.
        col_names:    List of strings corresponding to the names of the data columns.
        first_frame:  Integer corresponding to the first frame of the data set. Default value = 0.
        filepath:     String pointing to the MOT file associated with the object, if existing.
    """

    extension = ".mot"

    def __init__(self, name: str, filename: str, header_lines: MOTMetadata, data: pd.DataFrame, filepath: str = None) \
            -> None:
        """Creates a MOT object.

        Args:
            name: str, name given to the data set
            filename: str, name of the MOT file associated with the object
            header_lines: MOTMetadata, header lines of the MOT file
            data: pd.DataFrame, data
            filepath: str, path to the corresponding MOT file if existing
        """
        super().__init__(filename, data, filepath)
        self.name = name
        self.header_lines = header_lines
        self.col_names = data.columns.to_list()
        self.first_frame = data.index.values[0]

    def __eq__(self, other: object) -> bool:
        """Overrides the default implementation of equality operation.

        MOT objects are compared on data content. the index of the frames, the name and filename attributes are not considered.

        Args:
            other: object to compare

        Returns:
            bool
        """
        if not isinstance(other, MOT):
            return False
        if self.header_lines != other.header_lines or self.col_names != other.col_names or self.first_frame != other.first_frame:
            return False
        if not self.data.equals(other.data):
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """Overrides the default implementation of inequality operation.

        MOT objects are compared on data content. Name and filename attributes are considered.

        Args:
            other: object to compare

        Returns:
            bool
        """
        return not self.__eq__(other)

    def __gt__(self, other: Self) -> bool:
        """Overrides the default implementation of "strictly greater than" operation.

        MOT objects are compared on the lexical order of their names amd filenames, in that order.

        Args:
            other: MOT object to compare

        Returns:
            bool
        """
        n = deepcopy(self.name).lower()
        on = deepcopy(other.name).lower()
        if n == on:
            return self.filename.lower() > other.filename.lower()
        else:
            return n > on

    def __lt__(self, other: Self) -> bool:
        """Overrides the default implementation of "strictly lower than" operation.

        MOT objects are compared on the lexical order of their filenames.

        Args:
            other: MOT object to compare

        Returns:
            bool
        """
        n = deepcopy(self.name).lower()
        on = deepcopy(other.name).lower()
        if n == on:
            return self.filename.lower() < other.filename.lower()
        else:
            return n < on

    def __le__(self, other: Self) -> bool:
        """Overrides the default implementation of "equal or lower than" operation.

        MOT objects are compared on the lexical order of their filenames.

        Args:
            other: MOT object to compare

        Returns:
            bool
        """
        n = deepcopy(self.name).lower()
        on = deepcopy(other.name).lower()
        if n == on:
            return self.filename.lower() <= other.filename.lower()
        else:
            return n < on

    def __ge__(self, other: Self) -> bool:
        """Overrides the default implementation of "equal or greater than" operation.

        MOT objects are compared on the lexical order of their filenames.

        Args:
            other: MOT object to compare

        Returns:
            bool
        """
        n = deepcopy(self.name).lower()
        on = deepcopy(other.name).lower()
        if n == on:
            return self.filename.lower() >= other.filename.lower()
        else:
            return n > on

    @classmethod
    def load_from_mot(cls, filepath: str, filename: str = None, separator=r'\s+', start_index: int = 1) -> Self:
        """Reads data from a MOT file into a MOT object.

        Args:
            start_index: int, first frame and start of the index. 1 is the default value.
            separator: character used to separate data in the mot file.
                r'\\s' by default. OpenSim generated files require r'\\t'.
            filepath (string): path to the MOT file.
            filename (string): name of the MOT file. \
                Should be filled if path does not include filename, optional otherwise.

        Returns:
            MOT object

        Raises:
            OSError: if the file cannot be read.
        """
        # clean up paths if needed:
        if filename is None:
            filename = os.path.basename(filepath)
        else:
            if os.path.basename(filepath) != filename:
                filepath = os.path.join(filepath, filename)

        error_message = f"File {filename} at {filepath} couldn't be read: "

        # check if path is valid :
        if (not os.path.isfile(filepath)) or (not filepath.endswith(".mot")):
            error_message = error_message + " given path does not lead to a MOT file."
            logging.warning(error_message)
            raise OSError(error_message)

        # read the file:
        try:
            with open(filepath, 'r') as file:
                name = next(file).strip("\n").strip('.mot')
                header_lines = {}
                line = next(file).strip("\n")
                while line != "endheader":
                    if line:
                        temp = line.split('=')
                        if len(temp) > 1:
                            md = temp[1].strip()
                            try:
                                header_lines[temp[0].strip()] = ast.literal_eval(md)
                            except ValueError:
                                header_lines[temp[0].strip()] = md
                    line = next(file).strip("\n")
                metadata = MOTMetadata.from_dict(header_lines)

                data = pd.read_csv(file, sep=separator, engine='python')
                data.index = [i for i in range(start_index, start_index + data.shape[0])]
                file.close()
                return cls(name, filename, metadata, data, filepath = filepath)
        except Exception as e:
            error_message = error_message + getattr(e, 'message', repr(e))
            logging.warning(error_message)
            raise OSError(error_message)

    @classmethod
    def load_from_c3d(cls, filepath: str, filename: str = None) -> Self:
        c3d = Yac3do(filepath)
        c3d_name = os.path.basename(c3d.filename)
        ptb_mot = c3d.c3d_dict

        metadata = MOTMetadata()
        metadata.number_rows = ptb_mot['num_analog_frames']
        metadata.number_columns = ptb_mot['num_analog_channels']

        data_columns = ptb_mot['analog_channels_label']
        first_frame = ptb_mot['first_frame']
        raw_data = ptb_mot['analog_data']
        data = raw_data[data_columns]
        index = [i for i in range(first_frame, first_frame + data.shape[0] - 1)]
        data = pd.DataFrame(data, columns=data_columns, index=index)

        return cls(name=c3d_name.replace(".c3d", "") if filename is None else filename.replace(".mot", ""),
                   filename = c3d_name.replace(".c3d", ".mot") if filename is None else filename,
                   header_lines=metadata,
                   data=data)

    def rename(self, name: str = None, filename: str = None):
        """Updates the MOT object's name and/or file_name.

        Either arguments can be None. This method does nothing if both are None.

        Args:
            name     (str): The new name of the MOT object. Optional.
            filename (str): The new filename of the MOT object. Optional.
        """
        if name is not None:
            self.name = name
        if filename is not None:
            if not filename.endswith(".mot"):
                self.filename = filename + ".mot"
            else:
                self.filename = filename

    def update_data(self, new_data: pd.DataFrame = None, filepath: str = None):
        if new_data is not None:
            self.data = new_data
        self.first_frame = self.data.index.values[0]
        self.col_names = list(self.data.columns)
        self.filepath = filepath

    def save(self, file_path: str = None, file_name: str = None):
        """Writes the MOT object into a MOT file.

        Does so at the given location, using the MOT object's filename parameter as the file's name.

        Args:
            file_path (string): directory in which the file will be written.
            file_name (string): name of the file to write. Optional. \
                If none, the object's filename attribute is used.

        Returns:
            None

        Raises:
            OSError: if the file could not be written.
        """
        # path and filename management:
        if file_name is None:
            file_name = self.filename
        else:
            if not file_name.endswith(".mot"):
                file_name = file_name + ".mot"

        if file_path is None:
            if self.filepath is None:
                raise MissingPathException("path to directory",
                                           f"no path provided to save MOT object {self.filename}")
            else:
                full_path = self.filepath
        else:
            full_path = os.path.join(file_path, file_name)

        os.makedirs(file_path, exist_ok=True)

        # prepare content to be written:
        content = [self.name, "\n", str(self.header_lines), "endheader", "\n"]
        for col in self.col_names:
            content.append(col + "\t")
        content.append("\n")
        for line in range(self.first_frame, self.first_frame + self.data.shape[0]):
            for col in self.data.columns.to_list():
                d = self.data[col][line]
                d0 = str(d) if not np.isnan(d) else ""
                content.append(d0 + "\t")
            content.append("\n")

        # write content:
        try:
            with open(full_path, 'w') as writer:
                writer.writelines(content)
                print(f"File {file_name} written in directory {file_path}.")
        except Exception as e:
            raise OSError(f"Unable to write file {file_name}: {getattr(e, 'message', repr(e))}")
        self.filepath = full_path

    def copy(self) -> Self:
        """Copies and returns a new MOT object.

        "_copy" differentiator added to the returned MOT object's filename and name.

        Returns:
            MOT: Copied MOT object.
        """
        copy = deepcopy(self)
        copy.filename = copy.filename.replace(".mot", "_copy.mot")
        copy.name += '_copy'
        copy.filepath = None
        return copy

    def sample(self, first_point: int | float, last_point: int | float, force_time: bool = False) -> Self:
        """Samples the current MOT file between the given points.

        Object will be sampled at frames if both points are integers and force_time is False, and at time if not.

        Args:
            first_point: int or float, the index or the time of the first frame, included.
            last_point: int or float, the index or the time of the last frame, included.
            force_time: bool, whether the previous are to be read as timestamps even if they're integers

        Returns:
            MOT: sampled MOT object.

        Raises:
            IndexError: if the given points are out of bound for the data.
        """
        frames = sorted((first_point, last_point))
        first_point = frames[0]
        last_point = frames[1]

        message = f"Cannot cut {self.name} at given frames: out of bound index."

        try:
            if force_time or isinstance(first_point, float) or isinstance(last_point, float):
                time_scale = self.data['time']
                first_point = bisect.bisect_left(time_scale, first_point)
                last_point = bisect.bisect_right(time_scale, last_point)

            headers = deepcopy(self.header_lines)
            headers.number_rows = last_point - first_point
            name = self.name + "_segmented_" + str(first_point) + "-" + str(last_point - 1)
            file_name = name + ".mot"
            d = {}
            for col in self.data.columns.to_list():
                d[col] = self.data[col][first_point:last_point]
            return MOT(name, file_name, headers, pd.DataFrame(data=d))
        except IndexError:
            logging.warning(message)
            raise IndexError(message)


    def segment(self, points: list[int], index: bool = False) -> list[Self]:
        """Segments the current MOT file.

        Does so at the given points, returning a list of segmented MOT objects.
        The segments are in the form ]points[i], points[i+1]] except for the first segment who includes the first frame.

        Args:
            points (list of int): list of the frames before which the file needs to be segmented.
            index (bool): whether to rename the segments by their index.
                If False, segment name will include their starting and ending frame.

        Returns:
            list: list of MOT objects.

        Raises:
            IndexError: if the given points are out of bound for the data.
        """
        # sort the frames at which to segment the object:
        points = sorted(points)
        first_frame = self.first_frame
        last_frame =  first_frame + self.data.shape[0]
        if (points[0] < self.first_frame) or (points[-1] > last_frame):
            message = f"Cannot cut {self.name} at given frames: out of bound index."
            logging.warning(message)
            raise IndexError(message)
        points.append(last_frame)
        points.insert(0, self.first_frame)

        resulting_mots = []
        headers = deepcopy(self.header_lines)

        # segment the file:
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            name = self.name + "_segmented_" + str(start) + "-" + str(end - 1) \
                if not index else self.name + "_cycle" + str(i)
            file_name = name + ".mot"

            d = {}
            for col in self.data.columns.to_list():
                d[col] = self.data[col][start-first_frame:end-first_frame]
            headers.number_rows = len(d["time"])
            resulting_mots.append(MOT(name, file_name, deepcopy(headers), pd.DataFrame(data=d)))

        # return:
        return resulting_mots

    @classmethod
    def load_multiple(cls, data_path_mot: str) -> list[Self]:
        """Recursively reads data from MOT files.

        Args:
            data_path_mot (string): path to the MOT files' directory.

        Returns:
            List: list of the MOT objects.
        """
        motion_data_list = []
        file_list = sorted(f for f in os.listdir(data_path_mot) if f.endswith('.mot'))
        for filename in file_list:
            file_path = os.path.join(data_path_mot, filename)
            motion_data_list.append(cls.load_from_mot(file_path))
        return motion_data_list

    @classmethod
    def save_multiple(cls, mots: list[Self], directory_path: str) -> None:
        """Recursively writes MOT object into files.

        Args:
            mots (list): list of MOT objects.
            directory_path (string): output directory.

        Raises:
            OSError: if a file could not be written.
        """
        os.makedirs(directory_path, exist_ok=True)
        for mot in mots:
            try:
                mot.save(directory_path)
            except OSError:
                message = f"Object {mot.name} couldn't be saved."
                logging.warning(message)
                raise OSError(message)


class _MOTCleanup:
    """Static class to clean up test files."""

    @staticmethod
    def delete_mot_file(path_to_mot: str, force_delete: bool = False) -> None:
        """Deletes MOT file located at given filepath.

        Args:
            path_to_mot: path to the MOT file to be deleted.
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

        if not os.path.basename(path_to_mot).endswith('.mot'):
            message = f"Could not delete {path_to_mot}: invalid path."
            logging.warning(message)
            raise OSError(message)

        if force_delete:
            delete(path_to_mot)

        else:
            print(f"Confirm deletion of file {path_to_mot} (y/N):\n")
            confirmation = input().lower().strip()
            if confirmation == 'y' or confirmation == 'yes':
                delete(path_to_mot)
            else:
                logging.info(f"File {path_to_mot} has not been deleted.")

    @staticmethod
    def delete_all_files(path_to_directory: str, force_delete: bool = False) -> None:
        """Deletes all MOT files from given directory.

        Args:
            path_to_directory: path to the directory where all MOT files are to be deleted.
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
            message = f"Could not delete files from {path_to_directory}: path is not a directory."
            logging.warning(message)
            raise OSError(message)

        file_list = sorted(f for f in os.listdir(path_to_directory) if f.endswith('.mot'))

        if force_delete:
            delete(file_list)

        else:
            print(f"This directory contains: " + str(file_list))
            print(f"Confirm deletion of all MOT files from {path_to_directory} (y/[n]):")
            confirmation = input().lower().strip()
            if confirmation == 'y' or confirmation == 'yes':
                delete(file_list)
            else:
                logging.info(f"Files in {path_to_directory} have not been deleted.")


class _Test(unittest.TestCase):
    """Regression tests for the MOT class methods.

    Those tests are used to ensure working methods are not compromised by new code.
    """

    def test_load(self) -> None:
        try:
            MOT.load_from_mot(os.path.join(path, _filename_standard))
            MOT.load_from_mot(path, _filename_nan)
        except OSError:
            self.fail("File not read.")
        self.assertEqual(MOT.load_from_mot(os.path.join(path, _filename_standard)), MOT.load_from_mot(os.path.join(path, _filename_standard)),
            "MOT Object from same file should be equal.")
        self.assertNotEqual(MOT.load_from_mot(os.path.join(path, _filename_standard)),
                            MOT.load_from_mot( os.path.join(path, _filename_nan)),
            "MOT Object from different files should not be equal.")


    def test_nestled_loads(self) -> None:
        try:
            mot = MOT.load_from_mot(os.path.join(path, _filename_nan))
            mot.save(output, "first_save.mot")
            mot_first_save = MOT.load_from_mot(os.path.join(output, "first_save.mot"))
            mot_first_save.save(output, "second_save.mot")
            mot_second_save = MOT.load_from_mot(os.path.join(output, "second_save.mot"))
        except OSError:
            self.fail("Couldn't load and save files in a loop.")
        self.assertEqual(mot, mot_first_save, "Nestled loaded files should be equal.")
        self.assertEqual(mot, mot_second_save, "Nestled loaded files should be equal.")
        _MOTCleanup.delete_all_files(output, True)


    def test_operations(self) -> None:
        mot1 = MOT.load_from_mot(os.path.join(path, _filename_standard))
        mot2 = MOT.load_from_mot(os.path.join(path, _filename_nan))
        self.assertEqual(mot1, mot1, "Equality operation is not working.")
        self.assertEqual(mot2, mot2, "Equality operation is not working.")
        self.assertNotEqual(mot1, mot2, "Equality operation is not working.")
        self.assertNotEqual(mot2, mot1, "Equality operation is not working.")
        self.assertEqual(mot1, MOT.load_from_mot(path, _filename_standard), "Objects loaded from same file should be equal.")
        self.assertEqual(mot2, MOT.load_from_mot(path, _filename_nan),
                         "Objects loaded from same file should be equal.")

        mot3, mot4 = mot1.copy(), mot1.copy()
        mot3.rename(name='foo', filename=mot1.filename)
        mot4.rename(name=mot1.name, filename='foo')
        self.assertGreater(mot1, mot3, "Comparison operation > is not working.")
        self.assertGreaterEqual(mot1, mot3, "Comparison operation >= is not working.")
        self.assertLess(mot3, mot1, "Comparison operation < is not working.")
        self.assertLessEqual(mot3, mot1, "Comparison operation <= is not working.")
        self.assertGreater(mot1, mot4, "Comparison operation > is not working.")
        self.assertGreaterEqual(mot1, mot4, "Comparison operation >= is not working.")
        self.assertLess(mot4, mot1, "Comparison operation < is not working.")
        self.assertLessEqual(mot4, mot1, "Comparison operation <= is not working.")


    def test_copy(self) -> None:
        mot = MOT.load_from_mot(os.path.join(path, _filename_standard))
        self.assertEqual(mot.copy(), mot, "Copy method is not working.")


    def test_sample(self) -> None:
        mot = MOT.load_from_mot(os.path.join(path, _filename_standard))
        length = mot.data.shape[0]

        # test on frame sampling:
        for i in range(5):
            rands = sorted((random.randint(0, length - 1), random.randint(0, length - 1)))
            rand1, rand2 = rands[0], rands[1]
            error_message = f"Sampling method (frame) is not working with values {rand1, rand2}: "
            sample = mot.sample(rand1, rand2)
            self.assertEqual(sample.data.shape[1], mot.data.shape[1], error_message + "wrong number of columns.")
            self.assertEqual(sample.data.shape[0], rand2 - rand1, error_message + "sampling at wrong frames.")
            self.assertEqual(mot.data.shape[0], sample.data.shape[0] + rand1 + (length - rand2), error_message + "sampling at wrong frames.")
            self.assertNotEqual(mot, sample, "original MOT object should not equal sampled objects.")
            sample2 = mot.sample(rand1, rand2)
            self.assertEqual(sample, sample2, error_message + "calls on object with same parameters should be equal.")

        # test on time sampling:
        time_scale = mot.data['time']
        time_firstframe = time_scale[mot.first_frame]
        time_lastframe = time_scale[mot.first_frame + mot.data.shape[0] - 1]

        for i in range(5):
            rands = sorted([random.uniform(time_firstframe, time_lastframe - 1),
                            random.uniform(time_firstframe, time_lastframe - 1)])
            rand1, rand2 = rands[0], rands[1]
            frame1, frame2 = bisect.bisect_left(time_scale, rand1), bisect.bisect_right(time_scale, rand2)

            error_message = f"Sampling method (time) is not working with values {rand1, rand2}: "
            sample = mot.sample(rand1, rand2)

            self.assertEqual(sample.data.shape[1], mot.data.shape[1], error_message + "wrong number of columns.")
            self.assertEqual(sample.data.shape[0], frame2 - frame1, error_message + "sampling at wrong frames.")
            self.assertEqual(mot.data.shape[0], sample.data.shape[0] + frame1 + (length - frame2),
                             error_message + "sampling at wrong frames.")
            self.assertNotEqual(mot, sample, "original MOT object should not equal sampled objects.")
            sample2 = mot.sample(rand1, rand2)
            self.assertEqual(sample, sample2, error_message + "calls on object with same parameters should be equal.")


    def test_segmentation(self) -> None:
        mot = MOT.load_from_mot(os.path.join(path, _filename_standard))
        length = mot.data.shape[0]
        ff = mot.first_frame
        lf = length + ff - 1
        for i in range(5):
            rands = sorted((random.randint(ff, lf), random.randint(ff, lf)))
            rand1, rand2 = rands[0], rands[1]
            error_message = f"Segmentation method is not working with values {rand1, rand2}: "
            mots = mot.segment(rands)
            self.assertEqual(len(mots), 3, error_message + "wrong number of segments.")
            self.assertEqual(mots[0].data.shape[1], mot.data.shape[1], error_message + "wrong number of columns.")
            self.assertEqual(mots[1].data.shape[1], mot.data.shape[1], error_message + "wrong number of columns.")
            self.assertEqual(mots[2].data.shape[1], mot.data.shape[1], error_message + "wrong number of columns.")
            self.assertEqual(mots[0].data.shape[0] + mots[1].data.shape[0] + mots[2].data.shape[0], mot.data.shape[0],
                error_message + "data lost in segmentation.")

            self.assertEqual(mots[0].data.shape[0], mots[0].header_lines.number_rows,
                             error_message + "segmentation at wrong frames.")
            self.assertEqual(mots[0].data.shape[0], rand1 - ff,
                             error_message + "segmentation at wrong frames.")
            self.assertEqual(mots[1].data.shape[0], mots[1].header_lines.number_rows,
                             error_message + "segmentation at wrong frames.")
            self.assertEqual(mots[1].data.shape[0], rand2 - rand1,
                             error_message + "segmentation at wrong frames.")
            self.assertEqual(mots[2].data.shape[0], mots[2].header_lines.number_rows,
                             error_message + "segmentation at wrong frames.")
            self.assertEqual(mots[2].data.shape[0], lf - rand2 + 1,
                             error_message + "segmentation at wrong frames.")

            self.assertNotEqual(mot, mots[0], error_message + "original MOT object should not equal to segmented objects.")
            self.assertNotEqual(mot, mots[1],
                                error_message + "original MOT object should not equal to segmented objects.")
            self.assertNotEqual(mot, mots[2],
                                error_message + "original MOT object should not equal to segmented objects.")

            self.assertEqual(mots, mot.segment([rand1, rand2]),
                             error_message + "calls on object with same parameters should be equal.")


    def test_save(self) -> None:
        mot1 = MOT.load_from_mot(os.path.join(path, _filename_standard))
        try:
            mot1.save(output)
        except OSError:
            self.fail("File not written.")
        try:
            mot2 = MOT.load_from_mot(os.path.join(output, _filename_standard))
        except OSError:
            self.fail("Written file could not be read.")
        self.assertEqual(mot1, mot2, "Write method is not working.")
        _MOTCleanup.delete_all_files(output, True)


    def test_c3d_load(self) -> None:
        try:
            mot = MOT.load_from_c3d(os.path.join(path, _filename_c3d))
        except Exception as e:
            self.fail(f"MOT file couldn't be loaded from C3D file: + {getattr(e, 'message', repr(e))}")
        try:
            mot.save(output)
        except Exception as e:
            self.fail(f"MOT object loaded from C3D file couldn't be saved: + {getattr(e, 'message', repr(e))}")
        mot_copy = MOT.load_from_mot(os.path.join(output, mot.filename))
        self.assertEqual(mot, mot_copy, "MOT object loaded from the save of a C3D-loaded MOT object should equal the original")
        _MOTCleanup.delete_all_files(output, True)


if __name__ == "__main__":
    logging.basicConfig(filename='test.log', level=logging.INFO)
    unittest.main()
    _MOTCleanup.delete_all_files(output)
