import bisect
import os
from copy import deepcopy
import pandas as pd
import numpy as np
import ast
import random
from typing import Self
import logging

# todo: double-check operations when int/float/double difference

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing_files")
output = os.path.join(path, "test_output")

# working files:
filename_standard = "MOT_standard.mot"
filename_nan = "MOT_nan.mot"  # missing data should be handled


class MOT:
    """MOT object.

    Attributes:
        name:         String indicating the name given to the data set
        filename:     String indicating the name of the originating file
        header_lines: Directory of the header lines and their values.
        data:         DataFrame containing the data.
        col_names:    List of strings corresponding to the names of the data columns.
        first_frame:  Integer corresponding to the first frame of the data set. Default value = 0.
    """

    def __init__(self, name: str,
                 filename: str,
                 header_lines: dict[str: object],
                 data: pd.DataFrame,
                 first_frame: int = 0) \
            -> None:
        """Creates a MOT object.

        Args:
            name: name given to the data set
            filename: name of the MOT file associated with the object
            header_lines: header lines of the MOT file
            data: data
            first_frame: identifier of the first frame of the data
        """
        self.name = name
        self.filename = filename
        self.header_lines = header_lines
        self.data = data
        self.col_names = data.columns.to_list()
        self.first_frame = first_frame

    def __eq__(self, other: object) -> bool:
        """Overrides the default implementation of equality operation.

        MOT objects are compared on data content. Name and filename attributes are not considered.

        Args:
            other: object to compare

        Returns:
            bool
        """
        if not isinstance(other, MOT):
            return False
        if (self.header_lines != other.header_lines) \
                or (self.col_names != other.col_names) \
                or (self.first_frame != other.first_frame) \
                or not (self.data.equals(other.data)):
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
    def load_from_mot(cls, filepath: str, filename: str = None, separator=r'\s+') -> Self:
        """Reads data from a MOT file into a MOT object.

        Args:
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
                data = pd.read_csv(file, sep=separator, engine='python')
                file.close()
                return cls(name, filename, header_lines, data)
        except Exception as e:
            error_message = error_message + getattr(e, 'message', repr(e))
            logging.warning(error_message)
            raise OSError(error_message)

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

    def save(self, file_path: str, file_name: str = None):
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
        os.makedirs(file_path, exist_ok=True)
        if file_name is None:
            file_name = self.filename
        else:
            if not file_name.endswith(".mot"):
                file_name = file_name + ".mot"
        full_path = os.path.join(file_path, file_name)

        # prepare content to be written:
        content = [self.name + "\n"]
        for line in self.header_lines:
            content.append(line + "=" + str(self.header_lines[line]) + "\n")
        content.append("endheader" + "\n")
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

    def copy(self) -> Self:
        """Copies and returns a new MOT object.

        "_copy" differentiator added to the returned MOT object's filename and name.

        Returns:
            MOT: Copied MOT object.
        """
        copy = deepcopy(self)
        copy.filename = copy.filename.replace(".mot", "_copy.mot")
        copy.name += '_copy'
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

        ff = self.first_frame

        if isinstance(first_point, int) and isinstance(last_point, int) and not force_time:
            if (first_point < ff) or (last_point > ff + self.data.shape[0]):
                raise IndexError("Cannot cut at given frames: out of bound index.")

        else:
            time_scale = self.data['time']
            if first_point < time_scale[ff] or last_point > time_scale[ff + self.data.shape[0] - 1]:
                raise IndexError("Cannot cut at given times: out of bound index.")

            first_point = bisect.bisect_left(time_scale, first_point)
            last_point = bisect.bisect_right(time_scale, last_point)

        if (first_point < 0) or (last_point > self.data.shape[0]):
            message = f"Cannot cut {self.name} at given frames: out of bound index."
            logging.warning(message)
            raise IndexError(message)

        headers = deepcopy(self.header_lines)
        headers['nRows'] = last_point - first_point
        name = self.name + "_segmented_" + str(first_point) + "-" + str(last_point - 1)
        file_name = name + ".mot"
        d = {}
        for col in self.data.columns.to_list():
            d[col] = self.data[col][first_point:last_point]
        return MOT(name, file_name, headers, pd.DataFrame(data=d), first_point)

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
            start = points[i] + 1 if i != first_frame else points[i]
            end = points[i + 1] + 1 if i != len(points) - 1 else points[i + 1]

            name = self.name + "_segmented_" + str(start) + "-" + str(end - 1) \
                if not index else self.name + "_cycle" + str(i)
            file_name = name + ".mot"

            d = {}
            for col in self.data.columns.to_list():
                d[col] = self.data[col][start-first_frame:end-first_frame]
            headers['nRows'] = len(d["time"])
            resulting_mots.append(MOT(name, file_name, deepcopy(headers), pd.DataFrame(data=d), start))

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


class _Test:
    """Regression tests for the MOT class methods.

    Those tests are used to ensure working methods are not compromised by new code.
    """

    @staticmethod
    def main() -> None:
        _Test._test_load()
        _Test._test_operations()
        _Test._test_copy()
        for i in range(10):
            _Test._test_sample()
            _Test._test_segmentation()
        _Test._test_save()
        print("All tests passed, deleting testing files...")
        _MOTCleanup.delete_all_files(output, True)
        logging.info('All tests passed.')

    @staticmethod
    def _test_load() -> None:
        try:
            m1 = MOT.load_from_mot(os.path.join(path, filename_standard))
            m2 = MOT.load_from_mot(path, filename_nan)
            assert True
        except OSError:
            assert False, \
                "File not read."
        assert MOT.load_from_mot(os.path.join(path, filename_standard)) == MOT.load_from_mot(
            os.path.join(path, filename_standard)), \
            "MOT Object from same file should be equal."
        assert MOT.load_from_mot(os.path.join(path, filename_standard)) != MOT.load_from_mot(
            os.path.join(path, filename_nan)), \
            "MOT Object from different files should be not equal."

    @staticmethod
    def _test_nestled_loads() -> None:
        try:
            mot = MOT.load_from_mot(os.path.join(path, filename_nan))
            mot.save(output, "first_save.mot")
            mot_first_save = MOT.load_from_mot(os.path.join(output, "first_save.mot"))
            mot_first_save.save(output, "second_save.mot")
            mot_second_save = MOT.load_from_mot(os.path.join(output, "second_save.mot"))
            assert True
        except OSError:
            assert False, "Couldn't load and save files in a loop."
        assert mot == mot_first_save == mot_second_save, "Nestled loaded files should be equal."

    @staticmethod
    def _test_operations() -> None:
        mot1 = MOT.load_from_mot(os.path.join(path, filename_standard))
        mot2 = MOT.load_from_mot(os.path.join(path, filename_nan))
        assert mot1 == mot1 and mot2 == mot2, \
            "Equality operation is not working."
        assert mot1 != mot2 and mot2 != mot1, \
            "Inequality operation is not working."
        assert mot1 == MOT.load_from_mot(path, filename_standard) and mot2 == MOT.load_from_mot(path, filename_nan), \
            "Objects loaded from same file should be equal."
        mot3, mot4 = mot1.copy(), mot1.copy()
        mot3.rename(name='foo', filename=mot1.filename)
        mot4.rename(name=mot1.name, filename='foo')
        assert mot1 > mot3 and mot1 >= mot3 and mot3 < mot1 and mot3 <= mot1 and \
               mot1 > mot4 and mot1 >= mot4 and mot4 < mot1 and mot4 <= mot1, "Comparison operations are not working."

    @staticmethod
    def _test_copy() -> None:
        mot = MOT.load_from_mot(os.path.join(path, filename_standard))
        assert mot.copy() == mot, \
            "Copy method is not working."

    @staticmethod
    def _test_sample() -> None:
        mot = MOT.load_from_mot(os.path.join(path, filename_standard))
        length = mot.data.shape[0]

        # test on frame sampling:
        rands = sorted((random.randint(0, length - 1), random.randint(0, length - 1)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Sampling method (frame) is not working with values {rand1, rand2}: "
        sample = mot.sample(rand1, rand2)
        assert sample.data.shape[1] == mot.data.shape[1], \
            error_message + "wrong number of columns."
        assert sample.data.shape[0] == rand2 - rand1 \
               and mot.data.shape[0] == sample.data.shape[0] + rand1 + (length - rand2), (
                error_message + "sampling at wrong frames.")
        assert mot != sample, \
            error_message + "original MOT object should not equal sampled objects."
        sample2 = mot.sample(rand1, rand2)
        assert sample == sample2, \
            error_message + "calls on object with same parameters should be equal."

        # test on time sampling:
        time_scale = mot.data['time']
        time_firstframe = time_scale[mot.first_frame]
        time_lastframe = time_scale[mot.first_frame + mot.data.shape[0] - 1]

        rands = sorted([random.uniform(time_firstframe, time_lastframe - 1),
                        random.uniform(time_firstframe, time_lastframe - 1)])
        rand1, rand2 = rands[0], rands[1]
        frame1, frame2 = bisect.bisect_left(time_scale, rand1), bisect.bisect_right(time_scale, rand2)

        error_message = f"Sampling method (time) is not working with values {rand1, rand2}: "
        sample = mot.sample(rand1, rand2)
        assert sample.data.shape[1] == mot.data.shape[1], \
            error_message + "wrong number of columns."
        assert sample.data.shape[0] == frame2 - frame1 \
               and mot.data.shape[0] == sample.data.shape[0] + frame1 + (length - frame2), (
                error_message + "sampling at wrong frames.")
        assert mot != sample, \
            error_message + "original MOT object should not equal sampled objects."
        sample2 = mot.sample(rand1, rand2)
        assert sample == sample2, \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_segmentation() -> None:
        mot = MOT.load_from_mot(os.path.join(path, filename_standard))
        length = mot.data.shape[0]
        ff = mot.first_frame
        lf = length + ff - 1
        rands = sorted((random.randint(ff, lf), random.randint(ff, lf)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Segmentation method is not working with values {rand1, rand2}: "
        mots = mot.segment(rands)
        assert len(mots) == 3, \
            error_message + "wrong number of segments."
        assert mots[0].data.shape[1] == mot.data.shape[1] \
               and mots[1].data.shape[1] == mot.data.shape[1] \
               and mots[2].data.shape[1] == mot.data.shape[1], error_message + "wrong number of columns."
        assert mots[0].data.shape[0] + mots[1].data.shape[0] + mots[2].data.shape[0] == mot.data.shape[0], \
            error_message + "data lost in segmentation."
        assert mots[0].data.shape[0] == mots[0].header_lines['nRows'] == rand1 + 1 - ff
        assert mots[1].data.shape[0] == mots[1].header_lines['nRows'] == rand2 - rand1
        assert mots[2].data.shape[0] == mots[2].header_lines['nRows'] == lf - rand2, (
                error_message + "segmentation at wrong frames.")
        assert mot != mots[0] and mot != mots[1] and mot != mots[2], \
            error_message + "original MOT object should not equal to segmented objects."
        assert mots == mot.segment([rand1, rand2]), \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_save() -> None:
        mot1 = MOT.load_from_mot(os.path.join(path, filename_standard))
        try:
            mot1.save(output)
            assert True
        except OSError:
            assert False, "File not written."
        try:
            mot2 = MOT.load_from_mot(os.path.join(output, filename_standard))
            assert True
        except OSError:
            assert False, "Written file could not be read."
        assert mot1 == mot2, \
            "Write method is not working."


if __name__ == "__main__":
    logging.basicConfig(filename='test.log', level=logging.INFO)
    _Test.main()
