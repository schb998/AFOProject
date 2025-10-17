import os
from copy import deepcopy
import pandas as pd
import numpy as np
import ast
import random

# todo: further testing with nested load/write & segmentation

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

    def __init__(self, name, filename, header_lines, data, first_frame=0):
        self.name = name
        self.filename = filename
        self.header_lines = header_lines
        self.data = data
        self.col_names = data.columns.to_list()
        self.first_frame = first_frame

    def __eq__(self, other):
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

    def __ne__(self, other):
        """Overrides the default implementation of inequality operation.

        MOT objects are compared on data content. Name and filename attributes are considered.

        Args:
            other: object to compare

        Returns:
            bool
        """
        return not self.__eq__(other)

    @classmethod
    def load(cls, filepath, filename=None):
        """Reads data from a MOT file.

        Args:
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
            raise OSError(error_message + " given path does not lead to a MOT file.")

        # read the file:
        try:
            with open(filepath, 'r') as file:
                name = next(file).strip("\n").strip('.mot')
                header_lines = {}
                line = next(file).strip("\n")
                while line != "endheader":
                    temp = line.split('=')
                    md = temp[1].strip()
                    try:
                        header_lines[temp[0].strip()] = ast.literal_eval(md)
                    except ValueError:
                        header_lines[temp[0].strip()] = md
                    line = next(file).strip("\n")
                data = pd.read_csv(file, sep=r'\s', engine='python')
                file.close()
                return cls(name, filename, header_lines, data)
        except Exception as e:
            raise OSError(error_message + str(e))

    def rename(self, name=None, filename=None):
        """This method updates the MOT object's name and/or file_name.

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

    def save(self, file_path, file_name=None):
        """This method writes the MOT object into a MOT file.

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
            raise OSError(f"Unable to write file {file_name}: {e}")

    def copy(self):
        """Copies and returns a new MOT object.

        "_copy" has been added to the returned MOT object's filename and name.

        Returns:
            MOT: Copied MOT object.
        """
        copy = deepcopy(self)
        copy.filename = copy.filename.replace(".mot", "_copy.mot")
        copy.name += '_copy'
        return copy

    def sample(self, first_frame, last_frame):
        """Samples the current MOT file between the given points.

        Args:
            first_frame (int): index of the first frame.
            last_frame  (int): index of the last frame.

        Returns:
            MOT: sampled MOT object.

        Raises:
            IndexError: if the given points are out of bound for the data.
        """
        frames = sorted((first_frame, last_frame))
        first_frame = frames[0]
        last_frame = frames[1]

        if (first_frame < 0) or (last_frame > self.data.shape[0]):
            raise IndexError("Cannot cut at given frames: out of bound index.")

        headers = deepcopy(self.header_lines)
        headers['nRows'] = last_frame - first_frame
        name = self.name + "_segmented_" + str(first_frame) + "-" + str(last_frame - 1)
        file_name = name + ".mot"
        d = {}
        for col in self.data.columns.to_list():
            d[col] = self.data[col][first_frame:last_frame]
        return MOT(name, file_name, headers, pd.DataFrame(data=d), first_frame)

    def segment(self, points):
        """Segments the current MOT file.

        Does so at the given points, returning a list of segmented MOT objects.

        Args:
            points (list of int): list of the frames before which the file needs to be segmented.

        Returns:
            list: list of MOT objects.

        Raises:
            IndexError: if the given points are out of bound for the data.
        """
        # sort the frames at which to segment the object:
        points = sorted(points)
        if (points[0] < 0) or (points[-1] > self.data.shape[0]):
            raise IndexError("Cannot cut at given frames: out of bound index.")
        points.append(self.data.shape[0])
        points.insert(0, 0)

        resulting_mots = []
        headers = deepcopy(self.header_lines)

        # segment the file:
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            name = self.name + "_segmented_" + str(start) + "-" + str(end - 1)
            file_name = name + ".mot"
            d = {}
            for col in self.data.columns.to_list():
                d[col] = self.data[col][start:end]
            headers['nRows'] = end - start
            resulting_mots.append(MOT(name, file_name, deepcopy(headers), pd.DataFrame(data=d), start))

        # return:
        return resulting_mots

    @classmethod
    def load_multiple(cls, data_path_mot):
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
            motion_data_list.append(cls.load(file_path))
        return motion_data_list

    @classmethod
    def save_multiple(cls, mots, directory_path):
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
                raise OSError(f"Object {mot.name} couldn't be saved.")



class MOTCleanup:
    @staticmethod
    def delete_mot_file(path_to_mot):
        """Deletes MOT file from given path.

        Args:
            path_to_mot (string): path to the MOT file to be deleted.

        Raises:
            OSError: if a file could not be deleted.
        """
        if not os.path.basename(path_to_mot).endswith('.mot'):
            raise OSError(f"Could not delete {path_to_mot}: invalid path.")
        print(f"Confirm deletion of file {path_to_mot} (y/[n]):\n")
        confirmation = input().lower().strip()
        if confirmation == 'y' or confirmation == 'yes':
            try:
                os.remove(path_to_mot)
            except OSError:
                raise OSError(f"Could not delete {path_to_mot}")
            print(f"File {path_to_mot} has been deleted.")
        else:
            print(f"File {path_to_mot} has not been deleted.")

    @staticmethod
    def delete_all_files(path_to_directory):
        """Deletes all MOT files from given path.

        Args:
            path_to_directory (string): path to the directory where all MOT files are to be deleted.
        """
        if not os.path.isdir(path_to_directory):
            raise OSError(f"Could not delete files from {path_to_directory}: path is not a directory.")

        file_list = sorted(f for f in os.listdir(path_to_directory) if f.endswith('.mot'))
        print(f"This directory contains: " + str(file_list))
        print(f"Confirm deletion of all MOT files from {path_to_directory} (y/[n]):")
        confirmation = input().lower().strip()

        if confirmation == 'y' or confirmation == 'yes':
            for file in file_list:
                try:
                    os.remove(os.path.join(path_to_directory, file))
                except OSError:
                    raise OSError(f"Could not delete {file}")
                print(f"File {file} has been deleted.")
        else:
            print(f"Files in {path_to_directory} have not been deleted.")


class Test:
    """Regression tests for the MOT class methods.

    Those tests are used to ensure working methods are not compromised by new code.
    """

    @staticmethod
    def main():
        Test._test_load()
        Test._test_equality()
        Test._test_copy()
        Test._test_sample()
        Test._test_segmentation()
        Test._test_save()
        print("All tests passed, deleting testing files...")
        MOTCleanup.delete_all_files(output)

    @staticmethod
    def _test_load():
        try:
            MOT.load(os.path.join(path, filename_standard))
            MOT.load(path, filename_nan)
            assert True
        except Exception:
            assert False, \
                "File not read."
        assert MOT.load(os.path.join(path, filename_standard)) == MOT.load(os.path.join(path, filename_standard)), \
            "MOT Object from same file should be equal."
        assert MOT.load(os.path.join(path, filename_standard)) != MOT.load(os.path.join(path, filename_nan)), \
            "MOT Object from different files should be not equal."

    @staticmethod
    def _test_nestled_loads():
        try:
            mot = MOT.load(os.path.join(path, filename_nan))
            mot.save(output, "first_save.mot")
            mot_first_save = MOT.load(os.path.join(output, "first_save.mot"))
            mot_first_save.save(output, "second_save.mot")
            mot_second_save = MOT.load(os.path.join(output, "second_save.mot"))
            assert True
        except Exception:
            assert False, "Couldn't load and save files in a loop."
        assert mot == mot_first_save == mot_second_save, "Nestled loaded files should be equal."

    @staticmethod
    def _test_equality():
        mot = MOT.load(os.path.join(path, filename_standard))
        assert mot == mot, \
            "Equality operation is not working."
        assert mot != MOT.load(os.path.join(path, filename_nan)), \
            "Inequality operation is not working."

    @staticmethod
    def _test_copy():
        mot = MOT.load(os.path.join(path, filename_standard))
        assert mot.copy() == mot, \
            "Copy method is not working."

    @staticmethod
    def _test_sample():
        mot = MOT.load(os.path.join(path, filename_standard))
        length = mot.data.shape[0]
        rands = sorted((random.randint(0, length - 1), random.randint(0, length - 1)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Sampling method is not working with values {rand1, rand2}: "
        sample = mot.sample(rand1, rand2)
        assert sample.data.shape[1] == mot.data.shape[1], \
            error_message + "wrong number of columns."
        assert sample.data.shape[0] == rand2 - rand1 \
               and mot.data.shape[0] == sample.data.shape[0] + rand1 + (length - rand2), \
            error_message + "sampling at wrong frames."
        assert mot != sample, \
            error_message + "original MOT object should not equal sampled objects."
        sample2 = mot.sample(rand1, rand2)
        assert sample == sample2, \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_segmentation():
        mot = MOT.load(os.path.join(path, filename_standard))
        length = mot.data.shape[0]
        rands = sorted((random.randint(0, length - 1), random.randint(0, length - 1)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Segmentation method is not working with values {rand1, rand2}: "
        mots = mot.segment(rands)
        assert len(mots) == 3, \
            error_message + "wrong number of segments."
        assert mots[0].data.shape[1] == mot.data.shape[1] \
               and mots[1].data.shape[1] == mot.data.shape[1] \
               and mots[2].data.shape[1] == mot.data.shape[1], \
            error_message + "wrong number of columns."
        assert mots[0].data.shape[0] + mots[1].data.shape[0] + mots[2].data.shape[0] == mot.data.shape[0], \
            error_message + "data lost in segmentation."
        assert mots[0].data.shape[0] == mots[0].header_lines['nRows'] == rand1 \
               and mots[1].data.shape[0] == mots[1].header_lines['nRows'] == rand2 - rand1 \
               and mots[2].data.shape[0] == mots[2].header_lines['nRows'] == length - rand2, \
            error_message + "segmentation at wrong frames."
        assert mot != mots[0] and mot != mots[1] and mot != mots[2], \
            error_message + "original MOT object should not equal to segmented objects."
        assert mots == mot.segment([rand1, rand2]), \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_save():
        mot1 = MOT.load(os.path.join(path, filename_standard))
        try:
            mot1.save(output)
            assert True
        except Exception:
            assert False, "File not written."
        try:
            mot2 = MOT.load(os.path.join(output, filename_standard))
            assert True
        except Exception:
            assert False, "Written file could not be read."
        assert mot1 == mot2, \
            "Write method is not working."


if __name__ == "__main__":
    Test.main()
    print("All done.")