import os
from copy import deepcopy
import pandas as pd
import numpy as np
import numpy.typing
import ast
import random
import time
from typing import Self
import logging
import re

# todo: further testing for segmentation methods comparison
# todo: make get_first_frame method instead of first_frame attribute ?

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing_files")
output = os.path.join(path, "test_output")

# working files:
filename_standard = "TRC_standard.trc"
filename_nan = "TRC_nan.trc"  # missing values should be handled
# error management files:
filename_missing_z7 = "TRC_missing_z7.trc"  # error : missing marker coordinate z7


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
        file_header: List of string, content of the TRC file's header line. Optional.
    """

    def __init__(self,
                 filename: str,
                 meta_data: dict[str, str | int | float],
                 marker_set: list[str],
                 col_names: list[str],
                 marker_dict: dict[str, list[str]],
                 data: pd.DataFrame,
                 file_header: list[str] = None) \
            -> None:
        self.filename = filename
        self.metadata = meta_data
        self.marker_set = marker_set
        self.col_names = col_names
        self.marker_dict = marker_dict
        self.data = data
        self.first_frame = data.index[0]
        if file_header is not None:
            self.file_header = file_header
        else:
            self.file_header = []


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
        if isinstance(other, TRC):
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
    def load(cls, filepath: str, filename: str = None, header: bool = True, delimiter: str = "\t") -> Self:
        """Reads data from a TRC file.

        Args:
            filepath: path to the TRC file.
            filename:  name of the TRC file. \
                Should be filled if path does not include filename, optional otherwise.
            header: whether the TRC file includes a header. Default value is True.
            delimiter: delimiter of the TRC file. Default value is "\t".

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

                # data headers:
                headers = next(file).strip().split(delimiter)
                headers = [headers[i] for i in range(0, len(headers)) if len(headers[i]) > 0]
                sub_headers = next(file).strip().split(delimiter)
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
                    marker_dictionary[marker_set[m]] = sub_headers[i:i + 3]
                    i += 3

                res = cls(filename, meta_data, marker_set, sub_headers, marker_dictionary, data,
                          file_header if header else None)
                logging.info(f'TRC object successfully loaded from file {filepath}.')
                return res
        except Exception as e:
            error_message = error_message + getattr(e, 'message', repr(e))
            logging.warning(error_message)
            raise OSError(error_message)


    @classmethod
    def arrange(cls, filepath: str, filename: str = None, header: bool = True, delimiter: str = "\t") -> None:
        """Overwrites a TRC file with a copy of data with added marker ZERO located at position (0,0,0) at all frames.

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
        trc = cls.load(filepath, filename, header, delimiter)
        old_name = deepcopy(trc.filename)
        num_frames = trc.data.shape[0]
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


    def add_marker(self, marker_name: str, data: dict[str, numpy.typing.ArrayLike]) -> None:
        """Add a marker to the data.

        Args:
            marker_name: name of the marker
            data: dictionary of the marker's position data. Raises exception if size is not 3. Keys will be kept unless/
             they do not match typical naming patterns: each coordinate starting/ending with X/Y/Z or x/y/z.

        Returns:
            None

        Raises:
            Exception if given data does not contain exactly 3 coordinates.
        """
        if len(data) != 3:
            raise Exception("Markers require three coordinates in order to be added.")

        # manage marker name
        name = marker_name
        i = 2
        while name in self.marker_set:
            logging.info(f"Marker {name} already exists, changing name to {marker_name + str(i)}")
            name = marker_name + str(i)
        marker_name = name
        self.marker_set.append(marker_name)
        self.metadata['NumMarkers'] = self.metadata['NumMarkers'] + 1

        # manage marker coordinates:
        content = deepcopy(data)
        columns = list(content.keys())
        try:
            x_column = [x for x in columns if re.search("^([Xx])|([Xx])$", x) is not None][0]
            y_column = [y for y in columns if re.search("^([Yy])|([Yy])$", y) is not None][0]
            z_column = [z for z in columns if re.search("^([Zz])|([Zz])$", z) is not None][0]
            result = {x_column.upper(): content[x_column],
                      y_column.upper(): content[y_column],
                      z_column.upper(): content[z_column]}
        except KeyError:
            num = str(len(self.marker_set) + 1)
            x_column = 'X'+num
            y_column = 'Y'+num
            z_column = 'Z'+num
            logging.info(f"Columns names do not match expected X/Y/Z formulation. Assigning them coordinates X, Y, Z "
                         f"and names {x_column}, {y_column}, {z_column}.")
            result = {x_column: content[columns[0]],
                      y_column: content[columns[1]],
                      z_column: content[columns[2]]}
        self.marker_dict[marker_name] = [x_column, y_column, z_column]
        self.col_names.extend([x_column, y_column, z_column])
        for coo in list(result.keys()):
            self.data[coo] = result[coo]
        self.rename(self.filename.replace('.trc', f'added_{marker_name}'))


    def save(self, filepath: str, filename: str = None) -> None:
        """Saves data into a TRC file.

        Args:
            filepath (string): path to the directory in which to save the TRC file.
            filename (string): name of the save file. Optional. If not filled, attribute filename will be used.

        Raises:
            OSError: if the file cannot be saved.
        """
        if filename is None:
            filename = self.filename

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
            content.append(line + "\n")

        c0 = ""
        c1 = ""

        for md in self.metadata.keys():
            c0 += f"{md}\t"
            c1 += f"{str(self.metadata[md])}\t"
        content.append(c0.strip() + "\n")
        content.append(c1.strip() + "\n")
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

        with open(os.path.join(filepath, filename), 'w') as writer:
            writer.writelines(content)
        logging.info(f"File {filename} saved in directory {filepath}.")


    def copy(self) -> Self:
        """ Returns a copy of the object.

        Returns:
            TRC object
        """
        copy = deepcopy(self)
        copy.filename = copy.filename.replace(".trc", "_copy.trc")
        return copy


    def sample(self, first_frame: int, last_frame: int) -> Self:
        """Samples the current TRC file between the given points.

        Args:
            first_frame (int): index of the first frame, included.
            last_frame  (int): index of the last frame, excluded.

        Returns:
            TRC: A new TRC object.

        Raises:
            IndexError: if the given points are out of bound for the data.
        """
        frames = sorted((first_frame, last_frame))
        first_frame = frames[0]
        last_frame = frames[1]
        ff = self.first_frame

        if (first_frame < self.first_frame) or (last_frame > self.first_frame + self.data.shape[0]):
            raise IndexError("Cannot cut at given frames: out of bound index.")

        file_name = self.filename.replace('.trc', "_segmented_" + str(first_frame) + "-" + str(last_frame - 1) + '.trc')
        metadata = deepcopy(self.metadata)
        metadata['NumFrames'] = last_frame - first_frame

        d = {}
        for col in self.data.columns.to_list():
            d[col] = self.data[col][first_frame - ff:last_frame - ff]
        return TRC(file_name, metadata, deepcopy(self.marker_set), deepcopy(self.col_names), deepcopy(self.marker_dict),
                   pd.DataFrame(data=d), file_header=deepcopy(self.file_header))


    def segment(self, points: list[int]) -> list[Self]:
        """ Segments the data frame according to the given frames point.

         Each fragment contains frames from points[i-1] (included) to points[i] (excluded),
          with added segments: first segment from first_frame_of_data (included) to points[0] (excluded)
          and last segment from points[-1] (included) to last_frame_of_data (included).

        Parameters:
            points: list of integer, frames to segment the object at.

        Returns:
            List of the segmented TRC objects.

        Raises:
            IndexError if given points out of index.

        """
        # sort the frames at which to segment the object:
        points = sorted(points)
        ff = self.first_frame
        lf = self.first_frame + self.data.shape[0]
        if (points[0] < ff) or (points[-1] > lf):
            raise IndexError("Cannot cut at given frames: out of bound index.")
        points.append(lf)
        points.insert(0, ff)

        resulting_trcs = []

        # segment the file:
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1] if i + 1 != len(points) else points[i + 1] + 1
            metadata = deepcopy(self.metadata)
            metadata['NumFrames'] = end - start
            file_name = self.filename.replace(".trc", "_segmented_" + str(start) + "-" + str(end - 1) + ".trc")
            d = {}
            for col in self.data.columns.to_list():
                d[col] = self.data[col][start - ff:end - ff]
            resulting_trcs.append(TRC(file_name, metadata, deepcopy(self.marker_set), deepcopy(self.col_names),
                                      deepcopy(self.marker_dict), pd.DataFrame(data=d),
                                      file_header=deepcopy(self.file_header)))

        return resulting_trcs


    def segment_bis(self, points: list[int]) -> list[Self]:
        """ Segments the data frame according to the given frames point.

         Each fragment contains frames from points[i-1] (included) to points[i] (excluded),
          with added segments: first segment from first_frame_of_data (included) to points[0] (excluded)
          and last segment from points[-1] (included) to last_frame_of_data (included).

        Parameters:
            points: list of integer, frames to segment the object at.

        Returns:
            List of the segmented TRC objects.

        Raises:
            IndexError if given points out of index.

        """
        points = sorted(points)
        if (points[0] < self.first_frame) or (points[-1] > self.data.shape[0] + self.first_frame):
            raise IndexError("Cannot cut at given frames: out of bound index.")
        points.append(self.data.shape[0] + self.first_frame)
        points.insert(0, self.first_frame)
        resulting_trcs = []
        for i in range(len(points) - 1):
            resulting_trcs.append(self.sample(points[i], points[i + 1])) if i != len(points) - 1 else \
                resulting_trcs.append(self.sample(points[i], points[i + 1] + 1))
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
                resulting_trcs.append(TRC.load(os.path.join(dir_path, file)))
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


class _TRCCleanup:
    @staticmethod
    def delete_trc_file(path_to_trc: str, force_delete: bool = False) -> None:
        """Deletes TRC file from given path.

        Args:
            path_to_trc: path to the TR file to be deleted.
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
        _Test._test_sample()
        _Test._test_segmentation()
        _Test._test_segmentation_bis()
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
            TRC.load(path, filename_standard)
            TRC.load(path, filename_nan)
            TRC.load(os.path.join(path, filename_standard))
            TRC.load(os.path.join(path, filename_nan))
            assert True
        except OSError:
            assert False, "File couldn't be loaded."
        try:
            TRC.load(os.path.join(path, filename_missing_z7))
            assert False, f"File {filename_missing_z7} should raise an exception upon loading."
        except OSError:
            assert True

    @staticmethod
    def _test_nestled_loads() -> None:
        try:
            trc = TRC.load(os.path.join(path, filename_nan))
            trc.save(output, "first_save.trc")
            trc_first_save = TRC.load(os.path.join(output, "first_save.trc"))
            trc_first_save.save(output, "second_save.trc")
            trc_second_save = TRC.load(os.path.join(output, "second_save.trc"))
            assert True
        except OSError:
            assert False, "Couldn't load and save files in a loop."
        assert trc == trc_first_save == trc_second_save, "Nestled loaded files should be equal."

    @staticmethod
    def _test_operations() -> None:
        trc1 = TRC.load(path, filename_standard)
        trc2 = TRC.load(path, filename_nan)
        assert trc1 == trc1 and trc2 == trc2, \
            "Equality operation is not working."
        assert trc1 != trc2 and trc2 != trc1, \
            "Inequality operation is not working."
        assert trc1 == TRC.load(path, filename_standard) and trc2 == TRC.load(path, filename_nan), \
            "Objects loaded from same file should be equal."
        trc3 = trc1.copy()
        trc3.rename('foo')
        assert trc1 > trc3 and trc1 >= trc3 and trc3 < trc1 and trc3 <= trc1, "Comparison operations are not working."

    @staticmethod
    def _test_copy() -> None:
        trc = TRC.load(path, filename_standard)
        assert trc.copy() == trc, \
            "Objects should be equal to copy."

    @staticmethod
    def _test_sample() -> None:
        trc = TRC.load(os.path.join(path, filename_standard))
        length = trc.data.shape[0]
        rands = sorted((random.randint(0, length - 1), random.randint(0, length - 1)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Sampling method is not working with values {rand1, rand2}: "
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

    @staticmethod
    def _test_segmentation() -> None:
        trc = TRC.load(os.path.join(path, filename_standard))
        ff = trc.first_frame
        length = trc.data.shape[0]
        rands = sorted((random.randint(ff, length + ff), random.randint(ff, length + ff)))
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
        assert trcs[0].data.shape[0] == trcs[0].metadata['NumFrames'] == rand1 - ff \
               and trcs[1].data.shape[0] == trcs[1].metadata['NumFrames'] == rand2 - rand1 \
               and trcs[2].data.shape[0] == trcs[2].metadata['NumFrames'] == (length + ff) - rand2, (
                error_message + "segmentation at wrong frames.")
        assert trc != trcs[0] and trc != trcs[1] and trc != trcs[2], \
            error_message + "original TRC object should not equal to segmented objects."
        assert trcs == trc.segment([rand1, rand2]), \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_segmentation_bis() -> None:
        trc = TRC.load(os.path.join(path, filename_standard))
        ff = trc.first_frame
        length = trc.data.shape[0]
        rands = sorted((random.randint(ff, length + ff), random.randint(ff, length + ff)))
        rand1, rand2 = rands[0], rands[1]
        error_message = f"Segmentation method is not working with values {rand1, rand2}: "
        trcs = trc.segment_bis(rands)
        assert len(trcs) == 3, \
            error_message + "wrong number of segments."
        assert trcs[0].data.shape[1] == trc.data.shape[1] \
               and trcs[1].data.shape[1] == trc.data.shape[1] \
               and trcs[2].data.shape[1] == trc.data.shape[1], error_message + "wrong number of columns."
        assert trcs[0].data.shape[0] + trcs[1].data.shape[0] + trcs[2].data.shape[0] == trc.data.shape[0], \
            error_message + "data lost in segmentation."
        assert trcs[0].data.shape[0] == trcs[0].metadata['NumFrames'] == rand1 - ff \
               and trcs[1].data.shape[0] == trcs[1].metadata['NumFrames'] == rand2 - rand1 \
               and trcs[2].data.shape[0] == trcs[2].metadata['NumFrames'] == (length + ff) - rand2, (
                error_message + "segmentation at wrong frames.")
        assert trc != trcs[0] and trc != trcs[1] and trc != trcs[2], \
            error_message + "original TRC object should not equal to segmented objects."
        assert trcs == trc.segment([rand1, rand2]), \
            error_message + "calls on object with same parameters should be equal."

    @staticmethod
    def _test_load_all() -> None:
        trcs = TRC.load_all(path)
        for f in trcs:
            assert f == TRC.load(
                os.path.join(path, f.filename)), "Mass loaded objects should match object loaded from the same file."

    @staticmethod
    def _test_save() -> None:
        trc1 = TRC.load(os.path.join(path, filename_standard))
        try:
            trc1.save(output)
            assert True
        except OSError:
            assert False, "File not written."
        try:
            trc2 = TRC.load(os.path.join(output, filename_standard))
            assert True
        except OSError:
            assert False, "Written file could not be read."
        assert trc1 == trc2, \
            "Write method is not working."

    @staticmethod
    def _test_save_all() -> None:
        _TRCCleanup.delete_all_files(output, True)
        trc = TRC.load(os.path.join(path, filename_standard))
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
        trc1 = TRC.load(path, filename_standard)
        trc2 = trc1.copy()
        num_frames = trc2.data.shape[0]
        try:
            trc2.add_marker('TEST', {'X': np.zeros(num_frames),
                                     'Y': np.zeros(num_frames),
                                     'Z': np.zeros(num_frames)})
            assert True
        except Exception:
            assert False, "Adding marker method should work."
        assert trc2.metadata['NumMarkers'] == trc1.metadata['NumMarkers'] + 1 == len(trc2.marker_set), \
            "Wrong number of markers."
        assert len(trc2.data.columns) == len(trc1.data.columns) + 3, "Wrong number of columns."

    @staticmethod
    def _test_arrange() -> None:
        trc = TRC.load(path, filename_standard).copy()
        trc.rename("test")
        trc.save(output)
        try:
            TRC.arrange(output, "test.trc")
            assert True
        except Exception:
            assert False, "Arrange method should work."

    @staticmethod
    def comparison_segmentation(path_to_file: str) -> (pd.DataFrame, float):
        """This method is used to compare use of the two coded segmentation methods.

        This method does 100 tests with each segmenting method on the same file, \
            using a randomly generated number (1-10) of randomly generated values
            (in range of the object's data's length)
            to segment the file.

        At the moment, the segment_bis method seems to be faster, but further testing is required
            to observe impact of file size, number of segments, size of segments.

        Args:
            path_to_file (string): file to test, located in the testing folder.

        Returns:
            dataframe:  test results, with columns:
                - list of the values used to segment the file
                - duration of the segment method for those values
                - duration of the segment-bis method for those values
                - difference (duration segment - duration segment_bis)
            float:      mean value of the difference
        """
        trc = TRC.load(path, path_to_file)
        length = trc.data.shape[0]
        data = []
        for i in range(100):
            nb_segment = random.randint(1, 10)
            rands = []
            for j in range(nb_segment):
                rands.append(random.randint(1, length - 1))
            rands = sorted(rands)

            # first method
            t = time.time()
            trc.segment(rands)
            t1 = time.time() - t

            # second method
            t = time.time()
            trc.segment_bis(rands)
            t2 = time.time() - t

            data.append([rands, t1, t2, t1 - t2])

        data = pd.DataFrame(data)
        mean = np.mean(data[3])
        return data, mean


if __name__ == "__main__":
    logger = logging.getLogger("test")
    logging.basicConfig(filename='.test.log', level=logging.INFO)
    _Test.main()
