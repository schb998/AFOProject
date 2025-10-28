import os
from copy import deepcopy
import pandas as pd
import numpy as np
import ast
import random
import time

# todo: further testing for segmentation methods comparison
# todo : update save method to make directory when needed

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testing_files")
output = os.path.join(path, "test_output")

# working files:
filename_standard = "TRC_standard.trc"
filename_nan = "TRC_nan.trc"  # missing values should be handled
# error management files:
filename_missing_z7 = "TRC_missing_z7.trc"  # error : missing marker coordinate z7


class TRC:
    """TRC object.

    Attributes:
        filename:    String indicating the name of the originating file.
        metadata:    Dictionary with the TRC metadata.
        marker_set:  List of the markers used.
        col_names:   List of the names of the data columns.
        marker_dict: Dictionary of the columns associated with each marker.
        data:        Dataframe containing the data.
        first_frame: Integer, first frame of the data. Default value = 0.
        file_header: List of string, content of the TRC file's header line. Optional.
    """

    def __init__(self, filename, meta_data, marker_set, col_names, marker_dict, data, file_header=None):
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

    def __eq__(self, other):
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

    def __ne__(self, other):
        """Overrides the default implementation of inequality operation.

        TRC objects are compared on data content. Filename and file_header attributes are not considered.

        Args:
            other: object to compare

        Returns:
            bool
        """
        return not self.__eq__(other)

    @classmethod
    def load(cls, filepath, filename=None, header=True, delimiter="\t"):
        """Reads data from a TRC file.

        Args:
            filepath  (string): path to the TRC file.
            filename  (string): name of the TRC file. \
                Should be filled if path does not include filename, optional otherwise.
            header   (boolean): whether the TRC file includes a header. Default value is True.
            delimiter (string): delimiter of the TRC file. Default value is "\t".

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

        error_message = f"File {filename} at {filepath} couldn't be read: "

        # test that given path is valid :
        if (not os.path.isfile(filepath)) or (not filepath.endswith(".trc")):
            raise OSError(error_message + " given path does not lead to a TRC file.")

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

                return cls(filename, meta_data, marker_set, sub_headers, marker_dictionary, data,
                           file_header if header else None)
        except Exception as e:
            raise OSError(error_message + getattr(e, 'message', repr(e)))


    def save(self, filepath, filename=None):
        """Saves data into a TRC file.

        Args:
            filepath (string): path to the directory in which to save the TRC file.
            filename (string): name of the save file. Optional. If not filled, attribute filename will be used.

        Raises:
            OSError: if the file cannot be saved.
        """

        error_message = f"File {self.filename} couldn't be saved in {filepath}: "

        # check if valid path:
        try:
            os.makedirs(filepath, exist_ok=True)
        except Exception as e:
            raise OSError(error_message + getattr(e, 'message', repr(e)))

        if filename is None:
            filename = self.filename

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
        print(f"File {filename} written in directory {path}.")


    def copy(self):
        """ Returns a copy of the object.

        Returns:
            TRC object
        """
        copy = deepcopy(self)
        copy.filename = copy.filename.replace(".trc", "_copy.trc")
        return copy


    def sample(self, first_frame, last_frame):
        """Samples the current TRC file between the given points.

        Args:
            first_frame (int): index of the first frame.
            last_frame  (int): index of the last frame.

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


    def segment(self, points):
        """ Segments the data frame according to the given frames point.
         Each fragment will be in the form of [points[i-1]; points[i][ with added segment first segment \
         [first_frame_of_data; points[0][ and last segment [points[-1]; last_frame_of_data]

        Returns:
            List of the segmented TRC objects.
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
            end = points[i + 1] if i+1 != len(points) else points[i + 1] + 1
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


    def segment_bis(self, points):
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
    def load_all(cls, dir_path):
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
            except Exception as e:
                print(f"Could not load file: {file} because of {getattr(e, 'message', repr(e))}. Skipping.")
                pass
        return resulting_trcs


    @classmethod
    def save_multiple(cls, trcs, dir_path):
        """Recursively writes TRC object into files.

        Args:
            trcs       (list): list of MOT objects.
            dir_path (string): output directory.

        Raises:
            OSError: if a file could not be written.
        """
        os.makedirs(dir_path, exist_ok=True)
        for trc in trcs:
            try:
                trc.save(dir_path)
            except OSError:
                raise OSError(f"Object {trc.filename} couldn't be saved.")


class TRCCleanup:
    @staticmethod
    def delete_trc_file(path_to_trc):
        """Deletes TRC file from given path.

        Args:
            path_to_trc (string): path to the TR file to be deleted.

        Raises:
            OSError: if a file could not be deleted.
        """
        if not os.path.basename(path_to_trc).endswith('.trc'):
            raise OSError(f"Could not delete {path_to_trc}: invalid path.")
        print(f"Confirm deletion of file {path_to_trc} (y/N):\n")
        confirmation = input().lower().strip()
        if confirmation == 'y' or confirmation == 'yes':
            try:
                os.remove(path_to_trc)
            except OSError:
                raise OSError(f"Could not delete {path_to_trc}")
            print(f"File {path_to_trc} has been deleted.")
        else:
            print(f"File {path_to_trc} has not been deleted.")


    @staticmethod
    def delete_all_files(path_to_directory):
        """Deletes all TRC files from given path.

        Args:
            path_to_directory (string): path to the directory where all TRC files are to be deleted.
        """
        if not os.path.isdir(path_to_directory):
            raise OSError(f"Could not delete files from {path_to_directory}: path is not a directory.")

        file_list = sorted(f for f in os.listdir(path_to_directory) if f.endswith('.trc'))
        print(f"This directory contains: " + str(file_list))
        print(f"Confirm deletion of all TRC files from {path_to_directory} (y/N):")
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
        Test._test_segmentation_bis()
        Test._test_save()
        Test._test_load_all()
        Test._test_save_all()
        print("All tests passed. Deleting testing files...")
        TRCCleanup.delete_all_files(output)


    @staticmethod
    def _test_load():
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
    def _test_equality():
        trc = TRC.load(path, filename_standard)
        assert trc == trc, \
            "Equality operation is not working."
        assert trc != TRC.load(path, filename_nan), \
            "Inequality operation is not working."
        assert trc == TRC.load(path, filename_standard), \
            "Objects loaded from same file should be equal."


    @staticmethod
    def _test_copy():
        trc = TRC.load(path, filename_standard)
        assert trc.copy() == trc, \
            "Objects should be equal to copy."


    @staticmethod
    def _test_sample():
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
    def _test_segmentation():
        trc = TRC.load(os.path.join(path, filename_standard))
        ff = trc.first_frame
        length = trc.data.shape[0]
        rands = sorted((random.randint(ff, length+ff), random.randint(ff, length+ff)))
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
    def _test_segmentation_bis():
        trc = TRC.load(os.path.join(path, filename_standard))
        ff = trc.first_frame
        length = trc.data.shape[0]
        rands = sorted((random.randint(ff, length+ff), random.randint(ff, length+ff)))
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
    def _test_load_all():
        trcs = TRC.load_all(path)
        for f in trcs:
            assert f == TRC.load(
                os.path.join(path, f.filename)), "Mass loaded objects should match object loaded from the same file."


    @staticmethod
    def _test_save():
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
    def _test_save_all():
        TRCCleanup.delete_all_files(output)
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
        for i in range(len(trcs)):
            assert trcs[i] == trcs_copied[i], (error_message + f"File {trcs[i].filename} should be equal to its saved "
                                                               f"and loaded version.")


    @staticmethod
    def comparison_segmentation(file):
        """This method is used to compare use of the two coded segmentation methods.

        This method does 100 tests with each segmenting method on the same file, \
            using a randomly generated number (1-10) of randomly generated values
            (in range of the object's data's length)
            to segment the file.

        At the moment, the segment_bis method seems to be faster, but further testing is required
            to observe impact of file size, number of segments, size of segments.

        Args:
            file (string): file to test, located in the testing folder.

        Returns:
            dataframe:  test results, with columns:
                - list of the values used to segment the file
                - duration of the segment method for those values
                - duration of the segment-bis method for those values
                - difference (duration segment - duration segment_bis)
            float:      mean value of the difference
        """
        trc = TRC.load(path, file)
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
    # trc = TRC.load("C:\\Users\\lgre690\\Documents\\MyData\\osim_tests\\static_01.trc")
    Test.main()
    print('All done.')
