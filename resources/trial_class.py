import os.path

import opensim as osim

from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from typing import Self
from resources.custom_exceptions import *


# todo: update joint power documentation


class GaitCycle:
    """
    Structure regrouping data of a gait cycle in one objet.

    Attributes:
        side: str, side of the gait cycle.
        num: int, id number of the gait cycle
        grf: MOT object, ground force reaction data
        trc: TRC object, marker data
        ik: MOT object, inverse kinematic data
        exl: XML, extermnal loads data
        id: MOT object, inverse dynamic data
        jp: joint power data
    """

    def __init__(self, side: str, number: int, ground_reaction_forces: MOT = None, markers_trajectory: TRC = None,
                 inverse_kinematic: MOT = None, external_loads: osim.ExternalForce = None, inverse_dynamic: MOT = None, joint_power: MOT = None) -> None:
        """Creates a GaitCycle object.

        Args:
            side: str, side of the gait cycle.
            number: int, id number of the gait cycle
            ground_reaction_forces: MOT object, ground force reaction data
            markers_trajectory: TRC object, marker data
            inverse_kinematic: MOT object, inverse kinematic data
            external_loads: OpenSim.ExternalForce object, external forces data
            inverse_dynamic: MOT object, inverse dynamic data
            joint_power: MOT object (to check), joint power data
        """
        if side.lower() in ["right", "r"]:
            self.side = "Right"
        elif side.lower() in ["left", "l"]:
            self.side = "Left"
        else:
            raise KeyError(f"Given gait cycle cannot be added to the trial: {side} is not a valid side.")
        self.num = number
        self.grf = ground_reaction_forces
        self.trc = markers_trajectory
        self.ik = inverse_kinematic
        self.external_loads = external_loads
        self.id = inverse_dynamic
        self.jp = joint_power

    def add_grf(self, ground_reaction_forces: MOT | str, **kwargs) -> None:
        """Add the ground reaction forces data to the object.

        Args:
            ground_reaction_forces: MOT object, or path (str) to a loadable MOT file.

        Keyword Args:
            separator: str, additional argument when the MOT file has to be loaded and is coded in a particular way.

        Returns:
            None
        """
        if isinstance(ground_reaction_forces, MOT):
            self.grf = ground_reaction_forces
        else:
            if not os.path.isfile(ground_reaction_forces) or not os.path.basename(ground_reaction_forces).endswith(
                    ".mot"):
                raise WrongExtensionException("Ground reaction force Motion file", ground_reaction_forces, ".mot")
            self.grf = MOT.load_from_mot(ground_reaction_forces,
                                         separator=kwargs["separator"] if "separator" in kwargs else None)

    def add_trc(self, markers_trajectory: TRC | str, **kwargs) -> None:
        """Add the ground reaction forces data to the object.

        Args:
            markers_trajectory: TRC object, or path (str) to a loadable TRC file.

        Keyword Args:
            header: bool, additional argument when the TRC file has to be loaded and the presence of a header needs to be precised.
            delimiter: str, additional argument when the TRC file has to be loaded and is coded the data is separated in a particular way.
            num_coordinates, int: str, additional argument when the TRC file has to be loaded and contains a specific number of coordinates by marker.

        Returns:
            None
        """
        self.trc = markers_trajectory
        if isinstance(markers_trajectory, TRC):
            self.trc = markers_trajectory
        else:
            if not os.path.isfile(markers_trajectory) or not os.path.basename(markers_trajectory).endswith(".trc"):
                raise WrongExtensionException("Markers file", markers_trajectory, ".trc")
            self.trc = TRC.load_from_trc(markers_trajectory, header=kwargs["header"] if "header" in kwargs else None,
                                         delimiter=kwargs["delimiter"] if "delimiter" in kwargs else None,
                                         num_coordinates=kwargs[
                                             "num_coordinates"] if "num_coordinates" in kwargs else None)

    def add_ik(self, inverse_kinematic: MOT | str) -> None:
        """Add the inverse kinematics data to the object.

        Args:
            inverse_kinematic: MOT object, or path (str) to a loadable MOT file.

        Returns:
            None
        """
        if isinstance(inverse_kinematic, MOT):
            self.ik = inverse_kinematic
        else:
            if not os.path.isfile(inverse_kinematic) or not os.path.basename(inverse_kinematic).endswith(".mot"):
                raise WrongExtensionException("OpenSim generated Inverse Kinematics file", inverse_kinematic, ".mot")
            self.ik = MOT.load_from_mot(inverse_kinematic, separator=r"\t")

    def add_external_loads(self, external_loads: osim.ExternalLoads | str) -> None:
        if isinstance(external_loads,  osim.ExternalLoads):
            self.external_loads = external_loads
        else:
            if not os.path.isfile(external_loads) or not os.path.basename(external_loads).endswith(".xml"):
                raise WrongExtensionException("OpenSim generated External Forces file", external_loads, ".xml")
            self.external_loads = osim.ExternalLoads(external_loads)

    def add_id(self, inverse_dynamic: MOT | str) -> None:
        """Add the inverse dynamic data to the object.

        Args:
            inverse_dynamic : MOT object, or path (str) to a loadable MOT file.

        Returns:
            None
        """
        if isinstance(inverse_dynamic, MOT):
            self.id = inverse_dynamic
        else:
            if not os.path.isfile(inverse_dynamic) or not os.path.basename(inverse_dynamic).endswith(".mot"):
                raise WrongExtensionException("OpenSim generated Inverse Dynamic file", inverse_dynamic, ".mot")
            self.id = MOT.load_from_mot(inverse_dynamic, separator=r"\t")

    def add_joint_power(self, joint_power) -> None:
        """Add the joint power data to the object.

        Args:
            joint_power : joint power data.

        Returns:
            None
        """
        self.jp = joint_power

    def is_empty(self):
        """Check if the current object contains any data.

        Returns:
            bool, whether the current object is empty
        """
        return (self.grf is None
                and self.trc is None
                and self.ik is None
                and self.id is None
                and self.jp is None)

    def get_time_frame(self):
        """Returns the first and last timestamp of the data, or None if the object does not contain any data.

        Returns:
            first and last timestamp of the data. None if there is no data.
        """

        def get_start_and_end(obj: MOT | TRC) -> (float, float):
            timestring = 'time' if isinstance(obj, MOT) else 'Time'
            return obj.data[timestring][obj.first_frame], obj.data[timestring][obj.first_frame + obj.data.shape[0] - 1]

        if self.grf is not None:
            return get_start_and_end(self.grf)
        if self.trc is not None:
            return get_start_and_end(self.trc)
        if self.ik is not None:
            return get_start_and_end(self.ik)
        if self.id is not None:
            return get_start_and_end(self.id)
        return None

    def is_included(self, starting_time: float, ending_time: float) -> bool:
        """Check if the current object is included in the given time frame.

        Args:
            starting_time: starting time of the time frame
            ending_time: ending time of the time frame

        Returns:

        """
        time = self.get_time_frame()
        if time is None:
            return False
        s = time[0]
        e = time[1]
        return s > starting_time and e < ending_time

    def save(self, path: str, categorize: bool = False) -> None:
        """Save all data of the object in a "/side/number/" subdirectory of the given directory.

        Args:
            path: str, directory in which to save the files
            categorize:

        Returns:
            None

        """
        path = os.path.join(path, self.side, "cycle_" + str(self.num)) if categorize else path
        os.makedirs(path, exist_ok=True)
        for obj in [self.grf, self.trc, self.ik, self.id, self.jp]:
            if obj is not None:
                obj.save(path)

    @classmethod
    def to_gait_cycles(cls, grfs: list[MOT | str], side: str,
                       trcs: list[TRC | str] = None,
                       iks: list[MOT | str] = None,
                       ids: list[MOT | str] = None,
                       jps=None) -> list[Self]:
        length = len(grfs)
        check_trc, check_ik, check_id, check_jp = False, False, False, False

        if trcs is not None:
            if len(trcs) != length:
                raise Exception("Error matching Ground Reaction Forces (MOT) objects and Marker Positions (TRC) "
                                "objects: mismatched number of objects.")
            check_trc = True
        if iks is not None:
            if len(iks) != length:
                raise Exception("Error matching Ground Reaction Forces (MOT) objects and Inverse Kinematics (MOT) "
                                "objects: mismatched number of objects.")
            check_ik = True
        if ids is not None:
            if len(ids) != length:
                raise Exception("Error matching Ground Reaction Forces (MOT) objects and Inverse Dynamics (MOT) "
                                "objects: mismatched number of objects.")
            check_id = True
        if jps is not None:
            if len(jps) != length:
                raise Exception("Error matching Ground Reaction Forces (MOT) objects and Joint Powers (XML) objects: "
                                "mismatched number of objects.")
            check_jp = True

        cycles = []
        for i in range(length):
            cycle = GaitCycle(side, i, grfs[i])
            if check_trc:
                cycle.add_trc(trcs[i])
            if check_ik:
                cycle.add_ik(iks[i])
            if check_id:
                cycle.add_id(ids[i])
            if check_jp:
                cycle.add_joint_power(jps[i])
            cycles.append(cycle)

        return cycles

    @classmethod
    def add_to_gait_cycles(cls, cycles: list[Self], trcs: list[TRC | str] = None,
                           iks: list[MOT | str] = None,
                           ids: list[MOT | str] = None,
                           jps=None) -> None:
        length = len(cycles)
        check_trc, check_ik, check_id, check_jp = False, False, False, False

        if trcs is not None:
            if len(trcs) != length:
                raise Exception("Error matching GaitCycles objects and Marker Positions (TRC) "
                                "objects: mismatched number of objects.")
            check_trc = True
        if iks is not None:
            if len(iks) != length:
                raise Exception("Error matching GaitCycles objects and Inverse Kinematics (MOT) "
                                "objects: mismatched number of objects.")
            check_ik = True
        if ids is not None:
            if len(ids) != length:
                raise Exception("Error matching GaitCycles objects and Inverse Dynamics (MOT) "
                                "objects: mismatched number of objects.")
            check_id = True
        if jps is not None:
            if len(jps) != length:
                raise Exception("Error matching GaitCycles objects and Joint Powers (XML) objects: "
                                "mismatched number of objects.")
            check_jp = True

        for i in range(length):
            cycle = cycles[i]
            if check_trc:
                cycle.add_trc(trcs[i])
            if check_ik:
                cycle.add_ik(iks[i])
            if check_id:
                cycle.add_id(ids[i])
            if check_jp:
                cycle.add_joint_power(jps[i])


class Trial:
    """
    Structure regrouping trial in one objet.

    Attributes:
        name: str, name of the trial
        grf: MOT object, ground force forces
        trc: TRC object, marker data
        corrected_grf: MOT object, corrected grf
        notes: str, notes on the trial.
        gait_cycles: directory of GaitCycles objects, by side.
    """

    def __init__(self, mot: MOT, trc: TRC = None, name: str = None, notes: str = None) -> None:
        """Creates a new Trial object.

        Args:
            mot: MOT object, grf of the trial
            trc: TRC object, markers positions data of the trial
            name: str, name of the trial if doesn't match the name of the gfr file
            notes: str, notes on the trial. Optional.
        """
        self.name = mot.name.replace(".mot", "") if name is None else name
        self.grf = mot
        self.trc = trc
        self.corrected_grf = None
        self.notes = notes
        self.gait_cycles: dict[str, list[GaitCycle]] = {"Right": [], "Left": []}

    def add_cycles(self, right_cycles: list[GaitCycle], left_cycles: list[GaitCycle]) -> None:
        """Adding the given GaitCycles to the trial data, at the end of their respective sides.

        Args:
            right_cycles:
            left_cycles:

        Returns:

        """
        self.gait_cycles["Right"].extend(right_cycles)
        self.gait_cycles["Left"].extend(left_cycles)

    def add_cycle(self, cycle: GaitCycle) -> None:
        """Insert the given GaitCycle in the Trial gaitcycles, at the cycle's number index in the cycle's side list.

        Args:
            cycle: GaitCycle, gait cycle object to add to the Trial data

        Returns:
            None
        """
        self.gait_cycles[cycle.side].insert(cycle.num, cycle)

    def save(self, path: str) -> None:
        """Save the trial files at the given path.

        Args:
            path: str, path in which to save the trial data.

        Returns:
            None
        """
        new_path = os.path.join(path, 'trial_' + self.name)
        for file in [self.grf, self.trc, self.corrected_grf]:
            if file is not None:
                file.save(new_path)

        if self.notes is not None:
            note_file = os.path.join(new_path, 'notes.txt')
            with open(note_file, 'w') as f:
                f.write(self.notes)

        for side in ["Right", "Left"]:
            for cycle in self.gait_cycles[side]:
                cycle.save(new_path, categorize=True)

    def sample(self, starting_time: float, ending_time: float) -> Self:
        """Sample the Trial between the given time bounds.

        Args:
            starting_time: float, lower time bound
            ending_time: float, upper time bound

        Returns:
            Trial sampled from the current objesct.
        """
        time = [starting_time, ending_time]
        time.sort()
        starting_time = time[0]
        ending_time = time[1]
        cycles = {"Right": [], "Left": []}
        for side in ["Right", "Left"]:
            for cycle in self.gait_cycles[side]:
                if cycle.is_included(starting_time, ending_time):
                    cycles[side].append(cycle)
        trc = self.trc.sample(starting_time, ending_time, force_time=True)
        mot = self.grf.sample(starting_time, ending_time, force_time=True)
        trial = Trial(mot, trc, name=self.name,
                      notes=f"Trial sampled from original trial between the time {starting_time} and {ending_time}")
        trial.add_cycles(cycles["Right"], cycles["Left"])
        return trial
