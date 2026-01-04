import os.path
import opensim as osim
import pandas as pd
from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from typing import Self
from resources.custom_exceptions import *


class CyclePaths:
    """
    Structure regrouping the different paths to a GaitCycle's files.

    Attributes:
        grf: str, path to the ground reaction forces file (MOT)
        trc: str, path to the raw marker data file (TRC)
        ik_results: str, path to the Internal Dynamics file (MOT)
        external_loads: str, path to the External Loads file (XML)
        id_results: str, path to the Internal Dynamics file (MOT)
        joint_power_results: str, path to the Joint Power file (CSV)
    """

    def __init__(self):
        self.grf: str | None = None
        self.trc: str | None = None
        self.ik_results: str | None = None
        self.external_loads: str | None = None
        self.id_results: str | None = None
        self.joint_power_results: str | None = None


class GaitCycle:
    """
    Structure regrouping data of a gait cycle in one objet.

    Attributes:
        side: str, side of the gait cycle.
        num: int, id number of the gait cycle
        paths: CyclePaths object, paths of the data files
        grf: MOT object, ground force reaction data
        trc: TRC object, marker data
        ik: MOT object, inverse kinematic data
        exl: XML, external loads data
        id: MOT object, inverse dynamic data
        jp: Pd.DataFrame, joint power data
    """

    def __init__(self, side: str, number: int,
                 ground_reaction_forces: MOT | str = None,
                 markers_trajectory: TRC | str = None,
                 inverse_kinematic: MOT | str = None,
                 external_loads: osim.ExternalLoads | str = None,
                 inverse_dynamic: MOT | str = None,
                 joint_power: MOT | str = None) -> None:
        """Creates a GaitCycle object.

        Args:
            side: str, side of the gait cycle.
            number: int, id number of the gait cycle
            ground_reaction_forces: MOT object, ground force reaction data
            markers_trajectory: TRC object, marker data
            inverse_kinematic: MOT object, inverse kinematic data
            external_loads: OpenSim.ExternalLoads object, external forces data
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

        self.grf = None
        self.trc = None
        self.ik = None
        self.external_loads = None
        self.id = None
        self.jp = None
        self.paths = CyclePaths()

        if ground_reaction_forces is not None:
            self.add_grf(
                ground_reaction_forces_path=ground_reaction_forces if isinstance(ground_reaction_forces, str) else None,
                grf_object=ground_reaction_forces if isinstance(ground_reaction_forces, MOT) else None)
        if markers_trajectory is not None:
            self.add_trc(markers_trajectory_path=markers_trajectory if isinstance(markers_trajectory, str) else None,
                         trc_object=markers_trajectory if isinstance(markers_trajectory, TRC) else None)
        if inverse_kinematic is not None:
            self.add_ik(inverse_kinematic=inverse_kinematic if isinstance(inverse_kinematic, str) else None,
                        ik_object=inverse_kinematic if isinstance(inverse_kinematic, MOT) else None)
        if external_loads is not None:
            self.add_external_loads(
                external_loads_path=external_loads if isinstance(external_loads, str) else None,
                exl_object=external_loads if isinstance(external_loads, osim.ExternalLoads) else None)
        if inverse_dynamic is not None:
            self.add_id(
                inverse_dynamic_path=inverse_dynamic if isinstance(inverse_dynamic, str) else None,
                id_object=inverse_dynamic if isinstance(inverse_dynamic, MOT) else None)
        if joint_power is not None:
            self.add_joint_power(
                joint_power_path=joint_power if isinstance(joint_power, str) else None,
                jp_object=joint_power if isinstance(joint_power, pd.DataFrame) else None)

        if markers_trajectory is not None:
            if isinstance(markers_trajectory, tuple):
                self.add_trc(markers_trajectory_path=markers_trajectory[0], trc_object=markers_trajectory[1])
            else:
                self.add_trc(
                    markers_trajectory_path=markers_trajectory if isinstance(markers_trajectory, str) else None,
                    trc_object=markers_trajectory if isinstance(markers_trajectory, TRC) else None)
        if inverse_kinematic is not None:
            if isinstance(inverse_kinematic, tuple):
                self.add_trc(inverse_kinematic=inverse_kinematic[0], ik_object=inverse_kinematic[1])
            else:
                self.add_ik(inverse_kinematic=inverse_kinematic if isinstance(inverse_kinematic, str) else None,
                             ik_object=inverse_kinematic if isinstance(inverse_kinematic, MOT) else None)
        if external_loads is not None:
            if isinstance(external_loads, tuple):
                self.add_trc(external_loads_path=external_loads[0], exl_object=external_loads[1])
            else:
                self.add_external_loads(
                    external_loads_path=external_loads if isinstance(external_loads, str) else None,
                    exl_object=external_loads if isinstance(external_loads, osim.ExternalLoads) else None)
        if inverse_dynamic is not None:
            if isinstance(inverse_dynamic, tuple):
                self.add_trc(inverse_dynamic_path=inverse_dynamic[0], id_object=inverse_dynamic[1])
            else:
                self.add_id(
                    inverse_dynamic_path=inverse_dynamic if isinstance(inverse_dynamic, str) else None,
                    id_object=inverse_dynamic if isinstance(inverse_dynamic, MOT) else None)
        if joint_power is not None:
            if isinstance(joint_power, tuple):
                self.add_trc(joint_power_path=joint_power[0], jp_object=joint_power[1])
            else:
                self.add_joint_power(
                    joint_power_path=joint_power if isinstance(joint_power, str) else None,
                    jp_object=joint_power if isinstance(joint_power, pd.DataFrame) else None)


    def add_grf(self, ground_reaction_forces_path: str = None, grf_object: MOT = None, **kwargs) -> None:
        """Add the ground reaction forces data to the object. If both arguments are None, does nothing.

        Args:
            ground_reaction_forces_path: str, path to the GRF mot file
            grf_object: MOT, the GRF MOT object if previously loaded

        Keyword Args:
            separator: str, additional argument when the MOT file has to be loaded and is coded in a particular way.

        Returns:
            None
        """
        if ground_reaction_forces_path is not None:
            self.paths.grf = ground_reaction_forces_path
            self.grf = MOT.load_from_mot(ground_reaction_forces_path,
                                         separator=kwargs["separator"] if "separator" in kwargs else None) \
                if grf_object is None else grf_object
        elif grf_object is not None:
            self.grf = grf_object
            self.paths.grf = None

    def add_trc(self, markers_trajectory_path: str = None, trc_object: TRC = None, **kwargs) -> None:
        """Add the ground reaction forces data to the object.

        Args:
            markers_trajectory_path: str, path to the TRC file
            trc_object: TRC, the TRC object if previously loaded

        Keyword Args:
            header: bool, additional argument when the TRC file has to be loaded and the presence of a header needs to be precised.
            delimiter: str, additional argument when the TRC file has to be loaded and is coded the data is separated in a particular way.
            num_coordinates, int: str, additional argument when the TRC file has to be loaded and contains a specific number of coordinates by marker.

        Returns:
            None
        """
        if markers_trajectory_path is not None:
            self.paths.trc = markers_trajectory_path
            self.trc = (
                TRC.load_from_trc(markers_trajectory_path,
                                  header=kwargs["header"] if "header" in kwargs else None,
                                  delimiter=kwargs["delimiter"] if "delimiter" in kwargs else None,
                                  num_coordinates=kwargs["num_coordinates"] if "num_coordinates" in kwargs else None)) \
                if trc_object is None else trc_object
        elif trc_object is not None:
            self.trc = trc_object
            self.paths.trc = None

    def add_ik(self, inverse_kinematic: str = None, ik_object: MOT = None) -> None:
        """Add the inverse kinematics data to the object.

        Args:
            inverse_kinematic: MOT object, or path (str) to a loadable MOT file.
            ik_object: MOT object, Inverse Kinematics object if previously loaded

        Returns:
            None
        """
        if inverse_kinematic is not None:
            self.paths.ik_results = inverse_kinematic
            self.ik = MOT.load_from_mot(inverse_kinematic, separator=r"\t") if ik_object is None else ik_object
        elif ik_object is not None:
            self.ik = ik_object
            self.paths.ik_results = None

    def add_external_loads(self, external_loads_path: str = None, exl_object: osim.ExternalLoads = None) -> None:
        """Add the external loads data to the object.

        Args:
            external_loads_path: str, path to the External Load file
            exl_object: osim.ExternalLoads, external loads if previously loaded

        Returns:
            None
        """
        if external_loads_path is not None:
            self.paths.external_loads = external_loads_path
            self.external_loads = osim.ExternalLoads(external_loads_path) if exl_object is None else exl_object
        elif exl_object is not None:
            self.external_loads = exl_object
            self.paths.external_loads = None

    def add_id(self, inverse_dynamic_path: str = None, id_object: MOT = None) -> None:
        """Add the inverse dynamic data to the object.

        Args:
            inverse_dynamic_path: str, path to the Inverse Dynamic file
            id_object: MOT, Inverse Dynamic object if previously loaded

        Returns:
            None
        """
        if inverse_dynamic_path is not None:
            self.paths.id_results = inverse_dynamic_path
            self.id = MOT.load_from_mot(inverse_dynamic_path) if id_object is None else id_object
        elif id_object is not None:
            self.id = id_object
            self.paths.id_results = None

    def add_joint_power(self, joint_power_path: str = None, jp_object: pd.DataFrame = None) -> None:
        """Add the joint power data to the object.

        Args:
            joint_power_path : str, path to the joint power data.
            jp_object: pd.DataFrame, joint power data if already loaded

        Returns:
            None
        """
        if joint_power_path is not None:
            self.paths.joint_power_results = joint_power_path
            self.jp = MOT.load_from_mot(joint_power_path) if jp_object is None else jp_object
        elif jp_object is not None:
            self.jp = jp_object
            self.paths.joint_power_results = None

    def add_to_cycle(self, grfs: MOT | None = None, grf_path: str | None = None,
                     trcs: TRC | None = None, trc_path: str | None = None,
                     iks: MOT | None = None, ik_path: str | None = None,
                     exls: osim.ExternalLoads | None = None, exl_path: str | None = None,
                     ids: MOT | None = None, id_path: str | None = None,
                     jps: pd.DataFrame | None = None, jp_path: str | None = None):
            self.add_grf(ground_reaction_forces_path=grf_path, grf_object=grfs)
            self.add_trc(markers_trajectory_path=trc_path, trc_object=trcs)
            self.add_ik(inverse_kinematic=ik_path, ik_object=iks)
            self.add_external_loads(external_loads_path=exl_path, exl_object=exls)
            self.add_id(inverse_dynamic_path=id_path, id_object=ids)
            self.add_joint_power(joint_power_path=jp_path, jp_object=jps)

    def is_empty(self):
        """Check if the current object contains any data.

        Returns:
            bool, whether the current object is empty
        """
        return (self.grf is None
                and self.trc is None
                and self.ik is None
                and self.external_loads is None
                and self.id is None
                and self.jp is None)

    def get_time_frame(self):
        """Returns the first and last timestamp of the data, or None if the object does not contain any data.

        Returns:
            first and last timestamp of the data. None if there is no data.
        """

        def get_start_and_end(obj: MOT | TRC):
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
        if self.jp is not None:
            return self.jp['time'].iloc[0], self.jp['time'].iloc[-1]
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
        for obj in [self.grf, self.trc, self.ik, self.id]:
            if obj is not None:
                obj.save(path)
        if self.jp is not None:
            self.jp.to_csv(os.path.join(path, "joint_power.csv"), index=False)


    @staticmethod
    def _objects_and_paths(grfs: list[MOT] = None, grf_path: str | list[str] = None,
                           trcs: list[TRC] = None, trc_path: str | list[str] = None,
                           iks: list[MOT] = None, ik_path: str | list[str] = None,
                           exls: list[osim.ExternalLoads] = None, exl_path: str | list[str] = None,
                           ids: list[MOT] = None, id_path: str | list[str] = None,
                           jps: list[pd.DataFrame] = None, jp_path: str | list[str] = None):
        """Organize the objects and the paths to the matching files.

        Args:
            grfs: list of MOT objects, MOT objects of the ground force reactions data if already loaded
            grf_path: path to the ground force reaction data files, can be a directory of a list of files
            trcs: list of TRC objects, if already loaded
            trc_path: path to the trc data files, can be a directory of a list of files
            iks: list of MOT objects, MOT objects of the inverse kinematic data if already loaded
            ik_path: path to the inverse kinematic data files, can be a directory of a list of files
            exls: list of OpenSim ExternalLoads objects, if already loaded
            exl_path: path to the external loads data files, can be a directory of a list of files
            ids: list of MOT objects, MOT objects of the inverse dynamic data if already loaded
            id_path: path to the inverse dynamic data files, can be a directory of a list of files
            jps: list of panda DataFrame objects, if already loaded
            jp_path: path to the join power data files, can be a directory of a list of files

        Returns:
            given parameters, updated, in the order listed above

        """

        def management_call(objects: list | None, path: None | str | list[str]):
            """Manage the paths and input whether objects should be loaded

            Args:
                objects: list of the loaded objects, can be None
                path: path(s) to the objects, can be None

            Returns:
                list of string: listed paths
                bool: whether objects should be loaded from the paths

            """
            if objects is not None:
                if path is not None and isinstance(path, str):
                    temp = []
                    for grf in objects:
                        file = os.path.join(path, grf.filename)
                        temp.append(file) if os.path.isfile(file) else temp.append(None)
                    path = temp
            else:
                if path is not None:
                    if isinstance(path, str):
                        path = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".mot")]
            return path, objects is None and path is not None

        grf_path, load = management_call(grfs, grf_path)
        if load:
            grfs = [MOT.load_from_mot(file) for file in grf_path]

        trc_path, load = management_call(trcs, trc_path)
        if load:
            trcs = [TRC.load_from_trc(file) for file in trc_path]

        ik_path, load = management_call(iks, ik_path)
        if load:
            iks = [MOT.load_from_mot(file) for file in ik_path]

        exl_path, load = management_call(exls, exl_path)
        if load:
            exls = [osim.ExternalLoads(file) for file in exl_path]

        id_path, load = management_call(ids, id_path)
        if load:
            ids = [MOT.load_from_mot(file) for file in id_path]

        jp_path, load = management_call(jps, jp_path)
        if load:
            jps = [pd.read_csv(file) for file in jp_path]

        return grfs, grf_path, trcs, trc_path, iks, ik_path, exls, exl_path, ids, id_path, jps, jp_path

    @staticmethod
    def _are_same_length(*args: list | None):
        """Verifies that if existing, all list of objects passed in arguments are of the same length.

        Args:
            *args: list of object whose sizes are to be compared

        Returns:
            bool: whether the length of the objects match
            int: value of the matching length if matching
        """
        l = len(args[0]) if args[0] is not None else -1
        for a in args[1:]:
            if a is not None:
                if l == -1:
                    l = len(a)
                elif len(a) != l:
                    return False, None
        return True, l

    @classmethod
    def to_gait_cycles(cls, side: str,
                       grfs: list[MOT] = None, grf_path: str | list[str] = None,
                       trcs: list[TRC] = None, trc_path: str | list[str] = None,
                       iks: list[MOT] = None, ik_path: str | list[str] = None,
                       exls: list[osim.ExternalLoads] = None, exl_path: str | list[str] = None,
                       ids: list[MOT] = None, id_path: str | list[str] = None,
                       jps: list[pd.DataFrame] = None, jp_path: str | list[str] = None) -> list[Self]:
        """Create GaitCycles objects from the given data.

        Args:
            side: str, side of the gait cycle.
            grfs: list of MOT objects, MOT objects of the ground force reactions data if already loaded
            grf_path: path to the ground force reaction data files, can be a directory of a list of files
            trcs: list of TRC objects, if already loaded
            trc_path: path to the trc data files, can be a directory of a list of files
            iks: list of MOT objects, MOT objects of the inverse kinematic data if already loaded
            ik_path: path to the inverse kinematic data files, can be a directory of a list of files
            exls: list of OpenSim ExternalLoads objects, if already loaded
            exl_path: path to the external loads data files, can be a directory of a list of files
            ids: list of MOT objects, MOT objects of the inverse dynamic data if already loaded
            id_path: path to the inverse dynamic data files, can be a directory of a list of files
            jps: list of panda DataFrame objects, if already loaded
            jp_path: path to the join power data files, can be a directory of a list of files

        Returns:
            GaitCycles from the given data

        Raises:
            Exception if the number of objects do not match
        """

        # if there are objects to load:
        grfs, grf_path, \
            trcs, trc_path, \
            iks, ik_path, \
            exls, exl_path, \
            ids, id_path, \
            jps, jp_path = GaitCycle._objects_and_paths(grfs, grf_path,
                                              trcs, trc_path,
                                              iks, ik_path,
                                              exls, exl_path,
                                              ids, id_path,
                                              jps, jp_path)

        sl, length = cls._are_same_length(grfs, trcs, iks, exls, ids, jps)
        if not sl:
            raise Exception("GaitCycles not generated: couldn't match the objects, numbers of objects do not match.")

        cycles = []
        for i in range(length):
            cycle = GaitCycle(side, i)
            cycle.add_to_cycle(grfs[i] if grfs is not None else None,
                                                          grf_path[i] if grf_path is not None else None,
                                                          trcs[i] if trcs is not None else None,
                                                          trc_path[i] if trc_path is not None else None,
                                                          iks[i] if iks is not None else None,
                                                          ik_path[i] if ik_path is not None else None,
                                                          exls[i] if exls is not None else None,
                                                          exl_path[i] if exl_path is not None else None,
                                                          ids[i] if ids is not None else None,
                                                          id_path[i] if id_path is not None else None,
                                                          jps[i] if jps is not None else None,
                                                          jp_path[i] if jp_path is not None else None)
            cycles.append(cycle)

        return cycles

    @classmethod
    def add_to_gait_cycles(cls, cycles: list[Self],
                           grfs: list[MOT] = None, grf_path: str | list[str] = None,
                           trcs: list[TRC] = None, trc_path: str | list[str] = None,
                           iks: list[MOT] = None, ik_path: str | list[str] = None,
                           exls: list[osim.ExternalLoads] = None, exl_path: str | list[str] = None,
                           ids: list[MOT] = None, id_path: str | list[str] = None,
                           jps: list[pd.DataFrame] = None, jp_path: str | list[str] = None):
        """Add data to given GaitCycles.

        Args:
            cycles: list of GaitCycles objects to which data has to be added
            grfs: list of MOT objects, MOT objects of the ground force reactions data if already loaded
            grf_path: path to the ground force reaction data files, can be a directory of a list of files
            trcs: list of TRC objects, if already loaded
            trc_path: path to the trc data files, can be a directory of a list of files
            iks: list of MOT objects, MOT objects of the inverse kinematic data if already loaded
            ik_path: path to the inverse kinematic data files, can be a directory of a list of files
            exls: list of OpenSim ExternalLoads objects, if already loaded
            exl_path: path to the external loads data files, can be a directory of a list of files
            ids: list of MOT objects, MOT objects of the inverse dynamic data if already loaded
            id_path: path to the inverse dynamic data files, can be a directory of a list of files
            jps: list of panda DataFrame objects, if already loaded
            jp_path: path to the join power data files, can be a directory of a list of files

        Returns:
            None

        Raises:
            Exception if the number of objects do not match
        """
        # if there are objects to load:
        grfs, grf_path, \
            trcs, trc_path, \
            iks, ik_path, \
            exls, exl_path, \
            ids, id_path, \
            jps, jp_path = GaitCycle._objects_and_paths(grfs, grf_path,
                                              trcs, trc_path,
                                              iks, ik_path,
                                              exls, exl_path,
                                              ids, id_path,
                                              jps, jp_path)

        sl, length = cls._are_same_length(cycles, grfs, trcs, iks, exls, ids, jps)
        if not sl:
            raise Exception("GaitCycles not generated: couldn't match the objects, numbers of objects do not match.")

        for i in range(length):
            cycles[i].add_to_cycle(grfs[i] if grfs is not None else None, grf_path[i] if grf_path is not None else None,
                               trcs[i] if trcs is not None else None, trc_path[i] if trc_path is not None else None,
                               iks[i] if iks is not None else None, ik_path[i] if ik_path is not None else None,
                               exls[i] if exls is not None else None, exl_path[i] if exl_path is not None else None,
                               ids[i] if ids is not None else None, id_path[i] if id_path is not None else None,
                               jps[i] if jps is not None else None, jp_path[i] if jp_path is not None else None)



class TrialPaths:
    """
    Structure regrouping the different paths to a Trial's files.

    Attributes:
        grf: str, path to the ground reaction forces file (MOT)
        trc: str, path to the raw marker data file (TRC)
        corrected_grf: str, path to the corrected ground reaction forces file (MOT)
    """

    def __init__(self, grf: str, trc: str = None, corrected_grf: str = None):
        self.grf = grf
        self.trc = trc
        self.corrected_grf = corrected_grf


class Trial:
    """
    Structure regrouping trial in one objet.

    Attributes:
        name: str, name of the trial
        grf: MOT object, ground reaction forces
        trc: TRC object, marker data
        corrected_grf: MOT object, corrected grf
        notes: str, notes on the trial.
        gait_cycles: directory of GaitCycles objects, by side.
        paths: a TrialPath object storing the paths to saved files of the Trial data
    """

    def __init__(self, mot: str | MOT, trc: str | TRC = None, name: str = None, notes: str = None) -> None:
        """Creates a new Trial object.

        Args:
            mot: MOT object, grf of the trial
            trc: TRC object, markers positions data of the trial
            name: str, name of the trial if doesn't match the name of the gfr file
            notes: str, notes on the trial. Optional.
        """
        try:
            self.grf = MOT.load_from_mot(mot) if not isinstance(mot, MOT) else mot
        except OSError as e:
            raise MissingPathException("Ground Reaction Forces (MOT) file", detail=e.strerror)

        self.name = self.grf.name.replace(".mot", "") if name is None else name

        if trc is not None:
            try:
                self.trc = TRC.load_from_trc(trc) if not isinstance(trc, TRC) else trc
            except OSError as e:
                raise MissingPathException("Markers Motion (TRC) file", detail=e.strerror)

        self.corrected_grf = None
        self.notes = notes
        self.gait_cycles: dict[str, list[GaitCycle]] = {"Right": [], "Left": []}
        self.paths = TrialPaths(mot, trc=trc)

    def add_trc(self, path_to_trc: str, trc: TRC = None):
        """Add the Marker Motion data (TRC) to a trial.

        Args:
            path_to_trc: str, path to the TRC file of the Markers Motion data
            trc: TRC object, Markers Motion data of the trial if previously loaded. Can be None.

        Returns:
            None
        """
        if trc is not None:
            self.trc = trc

        if trc is None:
            try:
                self.trc = TRC.load_from_trc(path_to_trc)
            except OSError as e:
                raise MissingPathException("Corrected Ground Reaction Forces (MOT) file", detail=e.strerror)
        self.paths.trc = path_to_trc

    def add_corrected_grf(self, path_to_corrected_grf: str = None, corrected_grf: MOT = None) -> None:
        if corrected_grf is not None:
            self.corrected_grf = corrected_grf

        if path_to_corrected_grf is not None:
            if corrected_grf is None:
                try:
                    self.grf = MOT.load_from_mot(path_to_corrected_grf)
                except OSError as e:
                    raise MissingPathException("Corrected Ground Reaction Forces (MOT) file", detail=e.strerror)
            self.paths.corrected_grf = path_to_corrected_grf

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
