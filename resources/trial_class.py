import os.path
from resources.file_types.mot import MOT
from resources.file_types.trc import TRC
from typing import Self


class GaitCycle:
    def __init__(self, number: int, side: str, mot: MOT, trc: TRC) -> None:
        self.number = number
        self.side = side
        self.trc = trc
        self.mot = mot
        self.inverse_kinematics = None
        self.inverse_dynamic = None
        self.joint_power = None

    def __gt__(self, other: Self) -> bool:
        """Overrides the default implementation of "strictly greater than" operation.

        Args:
            other: GaitCycle object to compare

        Returns:
            bool
        """
        return self.number > other.number

    def add_ik(self, mot: MOT) -> None:
        self.inverse_kinematics = mot

    def add_id(self, mot: MOT) -> None:
        self.inverse_dynamic = mot

    def add_joint_power(self, mot: MOT) -> None:
        self.joint_power = mot


class Trial:
    def __init__(self, mot_path: str, trc_path: str) -> None:
        self.mot = MOT.load_from_mot(mot_path)
        self.name = os.path.basename(mot_path).replace(".mot", ".trc")
        self.trc = TRC.load_from_trc(trc_path)
        self.gait_cycles = {"Right": [], "Left": []}

    def add_cycles(self, right_cycles: list[GaitCycle], left_cycles: list[GaitCycle]) -> None:
        self.gait_cycles["Right"] = right_cycles
        self.gait_cycles["Left"] = left_cycles

    def add_cycle(self, cycle: GaitCycle) -> None:
        side = cycle.side
        self.gait_cycles[side].append(cycle)

    def sort_cycles(self) -> None:
        for side in ["Right", "Left"]:
            self.gait_cycles[side].sort()
