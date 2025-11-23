import logging
import os
import re

import numpy as np
import pandas as pd
import PySide6.QtCore
import resources.paths.paths_gui as paths
from ptb.util.io.mocap.low_lvl import c3d
from ptb.util.osim.osim_store import OSIMStorageV2

from resources.file_types.trc import TRC

logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'test.log'), level=logging.INFO)

# Prints PySide6 version
print(PySide6.__version__)

# Prints the Qt version used to compile PySide6
print(PySide6.QtCore.__version__)










print("All done.")
