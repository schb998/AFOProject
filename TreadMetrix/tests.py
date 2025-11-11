import logging
import os

import resources.paths.paths_gui as paths

logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'test.log'), level=logging.INFO)

paths.main()
print("All done.")
