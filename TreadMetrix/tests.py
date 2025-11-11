import logging
import resources.paths.paths_gui as paths

logger = logging.getLogger("test")
logging.basicConfig(filename='test.log', level=logging.INFO)

paths.main()
print("All done.")
