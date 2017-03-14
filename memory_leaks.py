import yt
from yt import memory_checker
import numpy as np
import time

# sim = yt.simulation("test_dir", "ExodusII")
with memory_checker(1):
    time.sleep(2)
    sim = yt.simulation("hundred_files", "ExodusII")
    sim.get_time_series()
    time.sleep(2)
