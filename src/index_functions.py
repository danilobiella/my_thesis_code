import src.lightcurve as lcu
from typing import List

def first_orbit_cycles(_: lcu.LightCurveCycles) -> List[int]:
    """
    Always returns the index of the first 35 cycles.
    """
    return list(range(38))
