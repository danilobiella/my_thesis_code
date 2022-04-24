import src.lightcurve as lcu


def first_orbit_cycles(_: lcu.LightCurveCycles) -> list[int]:
    """
    Always returns the index of the first 35 cycles.
    """
    return list(range(35))
