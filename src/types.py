"""
Custom types
"""
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

PositionalBounds = Union[list[int], npt.NDArray[np.int64]]

LossFunction = Callable[
    [Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]], float
]
