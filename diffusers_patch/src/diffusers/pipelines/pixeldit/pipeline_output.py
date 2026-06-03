from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput


@dataclass
class PixelDiTPipelineOutput(BaseOutput):
    """
    Output class for PixelDiT text-to-image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`):
            List of generated images or numpy array of shape `(batch, height, width, 3)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
