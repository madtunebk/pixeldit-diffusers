import unittest
import torch
from diffusers import PixelDiTPipeline
from diffusers.utils.testing_utils import require_torch_gpu, slow


@slow
@require_torch_gpu
class PixelDiTPipelineSlowTests(unittest.TestCase):

    def get_pipeline(self):
        return PixelDiTPipeline.from_pretrained(
            "madtune/pixeldit-diffusers",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def test_pixeldit_inference_default(self):
        pipe = self.get_pipeline()
        out = pipe("a white horse running in a meadow at sunset", generator=torch.Generator("cpu").manual_seed(42))
        self.assertEqual(len(out.images), 1)
        self.assertEqual(out.images[0].size, (512, 512))

    def test_pixeldit_inference_cfg(self):
        pipe = self.get_pipeline()
        out = pipe(
            "portrait of a woman with fire and ice elemental powers",
            guidance_scale=7.5,
            num_inference_steps=20,
            height=512,
            width=512,
            generator=torch.Generator("cpu").manual_seed(0),
        )
        self.assertEqual(len(out.images), 1)

    def test_pixeldit_negative_prompt(self):
        pipe = self.get_pipeline()
        out = pipe(
            "a dragon over a city",
            negative_prompt="blurry, low quality, cartoon",
            generator=torch.Generator("cpu").manual_seed(0),
        )
        self.assertEqual(len(out.images), 1)

    def test_pixeldit_output_np(self):
        pipe = self.get_pipeline()
        import numpy as np
        out = pipe("a sunset", output_type="np", generator=torch.Generator("cpu").manual_seed(0))
        self.assertIsInstance(out.images, np.ndarray)
        self.assertEqual(out.images.shape[-1], 3)
