from PIL import Image
import gradio as gr
import numpy as np
import safetensors
import argparse
import einops
import random
import torch
import copy
import math
import time
import uuid
import re
import os

from SUPIR.util import (
    HWC3,
    upscale_image,
    fix_resize,
    convert_dtype,
    create_SUPIR_model,
)

from modules.paths import models_path
from modules import sd_models
import spaces


with spaces.capture_gpu_object() as GO:
    SUPIR_device = spaces.gpu
    SUPIR_CKPT = os.path.join(models_path, "SUPIR", "SUPIR-v0Q.safetensors")
    CKPT = sd_models.model_data.forge_loading_parameters["checkpoint_info"].filename

    model, default_setting = create_SUPIR_model(
        "options/SUPIR_v0.yaml",
        SDXL_CKPT=CKPT,
        SUPIR_CKPT=SUPIR_CKPT,
        load_default_setting=True,
    )

    model = model.half()

    model.init_tile_vae(
        encoder_tile_size=512,
        decoder_tile_size=64,
    )

    model = model.to(SUPIR_device)

    model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(
        model.first_stage_model.denoise_encoder
    )

    model.current_model = "v0-Q"
    ckpt_Q = safetensors.torch.load_file(SUPIR_CKPT, device="cpu")
    ckpt_F = None

spaces.automatically_move_to_gpu_when_forward(model)
spaces.automatically_move_to_gpu_when_forward(ckpt_Q)

block = gr.Blocks().queue()
with block:
    gr.Markdown("Hola")

demo = block


if __name__ == "__main__":
    demo.launch()
