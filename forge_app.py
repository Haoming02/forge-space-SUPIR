from PIL import Image
import gradio as gr
import numpy as np
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
    load_QF_ckpt,
)

import spaces


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)


demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

if __name__ == "__main__":
    demo.launch()
