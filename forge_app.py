from rich import print
from PIL import Image
import gradio as gr
import numpy as np
import safetensors
import einops
import random
import torch
import copy
import math
import time
import sys
import re
import os

from SUPIR.util import (
    HWC3,
    upscale_image,
    fix_resize,
    convert_dtype,
    create_SUPIR_model,
)

from backend import memory_management
from modules.paths import models_path
from modules import sd_models
import spaces

stdout = sys.stdout

POS_PROMPT = "cinematic, high contrast, detailed, canon camera, photorealistic, maximum detail, 4k, color grading, ultra hd, sharpness, perfect"
NEG_PROMPT = "painting, illustration, drawing, art, sketch, anime, cartoon, CG Style, 3D render, blur, aliasing, unsharp, weird textures, ugly, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, lowres"


CSS: list[str] = [
    ".tab-nav { justify-content: center; }",
    ".gradio-image { margin: auto; }",
]

if os.path.exists("../style.css"):
    with open("../style.css", "r") as style:
        styles = style.readlines()
        CSS += styles

with spaces.capture_gpu_object() as GO:

    print("[bright_black]Loading SUPIR...")

    with open(os.devnull, "w") as fnull:
        sys.stdout = fnull

        SUPIR_device = spaces.gpu
        SUPIR_CKPT = os.path.join(models_path, "SUPIR", "SUPIR-v0Q.safetensors")
        CKPT = sd_models.model_data.forge_loading_parameters["checkpoint_info"].filename

        model = create_SUPIR_model(
            "options/SUPIR_v0.yaml",
            SDXL_CKPT=CKPT,
            SUPIR_CKPT=SUPIR_CKPT,
            load_default_setting=False,
        )

        model = model.half()

        model.init_tile_vae(
            encoder_tile_size=512,
            decoder_tile_size=64,
        )

        model = model.to(SUPIR_device)
        # model.to(dtype=torch.float8_e4m3fn, device=SUPIR_device)

        model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(
            model.first_stage_model.denoise_encoder
        )

        model.current_model = "v0-Q"
        ckpt_Q = safetensors.torch.load_file(SUPIR_CKPT, device="cpu")
        ckpt_F = None

    sys.stdout = stdout

spaces.automatically_move_to_gpu_when_forward(model, model.model)
spaces.automatically_move_to_gpu_when_forward(ckpt_Q)


def validate(input_image: np.ndarray) -> bool:
    if input_image is None:
        raise gr.Error("Upload an image to restore...")


@spaces.GPU(gpu_objects=GO, manual_load=False)
def stage1_process(
    input_image: np.ndarray,
    gamma_correction: float,
    diff_dtype: str,
    ae_dtype: str,
):

    print("[cyan]stage 1 preprocess")

    if torch.cuda.device_count() == 0:
        gr.Warning("Non-CUDA Device is probably not supported...")
        # return None, None

    # torch.cuda.set_device(SUPIR_device)

    LQ = HWC3(input_image)
    LQ = fix_resize(LQ, 512)
    LQ = np.array(LQ) / 255 * 2 - 1
    LQ = (
        torch.tensor(LQ, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(SUPIR_device)[:, :3, :, :]
    )

    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    LQ = model.batchify_denoise(LQ, is_stage1=True)
    LQ = (
        (LQ[0].permute(1, 2, 0) * 127.5 + 127.5)
        .cpu()
        .numpy()
        .round()
        .clip(0, 255)
        .astype(np.uint8)
    )

    # gamma correction
    if gamma_correction != 1.0:
        LQ /= 255.0
        LQ = np.power(LQ, gamma_correction)
        LQ *= 255.0

    LQ = LQ.round().clip(0, 255).astype(np.uint8)
    print("[green]stage 1 done")
    memory_management.soft_empty_cache()
    return LQ


@spaces.GPU(gpu_objects=GO, manual_load=False)
def stage2_process(
    noisy_image,
    denoise_image,
    prompt,
    p_prompt,
    n_prompt,
    upscale,
    edm_steps,
    s_stage1,
    s_stage2,
    s_cfg,
    seed,
    s_churn,
    s_noise,
    color_fix_type,
    diff_dtype,
    ae_dtype,
    gamma_correction,
    linear_CFG,
    linear_s_stage2,
    spt_linear_CFG,
    spt_linear_s_stage2,
):

    prompt = prompt or ""
    p_prompt = p_prompt or ""
    n_prompt = n_prompt or ""

    if prompt:
        a_prompt = f"{prompt}, {p_prompt}"
    else:
        a_prompt = p_prompt

    print(f"Final Prompt: {a_prompt}")

    denoise_image = denoise_image or noisy_image
    denoise_image = HWC3(denoise_image)

    if torch.cuda.device_count() == 0:
        gr.Warning("Non-CUDA Device is probably not supported...")

    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    start = time.time()

    memory_management.soft_empty_cache()
    print("[cyan]stage 2 restore")

    # torch.cuda.set_device(SUPIR_device)

    with torch.inference_mode():
        input_image = upscale_image(
            denoise_image, upscale, unit_resolution=32, min_size=512
        )

        LQ = np.asarray(input_image, dtype=np.float32)
        LQ = LQ / 255.0 * 2.0 - 1.0

        LQ = (
            torch.tensor(LQ, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(SUPIR_device)[:, :3, :, :]
        )

        captions = [""]

        samples = model.batchify_sample(
            LQ,
            captions,
            num_steps=edm_steps,
            restoration_scale=s_stage1,
            s_churn=s_churn,
            s_noise=s_noise,
            cfg_scale=s_cfg,
            control_scale=s_stage2,
            seed=seed,
            num_samples=1,
            p_p=a_prompt,
            n_p=n_prompt,
            color_fix_type=color_fix_type,
            use_linear_CFG=linear_CFG,
            use_linear_control_scale=linear_s_stage2,
            cfg_scale_start=spt_linear_CFG,
            control_scale_start=spt_linear_s_stage2,
        )

        x_samples = (
            (einops.rearrange(samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )

        result = x_samples[0]

    memory_management.soft_empty_cache()

    end = time.time()
    secondes = int(end - start)

    print("[green]restore done")
    print(f"Took: {secondes}s")

    return result


def reset() -> list:
    return [
        None,  # input_image
        None,  # denoised_image
        None,  # output_image
        "",  # prompt
        2,  # upscale
        -1,  # seed
        "AdaIn",  # color_fix_type
        1,  # gamma_correction
        "Quality",  # preset
        POS_PROMPT,  # p_prompt
        NEG_PROMPT,  # n_prompt
    ]


def load_preset(preset: str) -> list:

    if preset == "Quality":
        s_cfg = 7.5
        spt_linear_CFG = 4.0
    else:
        s_cfg = 4.0
        spt_linear_CFG = 1.0

    edm_steps = 48
    s_stage2 = 1.0
    s_stage1 = -1.0
    s_churn = 5
    s_noise = 1.003
    spt_linear_s_stage2 = 0.0
    linear_s_stage2 = False
    linear_CFG = True

    return (
        edm_steps,
        s_cfg,
        s_stage2,
        s_stage1,
        s_churn,
        s_noise,
        linear_CFG,
        linear_s_stage2,
        spt_linear_CFG,
        spt_linear_s_stage2,
    )


block = gr.Blocks(css="\n".join(CSS)).queue()

with block:
    gr.Markdown('<h1 align="center">SUPIR</h1>')

    if torch.cuda.device_count() == 0:
        gr.Markdown(
            '<h3 align="center">Non-CUDA Device is probably not supported...</h3>'
        )

    with gr.Row(variant="panel", elem_classes="Image IO"):

        with gr.Tab("Input Image"):
            gr.HTML('<p align="center">Upload your Input Image here:</p>')

            input_image = gr.Image(
                label="Low Resolution Input",
                sources="upload",
                type="numpy",
                image_mode="RGB",
                width=512,
                height=512,
                elem_id="image-input",
            )

        with gr.Tab("Preprocess (Optional)"):
            gr.HTML('<p align="center">Denoise the Input Image before Upscaling</p>')

            denoised_image = gr.Image(
                label="Preprocessed Input",
                sources="upload",
                type="numpy",
                image_mode="RGB",
                width=512,
                height=512,
                elem_id="image-s1",
                interactive=False,
            )

            with gr.Row(variant="compact"):

                gamma_correction = gr.Slider(
                    label="Gamma Correction",
                    info="brighter | darker",
                    minimum=0.2,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                )

                denoise_button = gr.Button(value="Denoise", variant="primary")

        output_image = gr.Image(
            label="High Resolution Output",
            sources="upload",
            type="numpy",
            image_mode="RGB",
            width=768,
            height=768,
            elem_id="gallery1",
            interactive=False,
            show_download_button=True,
            show_share_button=False,
        )

    with gr.Row(variant="panel", elem_classes="Settings"):

        with gr.Column(variant="compact", elem_classes="Prompts"):

            prompt = gr.Textbox(
                label="Image Description / Caption",
                info="Describe as much as possible, especially the details not present in the original image",
                value="",
                placeholder="A 33 years old man, walking on the street on a summer morning, Santiago",
                lines=4,
                max_lines=4,
            )

            with gr.Accordion("Additional Prompts", open=False):

                p_prompt = gr.Textbox(
                    label="Positive Prompt",
                    info="Quality description that gets appended after the main caption",
                    value=POS_PROMPT,
                    lines=4,
                    max_lines=4,
                )

                n_prompt = gr.Textbox(
                    label="Negative Prompt",
                    info="List what the image should not contain",
                    value=NEG_PROMPT,
                    lines=4,
                    max_lines=4,
                )

        with gr.Column(variant="compact", elem_classes="Configs"):

            with gr.Row(elem_classes="Presets"):

                preset = gr.Radio(
                    label="Presets",
                    choices=("Quality", "Fidelity"),
                    value="Quality",
                    scale=2,
                )

                apply_preset = gr.Button(value="Apply Preset", scale=1)

            with gr.Row(elem_classes="Steps"):

                edm_steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=128,
                    value=48,
                    step=1,
                )

                seed = gr.Slider(
                    label="Seed", value=-1, minimum=-1, maximum=4294967295, step=1
                )

            with gr.Accordion(label="Advanced Settings", open=False):

                with gr.Tab("CFG"):

                    s_cfg = gr.Slider(
                        label="Text Guidance Scale",
                        info="Guided by Image | Guided by Prompt",
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                    )

                    with gr.Row():

                        s_stage2 = gr.Slider(
                            label="Restoring Guidance Strength",
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                        )

                        s_stage1 = gr.Slider(
                            label="Pre-denoising Guidance Strength",
                            minimum=-1.0,
                            maximum=6.0,
                            value=-1.0,
                            step=1.0,
                        )

                    with gr.Row():

                        s_churn = gr.Slider(
                            label="S-Churn", minimum=0, maximum=40, value=5, step=1
                        )

                        s_noise = gr.Slider(
                            label="S-Noise",
                            minimum=1.0,
                            maximum=1.1,
                            value=1.003,
                            step=0.001,
                        )

                    with gr.Row():

                        linear_CFG = gr.Checkbox(label="Linear CFG", value=True)

                        linear_s_stage2 = gr.Checkbox(
                            label="Linear Restoring Guidance", value=False
                        )

                    with gr.Row():

                        spt_linear_CFG = gr.Slider(
                            label="CFG Start",
                            minimum=1.0,
                            maximum=9.0,
                            value=4,
                            step=0.5,
                        )

                        spt_linear_s_stage2 = gr.Slider(
                            label="Guidance Start",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                        )

                with gr.Tab("dtype"):

                    bf16: bool = torch.cuda.is_bf16_supported()

                    diff_dtype = gr.Radio(
                        label="Diffusion Data Type",
                        choices=("fp32", "fp16", "bf16", "fp8"),
                        value="bf16" if bf16 else "fp32",
                    )

                    ae_dtype = gr.Radio(
                        label="Auto-Encoder Data Type",
                        choices=("fp32", "bf16"),
                        value="bf16" if bf16 else "fp32",
                    )

        with gr.Column(variant="panel", elem_classes="Buttons"):

            color_fix_type = gr.Radio(
                label="Color-Fix Mode",
                choices=("None", "AdaIn", "Wavelet"),
                info="AdaIn for photo ; Wavelet for JPEG artifacts",
                value="AdaIn",
            )

            upscale = gr.Radio(
                label="Upscale Factor",
                choices=(
                    ("x2", 2),
                    ("x4", 4),
                    ("x8", 8),
                ),
                value=2,
            )

            diffusion_button = gr.Button(
                value="🚀 Upscale",
                variant="primary",
                elem_id="process_button",
            )

            reset_btn = gr.Button(
                value="🧹 Reset Page",
                variant="stop",
                elem_id="reset_button",
            )

    gr.HTML(
        """<p align="center">
            <a href="https://arxiv.org/abs/2401.13627">Paper</a> &emsp; <a href="http://supir.xpixel.group/">Project Page</a> &emsp; <a href="https://github.com/Fanghua-Yu/SUPIR">GitHub Repo</a>
        </p>"""
    )

    denoise_button.click(
        fn=validate, inputs=[input_image], show_progress="hidden"
    ).success(
        fn=stage1_process,
        inputs=[input_image, gamma_correction, diff_dtype, ae_dtype],
        outputs=[denoised_image],
    )

    diffusion_button.click(
        fn=validate, inputs=[input_image], show_progress="hidden"
    ).success(
        fn=stage2_process,
        inputs=[
            input_image,
            denoised_image,
            prompt,
            p_prompt,
            n_prompt,
            upscale,
            edm_steps,
            s_stage1,
            s_stage2,
            s_cfg,
            seed,
            s_churn,
            s_noise,
            color_fix_type,
            diff_dtype,
            ae_dtype,
            gamma_correction,
            linear_CFG,
            linear_s_stage2,
            spt_linear_CFG,
            spt_linear_s_stage2,
        ],
        outputs=[output_image],
    )

    apply_preset.click(
        fn=load_preset,
        inputs=[preset],
        outputs=[
            edm_steps,
            s_cfg,
            s_stage2,
            s_stage1,
            s_churn,
            s_noise,
            linear_CFG,
            linear_s_stage2,
            spt_linear_CFG,
            spt_linear_s_stage2,
        ],
        show_progress="hidden",
    )

    reset_btn.click(
        fn=reset,
        inputs=None,
        outputs=[
            input_image,
            denoised_image,
            output_image,
            prompt,
            upscale,
            seed,
            color_fix_type,
            gamma_correction,
            preset,
            p_prompt,
            n_prompt,
        ],
        show_progress="hidden",
    ).success(
        fn=load_preset,
        inputs=[preset],
        outputs=[
            edm_steps,
            s_cfg,
            s_stage2,
            s_stage1,
            s_churn,
            s_noise,
            linear_CFG,
            linear_s_stage2,
            spt_linear_CFG,
            spt_linear_s_stage2,
        ],
        show_progress="hidden",
    )


demo = block


if __name__ == "__main__":
    demo.launch()
