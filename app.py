# coding: utf-8

"""
The entrance of the gradio
"""

import tyro
import subprocess
import gradio as gr
import os.path as osp
import webbrowser
import os

from src.utils.helper import load_description, mkdir
from src.gradio_pipeline import GradioPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg before running this script. https://ffmpeg.org/download.html"
    )

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

gradio_pipeline = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)

# Ensure output directory exists for open folder
mkdir(args.output_dir)

#save folder button for maximum cross-platform compatibility
def open_output_folder():
    folder_path = os.path.normpath(os.path.realpath(args.output_dir))
    mkdir(folder_path)  # double ensure the directory exists
    webbrowser.open(f'file://{folder_path}')
        
#define img to copy to retargeting_input
def copy_image(img):
    return img
    
def gpu_wrapped_execute_video(*args, **kwargs):
    return gradio_pipeline.execute_video(*args, **kwargs)

def gpu_wrapped_execute_image(*args, **kwargs):
    return gradio_pipeline.execute_image(*args, **kwargs)


# assets
title_md = "assets/gradio_title.md"
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
data_examples = [
    [osp.join(example_portrait_dir, "s9.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s6.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s10.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s5.jpg"), osp.join(example_video_dir, "d18.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s7.jpg"), osp.join(example_video_dir, "d19.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s2.jpg"), osp.join(example_video_dir, "d13.mp4"), True, True, True, True],
]
#################### interface logic ####################

# Define components first
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
retargeting_input_image = gr.Image(type="filepath")
output_image = gr.Image(type="numpy")
output_image_paste_back = gr.Image(type="numpy")
output_video = gr.Video()
output_video_concat = gr.Video()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(load_description(title_md))
    gr.Markdown(load_description("assets/gradio_description_upload.md"))
    with gr.Row():
        with gr.Accordion(open=True, label="Source Portrait. Try the Image Examples below."):
            image_input = gr.Image(type="filepath")
            
        with gr.Accordion(open=True, label="Driving Video. Try the Driving Examples below."):
            video_input = gr.Video()
            
    with gr.Row():
        with gr.Accordion(open=False, label="Image Examples"):            
            gr.Examples(
                examples=[
                    [osp.join(example_portrait_dir, "AI_girl1.png")],
                    [osp.join(example_portrait_dir, "pinokio1.png")],
                    [osp.join(example_portrait_dir, "AI_girl2.png")],
                    [osp.join(example_portrait_dir, "AI_guy1.png")],
                    [osp.join(example_portrait_dir, "AI_guy2.png")],
                    [osp.join(example_portrait_dir, "s7.jpg")],
                    [osp.join(example_portrait_dir, "s9.jpg")],
                    [osp.join(example_portrait_dir, "s10.jpg")],
                    [osp.join(example_portrait_dir, "s5.jpg")],
                    [osp.join(example_portrait_dir, "s6.jpg")],
                ],
                inputs=[image_input],
                cache_examples=False,
            )
            
        with gr.Accordion(open=False, label="Driving Examples"):
            gr.Examples(
                examples=[
                    [osp.join(example_video_dir, "d7.mp4")],
                    [osp.join(example_video_dir, "d6.mp4")],
                    [osp.join(example_video_dir, "d3.mp4")],
                    [osp.join(example_video_dir, "d12.mp4")],
                    [osp.join(example_video_dir, "d18.mp4")],
                    [osp.join(example_video_dir, "d19.mp4")],
                    [osp.join(example_video_dir, "smile1.mp4")],
                    [osp.join(example_video_dir, "d0.mp4")],
                    [osp.join(example_video_dir, "d10.mp4")],
                    [osp.join(example_video_dir, "d5.mp4")],
                    [osp.join(example_video_dir, "d9.mp4")],
                    [osp.join(example_video_dir, "d14.mp4")],
                    [osp.join(example_video_dir, "d8.mp4")],
                ],
                inputs=[video_input],
                cache_examples=False,
            )
    with gr.Row():
        with gr.Accordion(open=False, label="Animation Instructions"):
            gr.Markdown(load_description("assets/gradio_description_animation.md"))
            
    with gr.Row():
        with gr.Accordion(open=True, label="Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=True, label="relative motion")
                flag_do_crop_input = gr.Checkbox(value=True, label="crop (source)")
                flag_remap_input = gr.Checkbox(value=True, label="paste-back")
                flag_crop_driving_video_input = gr.Checkbox(value=False, label="crop (driving video)")
                flag_lip_zero = gr.Checkbox(value=True, label="lip-zero")    
                flag_stitching =  gr.Checkbox(value=True, label="stitching(?)")        
                open_folder_button = gr.Button("📁 output folder", variant="primary", size="sm")

            open_folder_button.click(fn=open_output_folder)
                
            
    with gr.Row():
        with gr.Column():
            process_button_animation = gr.Button("🚀 Animate", variant="primary")
        with gr.Column():
            process_button_reset = gr.ClearButton([image_input, video_input, output_video, output_video_concat], value="🧹 Clear")
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video in the original image space"):
                output_video.render()
        with gr.Column():
            with gr.Accordion(open=True, label="The animated video"):
                output_video_concat.render()
    with gr.Row():
        # Examples
        with gr.Accordion(open=False, label="Preset Examples"):
            with gr.Row():
                gr.Examples(
                examples=data_examples,
                fn=gpu_wrapped_execute_video,
                inputs=[
                    image_input,
                    video_input,
                    flag_relative_input,
                    flag_do_crop_input,
                    flag_remap_input,
                    flag_crop_driving_video_input,
                    flag_lip_zero,
                    flag_stitching
            ],
            outputs=[output_image, output_image_paste_back],
            examples_per_page=len(data_examples),
            cache_examples=False,
        )
    gr.Markdown(load_description("assets/gradio_description_retargeting.md"), visible=True)
    with gr.Row(visible=True):
        eye_retargeting_slider.render()
        lip_retargeting_slider.render()
    with gr.Row(visible=True):
        process_button_retargeting = gr.Button("🚗 Retargeting", variant="primary")
        process_button_reset_retargeting = gr.ClearButton(
            [
                eye_retargeting_slider,
                lip_retargeting_slider,
                retargeting_input_image,
                output_image,
                output_image_paste_back
            ],
            value="🧹 Clear"
        )
    with gr.Row():
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Input"):
                retargeting_input_image.render()
#                gr.Examples(
#                    examples=[
#                        [osp.join(example_portrait_dir, "s9.jpg")],
#                        [osp.join(example_portrait_dir, "s6.jpg")],
#                        [osp.join(example_portrait_dir, "s10.jpg")],
#                        [osp.join(example_portrait_dir, "s5.jpg")],
#                        [osp.join(example_portrait_dir, "s7.jpg")],
#                        [osp.join(example_portrait_dir, "s12.jpg")],
#                    ],
#                inputs=[retargeting_input_image],
#                cache_examples=False,
#                )
        with gr.Column():
            with gr.Accordion(open=True, label="Retargeting Result"):
                output_image.render()
        with gr.Column():
            with gr.Accordion(open=True, label="Paste-back Result"):
                output_image_paste_back.render()
    # binding functions for buttons
    process_button_retargeting.click(
        # fn=gradio_pipeline.execute_image,
        fn=gpu_wrapped_execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image, flag_do_crop_input],
        outputs=[output_image, output_image_paste_back],
        show_progress=True
    )
    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            image_input,
            video_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            flag_crop_driving_video_input,
            flag_lip_zero,
            flag_stitching
        ],
        outputs=[output_video, output_video_concat],
        show_progress=True
    )
    image_input.change(
        fn=copy_image,
        inputs=image_input,
        outputs=retargeting_input_image
    )


demo.launch(
    server_port=args.server_port,
    share=args.share,
    server_name=args.server_name
)
