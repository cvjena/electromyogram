import json
from pathlib import Path

import cv2
import face_projection
import gradio as gr
import numpy as np
from PIL import Image

import electromyogram

# load schaede_images
test_values = json.load(open("data/fridlund.json"))
schaede_img = {
    f.stem.split("_")[1]: {
        "img": np.array(Image.open(f)),
        "lms": np.load(f.with_suffix(".npy"))[:468],
        "val": test_values.get(f.stem.split("_")[1].replace("-", " "), None),
    }
    for f in sorted(list(Path("schaede_images").glob("*.png")))
}

WARPER = face_projection.Warper()
scheme = electromyogram.Fridlund()

# todo kuramoto
# todo mirroring


def visualize(
    beta: float,
    colormap: str,
    blackandwhite: bool,
    use_global: bool,
    size: int,
):
    plots = []
    size = int(size)

    scale = size / 4096
    SIZE = (size, size)
    WARPER.set_scale(scale=scale)

    vmin = np.inf
    vmax = -np.inf
    if use_global:
        for name, data in schaede_img.items():
            if data["val"] is None:
                continue
            vmin = 0
            vmax = max(vmax, max(data["val"].values()))

    for name, data in schaede_img.items():
        if data["val"] is None:
            continue
        if not use_global:
            vmin = 0
            vmax = max(data["val"].values())
        plot = electromyogram.plot(None, scheme=scheme, emg_values=data["val"], shape=SIZE)
        plot = electromyogram.colorize(plot, vmin=vmin, vmax=vmax, cmap=colormap)

        temp = cv2.resize(data["img"], SIZE)
        if blackandwhite:
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

        warp_face = WARPER.apply(temp, img_data=plot, beta=beta)
        warp_zero = WARPER.apply(np.zeros_like(temp), img_data=plot, beta=1.0)
        plot = cv2.hconcat([temp, plot, warp_zero, warp_face])
        plots.append(plot)

    new_image = cv2.vconcat(plots)

    return gr.Image.update(value=new_image)


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Schaede Visualization")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Visualization Settings")

            beta = gr.Slider(0.0, 0.999, 0.4, label="Beta")
            colormap = gr.Dropdown(["viridis", "plasma", "inferno", "magma", "jet", "bone", "parula"], value="parula", label="Colormap")
            blackandwhite = gr.Checkbox(False, label="Black and White")
            use_global = gr.Checkbox(False, label="Use Global")
            size = gr.Slider(minimum=128, maximum=2048, step=128, value=256, label="Size")
            btn = gr.Button("Visualize", variant="primary")

        with gr.Column():
            gr.Markdown("## Visualization")
            vis_img = gr.Image(schaede_img["Neutral"]["img"], label="Visualization")

    inp_params = [beta, colormap, blackandwhite, use_global, size]
    out_params = vis_img

    btn.click(visualize, inputs=inp_params, outputs=out_params)
    demo.load(fn=visualize, inputs=inp_params, outputs=out_params)


if __name__ == "__main__":
    demo.launch(show_api=False, title="sEMG Sensor Positioning")
