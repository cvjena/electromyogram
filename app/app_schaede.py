import json
from pathlib import Path

import cv2
import face_projection
import gradio as gr
import numpy as np
from PIL import Image

import electromyogram

# load schaede_images
vals_fridlund = json.load(open("data/fridlund.json"))
vals_kuramoto = json.load(open("data/kuramoto.json"))

schaede_img = {
    f.stem.split("_")[1]: {
        "img": np.array(Image.open(f)),
        "lms": np.load(f.with_suffix(".npy"))[:468],
        "val_fridlund": vals_fridlund.get(f.stem.split("_")[1].replace("-", " "), None),
        "val_kuramoto": vals_kuramoto.get(f.stem.split("_")[1].replace("-", " "), None),
    }
    for f in sorted(list(Path("schaede_images").glob("*.png")))
}

WARPER = face_projection.Warper()
scheme_fri = electromyogram.Fridlund()
scheme_kur = electromyogram.Kuramoto()

# todo mirroring


def visualize(
    scheme: str,
    beta: float,
    colormap: str,
    white_background: bool,
    blackandwhite: bool,
    use_global: bool,
    size: int,
    mirror: bool,
    mirror_plane_width: int,
):
    movements = []
    size = int(size)

    scale = size / 4096
    SIZE = (size, size)
    WARPER.set_scale(scale=scale)

    vmin = np.inf
    vmax = -np.inf
    scheme_sel = scheme_fri if scheme == "Fridlund" else scheme_kur
    scheme_val = "val_fridlund" if scheme == "Fridlund" else "val_kuramoto"

    if use_global:
        for name, data in schaede_img.items():
            if data[scheme_val] is None:
                continue
            vmin = 0
            vmax = max(vmax, max(data[scheme_val].values()))

    for name, data in schaede_img.items():
        if data[scheme_val] is None:
            print("Skipping", name)
            continue
        if not use_global:
            vmin = 0
            vmax = max(data[scheme_val].values())

        intr = electromyogram.interpolate(scheme=scheme_sel, emg_values=data[scheme_val], shape=SIZE, mirror=mirror, mirror_plane_width=mirror_plane_width)
        if mirror:
            colorized = [electromyogram.colorize(i, vmin=vmin, vmax=vmax, cmap=colormap, white_background=white_background) for i in intr]
        else:
            colorized = [electromyogram.colorize(intr, vmin=vmin, vmax=vmax, cmap=colormap, white_background=white_background)]

        temp = cv2.resize(data["img"], SIZE)
        if blackandwhite:
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

        warp_face = WARPER.apply(temp, img_data=colorized[0], beta=beta)
        warp_zero = WARPER.apply(np.zeros_like(temp), img_data=colorized[0], beta=1.0)

        plot = cv2.hconcat([temp, *colorized, warp_zero, warp_face])
        movements.append(plot)

    new_image = cv2.vconcat(movements)

    return gr.Image.update(value=new_image, label=f"Visualization {scheme}")


with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown("# Schaede Visualization")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Visualization Settings")

            scheme = gr.Dropdown(["Fridlund", "Kuramoto"], value="Kuramoto", label="Scheme")
            beta = gr.Slider(0.0, 0.999, 0.4, label="Beta")
            colormap = gr.Dropdown(["viridis", "plasma", "inferno", "magma", "jet", "bone", "parula"], value="parula", label="Colormap")
            white_background = gr.Checkbox(False, label="White Background")
            blackandwhite = gr.Checkbox(False, label="Black and White")
            use_global = gr.Checkbox(False, label="Use Global")
            size = gr.Slider(minimum=128, maximum=2048, step=128, value=256, label="Size")

            mirror = gr.Checkbox(False, label="Mirror")
            mirror_plane_width = gr.Number(
                2,
                precision=0,
                label="Mirror Plane Width",
            )

            btn = gr.Button("Visualize", variant="primary")

        with gr.Column():
            gr.Markdown("## Visualization")
            vis_img = gr.Image(schaede_img["Neutral"]["img"], label="Visualization")

    inp_params = [scheme, beta, colormap, white_background, blackandwhite, use_global, size, mirror, mirror_plane_width]
    out_params = vis_img

    btn.click(visualize, inputs=inp_params, outputs=out_params)
    demo.load(fn=visualize, inputs=inp_params, outputs=out_params)


if __name__ == "__main__":
    demo.launch(show_api=False, title="sEMG Sensor Positioning")
