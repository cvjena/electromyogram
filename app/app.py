# This is a gradio app which allows us to position the locations for the EMG sensors on the face.
# There are sliders for each sensor (x, y) relative to the center of the face canvas.
# if the sliders are moved, the canvas is redrawn with the new positions.
# including the projection of the EMG sensors onto the face.

import face_projection
import gradio as gr
import numpy as np
from PIL import Image

import electromyogram

FACE_CANVAS = np.array(Image.open("face_canvas.png"))
FACE_IMG = np.array(Image.open("AdobeStock_35393559.jpeg"))[500:2000, 700:2000]

WARPER = face_projection.Warper()
# WARPER.set_scale(0.25)

SLIDERS_KURAMOTO = {}

KURAMOTO = electromyogram.Kuramoto()
FRIDLUND = electromyogram.Fridlund()


def reset():
    global KURAMOTO, FRIDLUND
    KURAMOTO = electromyogram.Kuramoto()
    FRIDLUND = electromyogram.Fridlund()


def save_locations(current):
    if current == "Kuramoto":
        KURAMOTO.save_locs()
    elif current == "Fridlund":
        FRIDLUND.save_locs()
    else:
        raise ValueError("Unknown scheme: {current}")


def change_tab(tab):
    if tab == 0:
        current = "Kuramoto"
    elif tab == 1:
        current = "Fridlund"
    else:
        raise ValueError("Unknown tab: {tab}")
    return [
        gr.Dropdown.update(current, value=current),
        gr.Tab.update(0, active=tab == 0),
        gr.Tab.update(1, active=tab == 1),
    ]


def apply(current, beta):
    out_image = FACE_CANVAS.copy()
    if current == "Kuramoto":
        out_image = electromyogram.plot_locations(out_image, KURAMOTO)
    elif current == "Fridlund":
        out_image = electromyogram.plot_locations(out_image, FRIDLUND)
    # out_image = Image.fromarray(out_image).resize((WARPER.width(), WARPER.height()))
    out_face = WARPER.apply(FACE_IMG, np.array(out_image), beta=beta)

    return [
        gr.Image.update(out_image),
        gr.Image.update(out_face),
    ]


with gr.Blocks() as demo:
    sliders = []
    gr.Markdown("# sEMG Sensor Positioning")

    # we have two vertical columns, one for the sliders and one for the canvas
    with gr.Row():
        with gr.Column():
            force_update = gr.Button("Force Update")
            save_locs = gr.Button("Save Locations", variant="primary")

            with gr.Blocks():
                current = gr.Dropdown(["Kuramoto", "Fridlund"], value="Kuramoto", label="Current Scheme", interactive=True)
                beta = gr.Slider(0.1, 0.999, 0.8, label="Beta", interactive=True)

            with gr.Tab("Kuramoto", id=0):
                with gr.Row():
                    # E1 and E2
                    x = gr.Number(KURAMOTO.locations["E1"][0], label="E1 E2 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E1": (x, KURAMOTO.locations["E1"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E2": (-x, KURAMOTO.locations["E2"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E1"][1], label="E1 E2 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E1": (KURAMOTO.locations["E1"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E2": (KURAMOTO.locations["E2"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E3 and E4
                    x = gr.Number(KURAMOTO.locations["E3"][0], label="E3 E4 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E3": (x, KURAMOTO.locations["E3"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E4": (-x, KURAMOTO.locations["E4"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E3"][1], label="E3 E4 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E3": (KURAMOTO.locations["E3"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E4": (KURAMOTO.locations["E4"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E5 and E6
                    x = gr.Number(KURAMOTO.locations["E5"][0], label="E5 E6 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E5": (x, KURAMOTO.locations["E5"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E6": (-x, KURAMOTO.locations["E6"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E5"][1], label="E5 E6 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E5": (KURAMOTO.locations["E5"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E6": (KURAMOTO.locations["E6"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E7 and E8
                    x = gr.Number(KURAMOTO.locations["E7"][0], label="E7 E8 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E7": (x, KURAMOTO.locations["E7"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E8": (-x, KURAMOTO.locations["E8"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E7"][1], label="E7 E8 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E7": (KURAMOTO.locations["E7"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E8": (KURAMOTO.locations["E8"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E9 and E10
                    x = gr.Number(KURAMOTO.locations["E9"][0], label="E9 E10 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E9": (x, KURAMOTO.locations["E9"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E10": (-x, KURAMOTO.locations["E10"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E9"][1], label="E9 E10 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E9": (KURAMOTO.locations["E9"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E10": (KURAMOTO.locations["E10"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E13 and E14
                    x = gr.Number(KURAMOTO.locations["E13"][0], label="E13 E14 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E13": (x, KURAMOTO.locations["E13"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E14": (-x, KURAMOTO.locations["E14"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E13"][1], label="E13 E14 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E13": (KURAMOTO.locations["E13"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E14": (KURAMOTO.locations["E14"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E15 and E16
                    x = gr.Number(KURAMOTO.locations["E15"][0], label="E15 E16 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E15": (x, KURAMOTO.locations["E15"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E16": (-x, KURAMOTO.locations["E16"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E15"][1], label="E15 E16 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E15": (KURAMOTO.locations["E15"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E16": (KURAMOTO.locations["E16"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E17 and E18
                    x = gr.Number(KURAMOTO.locations["E17"][0], label="E17 E18 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E17": (x, KURAMOTO.locations["E17"][1])}), inputs=x)
                    x.change(lambda x: KURAMOTO.locations.update({"E18": (-x, KURAMOTO.locations["E18"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E17"][1], label="E17 E18 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E17": (KURAMOTO.locations["E17"][0], y)}), inputs=y)
                    y.change(lambda y: KURAMOTO.locations.update({"E18": (KURAMOTO.locations["E18"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E19
                    x = gr.Number(KURAMOTO.locations["E19"][0], label="E19 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E19": (x, KURAMOTO.locations["E19"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E19"][1], label="E19 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E19": (KURAMOTO.locations["E19"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E20
                    x = gr.Number(KURAMOTO.locations["E20"][0], label="E20 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E20": (x, KURAMOTO.locations["E20"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E20"][1], label="E20 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E20": (KURAMOTO.locations["E20"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # E24
                    x = gr.Number(KURAMOTO.locations["E24"][0], label="E24 x", interactive=True)
                    x.change(lambda x: KURAMOTO.locations.update({"E24": (x, KURAMOTO.locations["E24"][1])}), inputs=x)
                    y = gr.Number(KURAMOTO.locations["E24"][1], label="E24 y", interactive=True)
                    y.change(lambda y: KURAMOTO.locations.update({"E24": (KURAMOTO.locations["E24"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

            with gr.Tab("Fridlund", id=1):
                with gr.Row():
                    # DAO
                    x = gr.Number(FRIDLUND.locations["DAO li"][0], label="DAO x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"DAO li": (x, FRIDLUND.locations["DAO li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"DAO re": (-x, FRIDLUND.locations["DAO re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["DAO li"][1], label="DAO y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"DAO li": (FRIDLUND.locations["DAO li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"DAO re": (FRIDLUND.locations["DAO re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # OrbOr
                    x = gr.Number(FRIDLUND.locations["OrbOr li"][0], label="OrbOr x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"OrbOr li": (x, FRIDLUND.locations["OrbOr li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"OrbOr re": (-x, FRIDLUND.locations["OrbOr re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["OrbOr li"][1], label="OrbOr y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"OrbOr li": (FRIDLUND.locations["OrbOr li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"OrbOr re": (FRIDLUND.locations["OrbOr re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # Ment
                    x = gr.Number(FRIDLUND.locations["Ment li"][0], label="Ment x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"Ment li": (x, FRIDLUND.locations["Ment li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"Ment re": (-x, FRIDLUND.locations["Ment re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["Ment li"][1], label="Ment y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"Ment li": (FRIDLUND.locations["Ment li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"Ment re": (FRIDLUND.locations["Ment re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # Mass
                    x = gr.Number(FRIDLUND.locations["Mass li"][0], label="Mass x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"Mass li": (x, FRIDLUND.locations["Mass li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"Mass re": (-x, FRIDLUND.locations["Mass re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["Mass li"][1], label="Mass y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"Mass li": (FRIDLUND.locations["Mass li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"Mass re": (FRIDLUND.locations["Mass re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # Zyg
                    x = gr.Number(FRIDLUND.locations["Zyg li"][0], label="Zyg x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"Zyg li": (x, FRIDLUND.locations["Zyg li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"Zyg re": (-x, FRIDLUND.locations["Zyg re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["Zyg li"][1], label="Zyg y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"Zyg li": (FRIDLUND.locations["Zyg li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"Zyg re": (FRIDLUND.locations["Zyg re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # OrbOc
                    x = gr.Number(FRIDLUND.locations["OrbOc li"][0], label="OrbOc x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"OrbOc li": (x, FRIDLUND.locations["OrbOc li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"OrbOc re": (-x, FRIDLUND.locations["OrbOc re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["OrbOc li"][1], label="OrbOc y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"OrbOc li": (FRIDLUND.locations["OrbOc li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"OrbOc re": (FRIDLUND.locations["OrbOc re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # lat Front
                    x = gr.Number(FRIDLUND.locations["lat Front li"][0], label="lat Front x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"lat Front li": (x, FRIDLUND.locations["lat Front li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"lat Front re": (-x, FRIDLUND.locations["lat Front re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["lat Front li"][1], label="lat Front y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"lat Front li": (FRIDLUND.locations["lat Front li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"lat Front re": (FRIDLUND.locations["lat Front re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # med Front
                    x = gr.Number(FRIDLUND.locations["med Front li"][0], label="med Front x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"med Front li": (x, FRIDLUND.locations["med Front li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"med Front re": (-x, FRIDLUND.locations["med Front re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["med Front li"][1], label="med Front y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"med Front li": (FRIDLUND.locations["med Front li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"med Front re": (FRIDLUND.locations["med Front re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # Corr
                    x = gr.Number(FRIDLUND.locations["Corr li"][0], label="Corr x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"Corr li": (x, FRIDLUND.locations["Corr li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"Corr re": (-x, FRIDLUND.locations["Corr re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["Corr li"][1], label="Corr y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"Corr li": (FRIDLUND.locations["Corr li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"Corr re": (FRIDLUND.locations["Corr re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)

                with gr.Row():
                    # Deprsup
                    x = gr.Number(FRIDLUND.locations["Deprsup li"][0], label="Deprsup x", interactive=True)
                    x.change(lambda x: FRIDLUND.locations.update({"Deprsup li": (x, FRIDLUND.locations["Deprsup li"][1])}), inputs=x)
                    x.change(lambda x: FRIDLUND.locations.update({"Deprsup re": (-x, FRIDLUND.locations["Deprsup re"][1])}), inputs=x)
                    y = gr.Number(FRIDLUND.locations["Deprsup li"][1], label="Deprsup y", interactive=True)
                    y.change(lambda y: FRIDLUND.locations.update({"Deprsup li": (FRIDLUND.locations["Deprsup li"][0], y)}), inputs=y)
                    y.change(lambda y: FRIDLUND.locations.update({"Deprsup re": (FRIDLUND.locations["Deprsup re"][0], y)}), inputs=y)
                    sliders.append(x), sliders.append(y)
            btn_reset = gr.Button("Reset")

        with gr.Column():
            img_canvas = gr.Image(FACE_CANVAS, label="Face Canvas")
            img_face = gr.Image(FACE_IMG, label="Face Image")

    for slider in sliders:
        slider.change(apply, inputs=[current, beta], outputs=[img_canvas, img_face])

    force_update.click(apply, inputs=[current, beta], outputs=[img_canvas, img_face])
    save_locs.click(save_locations, inputs=current)

    current.change(apply, inputs=[current, beta], outputs=[img_canvas, img_face])
    beta.change(apply, inputs=[current, beta], outputs=[img_canvas, img_face])

    btn_reset.click(reset)
    btn_reset.click(apply, inputs=[current, beta], outputs=[img_canvas, img_face])
    demo.load(fn=apply, inputs=[current, beta], outputs=[img_canvas, img_face])

if __name__ == "__main__":
    demo.launch(show_api=False, title="sEMG Sensor Positioning")
