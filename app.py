import os

import gradio as gr

from src.run.unet.inference import ResUnetInfer


infer = ResUnetInfer(
    model_path="./checkpoint/resunet/decoder.pt",
    config_path="./src/models/unet/config/resnet_config.yml",
)

demo = gr.Interface(
    fn=infer.infer,
    inputs=[
        gr.Image(
            shape=(224, 224),
            label="Input Image",
            value="./sample/bird_plane.jpeg",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            label="Mask Transparency",
            info="Mask opacity for image segmentation overlay",
        ),
    ],
    outputs=[
        gr.Image(),
    ],
    examples=[
        [os.path.join("./sample/", f)]
        for f in os.listdir("./sample/")
    ],
)


demo.launch()