import modules.scripts as scripts
import gradio as gr
from fastapi import FastAPI, Body
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed

from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
import numpy as np
import cv2
from PIL import Image

def apply_canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def canny(img, res=512, thr_a=100, thr_b=200, **kwargs):
    l, h = thr_a, thr_b
    img = resize_image(HWC3(img), res)
    result = cv2.Canny(img, thr_a, thr_b)
    return result

class Script(scripts.Script):

    def title(self):
        return "Figma"

    def show(self, is_img2img):
        return cmd_opts.allow_code

    def ui(self, is_img2img):
        annotator_resolution=gr.inputs.Slider(label="Annotator resolution", value=512, minimum=64, maximum=2048, step=1, interactive=True)
        canny_low_threshold=gr.inputs.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1, interactive=True)
        canny_high_threshold=gr.inputs.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1, interactive=True)
        with gr.Blocks() as demo:
            with gr.Row().style(equal_height=True):
                image=gr.Image(type="pil")
                mask=gr.Image(type="pil")
        btn = gr.Button(value="Preview Canny")
        if image is not None:
             btn.click(canny,inputs=[image,annotator_resolution,canny_low_threshold,canny_high_threshold],outputs=[mask])
        return [annotator_resolution,canny_low_threshold,canny_high_threshold]

    def run(self, p, annotator_resolution,canny_low_threshold,canny_high_threshold):
        proc = process_images(p)
        return proc
    
def figma_api(_: gr.Blocks, app: FastAPI):
    @app.get("/figma/status")
    async def get_status():
        return {"status": "ok", "version": "1.0.0"}
    @app.post("/figma/canny")
    async def post_canny(image_str: str = Body(...), annotator_resolution: int = Body(...), canny_low_threshold: int = Body(...), canny_high_threshold: int = Body(...)):
        import base64
        import io
        image_bytes = base64.b64decode(image_str)
        image=cv2.cvtColor(np.array(Image.open(io.BytesIO(image_bytes),formats=["PNG"]).convert('RGB')), cv2.COLOR_RGB2BGR)
        c_img=canny(image, annotator_resolution, canny_low_threshold, canny_high_threshold)
        _, buffer = cv2.imencode('.png', c_img)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return {"image": base64_image}
    
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(figma_api)
except:
    pass
