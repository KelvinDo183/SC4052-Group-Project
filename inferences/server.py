from fastapi import FastAPI, Request, Response, Query
import io
from functools import lru_cache
from inference import generate_image, image2image
from PIL import Image
import numpy as np

app = FastAPI()


@lru_cache(maxsize=None)
def cached_generate_image(
    prompt,
    negative_prompt,
    mode,
    batch_size,
    width,
    height,
    seed,
    base_step,
    refiner_step,
    cfg,
):
    return generate_image(
        prompt,
        negative_prompt,
        mode,
        batch_size,
        width,
        height,
        seed,
        base_step,
        refiner_step,
        cfg,
    )


@app.post("/text2img")
async def text2img(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    negative_prompt = body.get("negative_prompt")
    batch_size = body.get("batch_size")
    width = body.get("width")
    height = body.get("height")
    seed = body.get("seed")
    mode = body.get("mode")
    base_step = body.get("base_step")
    refiner_step = body.get("refiner_step")
    cfg = body.get("cfg")

    # NOTE: list of numpy tensors
    output_images = cached_generate_image(
        prompt,
        negative_prompt,
        mode,
        batch_size,
        width,
        height,
        seed,
        base_step,
        refiner_step,
        cfg,
    )

    # Convert the images to bytes and join them with a delimiter
    delimiter = b"--DELIMITER--"
    image_bytes_list = []
    for image in output_images:
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        image_bytes_list.append(image_bytes.getvalue())

    image_bytes_stream = delimiter.join(image_bytes_list)

    return Response(content=image_bytes_stream, media_type="application/octet-stream")


def cached_img2img(prompt, image, negative_prompt, mode, seed):
    return image2image(
        prompt,
        image,
        negative_prompt,
        mode,
        seed,
    )


@app.post("/img2img")
async def img2img(
    request: Request,
    prompt: str = Query(...),
    negative_prompt: str = Query(...),
    seed: int = Query(...),
    mode: str = Query(...),
):
    # Read the raw bytes from the request body
    image_bytes = await request.body()

    # Create a PIL Image object from the image bytes
    image = Image.open(io.BytesIO(image_bytes))

    # NOTE: list of numpy tensors
    output_images = cached_img2img(
        prompt,
        image,
        negative_prompt,
        mode,
        seed,
    )

    # Convert the images to bytes and join them with a delimiter
    delimiter = b"--DELIMITER--"
    image_bytes_list = []
    for image in output_images:
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        image_bytes_list.append(image_bytes.getvalue())
    image_bytes_stream = delimiter.join(image_bytes_list)

    return Response(content=image_bytes_stream, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
