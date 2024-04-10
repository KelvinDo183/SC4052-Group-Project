from fastapi import FastAPI, Request, Response
import io
from functools import lru_cache
from inference import generate_image
from PIL import Image
import numpy as np

app = FastAPI()

@lru_cache(maxsize=None)
def cached_generate_image(mode, prompt, negative_prompt, batch_size, width, height, seed):
    return generate_image(mode, prompt, negative_prompt, batch_size, width, height, seed)

@app.post("/inference")
async def inference(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    negative_prompt = body.get("negative_prompt")
    batch_size = body.get("batch_size")
    width = body.get("width")
    height = body.get("height")
    seed = body.get("seed")
    mode = body.get("mode")

    # NOTE: list of numpy tensors
    output_images = cached_generate_image(
        mode, prompt, negative_prompt, batch_size, width, height, seed
    )

    # Convert the images to bytes and join them with a delimiter
    delimiter = b"--DELIMITER--"
    image_bytes_list = []
    for image in output_images:
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        image_bytes_list.append(image_bytes.getvalue())
    
    image_bytes_stream = delimiter.join(image_bytes_list)
    
    return Response(content=image_bytes_stream, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
