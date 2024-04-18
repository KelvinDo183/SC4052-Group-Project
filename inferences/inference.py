from typing import Literal, Sequence, Mapping, Any, Union
import torch
from functools import lru_cache
from PIL import Image, ImageOps, ImageSequence
import numpy as np

from nodes import (
    VAEEncode,
    KSampler,
    KSamplerAdvanced,
    CheckpointLoaderSimple,
    VAEDecode,
    CLIPTextEncode,
    EmptyLatentImage,
    LoraLoader,
    LoraLoaderModelOnly,
    ImageScale,
    SaveImage,
    LoadImage
)


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


checkpointloadersimple = CheckpointLoaderSimple()
base_checkpoints = {
    "anime": "aamXLAnimeMix_v10.safetensors",
    "realistic": "realvisxlV40_v40LightningBakedvae.safetensors",
    "cartoon": "samaritan3dCartoon_v40SDXL.safetensors",
}

loraloader = LoraLoader()
loraloadermodel = LoraLoaderModelOnly()


@lru_cache(maxsize=1)
def load_refiner():
    return checkpointloadersimple.load_checkpoint(
        ckpt_name="sdXL_v10RefinerVAEFix.safetensors"
    )


@lru_cache(maxsize=1)
def load_base_model(mode):
    ckpt_name = base_checkpoints[mode]
    return checkpointloadersimple.load_checkpoint(ckpt_name=ckpt_name)


@lru_cache(maxsize=1)
def load_lora_model(base_model):
    return loraloader.load_lora(
        lora_name="sdxl_lightning_8step_lora.safetensors",
        strength_model=1,
        strength_clip=1,
        model=get_value_at_index(base_model, 0),
        clip=get_value_at_index(base_model, 1),
    )


@lru_cache(maxsize=1)
def load_lora_model_only(base_model):
    return loraloadermodel.load_lora_model_only(
        lora_name="sdxl_lightning_8step_lora.safetensors",
        strength_model=1,
        model=get_value_at_index(base_model, 0),
    )


DEFAULT_PROMPT = "(1girl:2),(beautiful eyes:1.2),beautiful girl,(hands in pocket:2),red hat,hoodie,computer"
DEFAULT_NEGATIVE_PROMPT = "(bad hands:5),(fused fingers:5),(bad arms:4),(deformities:1.3),(extra limbs:2),(extra arms:2),(disfigured:2)"


def generate_image(
    prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    mode: Union[Literal["anime"], Literal["realistic"], Literal["cartoon"]] = "anime",
    batch_size=4,
    width=1024,
    height=1024,
    seed=4052,
    base_step=8,
    refiner_step=1,
    cfg=1,
):
    with torch.inference_mode():
        SDXL_BaseModel = load_base_model(mode)
        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=width, height=height, batch_size=batch_size
        )

        cliptextencode = CLIPTextEncode()
        with_lightning_lora = mode != "realistic"
        SDXL_Lightning_Lora = None
        if with_lightning_lora:
            SDXL_Lightning_Lora = load_lora_model_only(SDXL_BaseModel)

        base_prompt = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(SDXL_BaseModel, 1),
        )
        base_negativve_prompt = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(SDXL_BaseModel, 1),
        )

        ksampler = KSampler()
        ksampler_3 = ksampler.sample(
            seed=seed,
            steps=base_step,
            cfg=cfg,
            sampler_name="euler",
            scheduler="sgm_uniform",
            denoise=1,
            model=get_value_at_index(
                (
                    SDXL_Lightning_Lora
                    if SDXL_Lightning_Lora is not None
                    else SDXL_BaseModel
                ),
                0,
            ),
            positive=get_value_at_index(base_prompt, 0),
            negative=get_value_at_index(base_negativve_prompt, 0),
            latent_image=get_value_at_index(emptylatentimage_5, 0),
        )

        vaedecode = VAEDecode()
        base_output = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(SDXL_BaseModel, 2),
        )

        vaeencode = VAEEncode()
        vaeencode_80 = vaeencode.encode(
            pixels=get_value_at_index(base_output, 0),
            vae=get_value_at_index(SDXL_BaseModel, 2),
        )

        SDXL_Refiner_v10 = load_refiner()

        refiner_prompt = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(SDXL_Refiner_v10, 1),
        )

        refiner_negative_prompt = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(SDXL_Refiner_v10, 1),
        )

        ksampleradvanced = KSamplerAdvanced()
        ksampleradvanced_29 = ksampleradvanced.sample(
            add_noise="disable",
            noise_seed=seed,
            steps=base_step + refiner_step,
            cfg=cfg,
            sampler_name="euler",
            scheduler="sgm_uniform",
            start_at_step=base_step,
            end_at_step=10000,
            return_with_leftover_noise="disable",
            model=get_value_at_index(SDXL_Refiner_v10, 0),
            positive=get_value_at_index(refiner_prompt, 0),
            negative=get_value_at_index(refiner_negative_prompt, 0),
            latent_image=get_value_at_index(vaeencode_80, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampleradvanced_29, 0),
            vae=get_value_at_index(SDXL_Refiner_v10, 2),
        )

        output_images = get_value_at_index(vaedecode_8, 0)
        return output_images


def load_image(img: Image.Image):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == "I":
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


def image2image(
    prompt,
    image: Image.Image,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    mode: Union[Literal["anime"], Literal["realistic"], Literal["cartoon"]] = "anime",
    seed=4052,
    denoise=0.7
):
    with torch.inference_mode():
        print("Input", prompt, negative_prompt, mode, seed)
        SDXL_BaseModel = load_base_model(mode)
        with_lightning_lora = mode != "realistic"
        SDXL_Lightning_Lora = None
        if with_lightning_lora:
            SDXL_Lightning_Lora = load_lora_model_only(SDXL_BaseModel)

        SDXL_Refiner_v10 = load_refiner()
        cliptextencode = CLIPTextEncode()
        base_prompt = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(SDXL_BaseModel, 1),
        )
        base_negative_prompt = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(SDXL_BaseModel, 1),
        )
        refiner_prompt = cliptextencode.encode(
            text=prompt, clip=get_value_at_index(SDXL_Refiner_v10, 1)
        )

        refiner_negative_prompt = cliptextencode.encode(
            text=negative_prompt, clip=get_value_at_index(SDXL_Refiner_v10, 1)
        )

        input_image = load_image(image)

        imagescale = ImageScale()
        lanczos_interpolated_image = imagescale.upscale(
            upscale_method="lanczos",
            width=image.width,
            height=image.height,
            crop="center",
            image=get_value_at_index(input_image, 0),
        )

        vaeencode = VAEEncode()
        vaeencode_52 = vaeencode.encode(
            pixels=get_value_at_index(lanczos_interpolated_image, 0),
            vae=get_value_at_index(SDXL_BaseModel, 2),
        )

        ksampler = KSampler()
        ksampleradvanced = KSamplerAdvanced()
        vaedecode = VAEDecode()
        saveImage = SaveImage()

        base_output = ksampler.sample(
            seed=seed,
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="sgm_uniform",
            denoise=denoise,
            model=get_value_at_index(
                (
                    SDXL_Lightning_Lora
                    if SDXL_Lightning_Lora is not None
                    else SDXL_BaseModel
                ),
                0,
            ),
            positive=get_value_at_index(base_prompt, 0),
            negative=get_value_at_index(base_negative_prompt, 0),
            latent_image=get_value_at_index(vaeencode_52, 0),
        )

        refiner_output = ksampleradvanced.sample(
            add_noise="disable",
            noise_seed=seed,
            steps=9,
            cfg=2,
            sampler_name="euler",
            scheduler="sgm_uniform",
            start_at_step=8,
            end_at_step=10000,
            return_with_leftover_noise="disable",
            model=get_value_at_index(SDXL_Refiner_v10, 0),
            positive=get_value_at_index(refiner_prompt, 0),
            negative=get_value_at_index(refiner_negative_prompt, 0),
            latent_image=get_value_at_index(base_output, 0),
        )

        decoded_output_image = vaedecode.decode(
            samples=get_value_at_index(refiner_output, 0),
            vae=get_value_at_index(SDXL_Refiner_v10, 2),
        )

        output_images = get_value_at_index(decoded_output_image, 0)
        saveImage.save_images(images=output_images, filename_prefix="Test_Inference")
        return output_images


if __name__ == "__main__":
    output_images = generate_image(prompt=DEFAULT_PROMPT)
    saveImage = SaveImage()
    saveImage.save_images(images=output_images, filename_prefix="Test_Inference")
