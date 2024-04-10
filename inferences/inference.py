from typing import Literal, Sequence, Mapping, Any, Union
import torch
from functools import lru_cache
import logging

from nodes import (
    VAEEncode,
    KSampler,
    KSamplerAdvanced,
    CheckpointLoaderSimple,
    VAEDecode,
    CLIPTextEncode,
    EmptyLatentImage,
    LoraLoader,
    SaveImage,
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


@lru_cache(maxsize=1)
def load_refiner():
    return checkpointloadersimple.load_checkpoint(
        ckpt_name="sdXL_v10RefinerVAEFix.safetensors"
    )


@lru_cache(maxsize=1)
def load_base_model(mode):
    # ckpt_name=base_checkpoints[mode]
    # print(ckpt_name)
    ckpt_name=base_checkpoints['anime']
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


# SDXL_Refiner_v10 = checkpointloadersimple.load_checkpoint(
#     ckpt_name="sdXL_v10RefinerVAEFix.safetensors"
# )

DEFAULT_PROMPT = "(1girl:2,masterpiece,beautiful eyes:1.2,beautiful body:0.6,slender body),floral dress,no hands"
DEFAULT_NEGATIVE_PROMPT = "(bad hands:5,bad arms:4,deformities:1.3),(extra limbs:2,extra arms:2,fused fingers:5,out of focus:3.0, deformed:2, disfigured:2),badpic,bad anatomy,"


def generate_image(
    mode: Union[Literal["anime"], Literal["realistic"], Literal["cartoon"]] = "anime",
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    batch_size=4,
    width=1024,
    height=1024,
    seed=420,
):
    with torch.inference_mode():
        SDXL_BaseModel = load_base_model(mode)
        SDXL_Lightning_Lora = load_lora_model(SDXL_BaseModel)

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_5 = emptylatentimage.generate(
            width=width, height=height, batch_size=batch_size
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_22 = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(SDXL_Lightning_Lora, 1),
        )

        cliptextencode_23 = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(SDXL_Lightning_Lora, 1),
        )

        SDXL_Refiner_v10 = load_refiner()

        cliptextencode_31 = cliptextencode.encode(
            text=prompt,
            clip=get_value_at_index(SDXL_Refiner_v10, 1),
        )

        cliptextencode_32 = cliptextencode.encode(
            text=negative_prompt,
            clip=get_value_at_index(SDXL_Refiner_v10, 1),
        )

        ksampler = KSampler()
        ksampler_3 = ksampler.sample(
            seed=seed,
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="sgm_uniform",
            denoise=1,
            model=get_value_at_index(SDXL_Lightning_Lora, 0),
            positive=get_value_at_index(cliptextencode_22, 0),
            negative=get_value_at_index(cliptextencode_23, 0),
            latent_image=get_value_at_index(emptylatentimage_5, 0),
        )

        vaedecode = VAEDecode()
        vaedecode_78 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(SDXL_BaseModel, 2),
        )

        vaeencode = VAEEncode()
        vaeencode_80 = vaeencode.encode(
            pixels=get_value_at_index(vaedecode_78, 0),
            vae=get_value_at_index(SDXL_BaseModel, 2),
        )

        ksampleradvanced = KSamplerAdvanced()
        ksampleradvanced_29 = ksampleradvanced.sample(
            add_noise="disable",
            noise_seed=seed,
            steps=9,
            cfg=1,
            sampler_name="euler",
            scheduler="sgm_uniform",
            start_at_step=8,
            end_at_step=10000,
            return_with_leftover_noise="disable",
            model=get_value_at_index(SDXL_Refiner_v10, 0),
            positive=get_value_at_index(cliptextencode_31, 0),
            negative=get_value_at_index(cliptextencode_32, 0),
            latent_image=get_value_at_index(vaeencode_80, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampleradvanced_29, 0),
            vae=get_value_at_index(SDXL_Refiner_v10, 2),
        )

        output_images = get_value_at_index(vaedecode_8, 0)
        return output_images


if __name__ == "__main__":
    output_images = generate_image()
    saveImage = SaveImage()
    saveImage.save_images(images=output_images, filename_prefix="Test_Inference")
