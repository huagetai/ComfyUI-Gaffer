import os
import logging
from typing import TypedDict, Callable
import torch
import numpy as np
import folder_paths
import comfy.model_management as model_management
from comfy.sd import VAE
from comfy.utils import load_torch_file
from comfy.diffusers_convert import convert_unet_state_dict
from comfy.ldm.models.autoencoder import AutoencoderKL
from comfy.model_base import BaseModel, IP2P
from comfy.model_patcher import ModelPatcher
from nodes import MAX_RESOLUTION
from .utils.image import generate_gradient_image, LightPosition, resize_and_center_crop
from .utils.patches import calculate_weight_adjust_channel

# logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# contents
MODEL_TYPE_ICLIGHT = "iclight"

# set the models directory
if MODEL_TYPE_ICLIGHT not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, MODEL_TYPE_ICLIGHT)]
else:
    current_paths, _ = folder_paths.folder_names_and_paths[MODEL_TYPE_ICLIGHT]
folder_paths.folder_names_and_paths[MODEL_TYPE_ICLIGHT] = (current_paths, folder_paths.supported_pt_extensions)


class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor

class ICLightPatcher():
    def __init__(self, model: ModelPatcher, vae: VAE, iclight):
        modelType = str(type(model.model.model_config).__name__)
        if "SD15" not in modelType:
            raise Exception(f"Attempted to load {modelType} model, IC-Light is only compatible with SD 1.5 models.")
        assert isinstance(vae.first_stage_model, AutoencoderKL), "vae only supported for AutoencoderKL"
        self.vae = vae

        self.device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if model_management.should_use_fp16() else torch.float32
        self.dtype = dtype
        self.work_model = model.clone()

        # Patch ComfyUI's LoRA weight application to accept multi-channel inputs.
        try:
            ModelPatcher.calculate_weight = calculate_weight_adjust_channel(ModelPatcher.calculate_weight)
        except:
            raise Exception("Could not patch calculate_weight")

        self.fbc = iclight.get("fbc", False)
        self.sd_dict = iclight.get("sd_dict", {})
        in_channels = self.sd_dict["input_blocks.0.0.weight"].shape[1]
        self.work_model.model.model_config.unet_config["in_channels"] = in_channels

        #  add iclight patches
        self.add_patches_iclight()
    
    def encode(self, pixels):
        original_sample_mode = self.vae.first_stage_model.regularization.sample
        self.vae.first_stage_model.regularization.sample = False
        out_samples = self.vae.encode(pixels)
        self.vae.first_stage_model.regularization.sample = original_sample_mode
        return out_samples
    
    def set_concat_conds(self, fg_pixels, bg_pixels=None):
        self.fg_samples = self.encode(fg_pixels)
        if self.fbc:
            if bg_pixels is None:
                raise ValueError("When using background-conditioned Model, 'bg_pixel' is required")

            bg_pixels = resize_and_center_crop(bg_pixels, fg_pixels.shape[2], fg_pixels.shape[1])
            self.bg_samples = self.encode(bg_pixels)
            concat_samples = torch.cat((self.fg_samples, self.bg_samples), dim=1)
        else:
            concat_samples = self.fg_samples

        base_model: BaseModel = self.work_model.model
        scale_factor = base_model.model_config.latent_format.scale_factor
        concat_conds = concat_samples * scale_factor
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        self.concat_conds = concat_conds
        self.set_model_unet_function_wrapper()

    def add_patches_iclight(self):
        ic_model_state_dict = self.sd_dict
        self.work_model.add_patches(
            patches={
                ("diffusion_model." + key): (value.to(dtype=self.dtype, device=self.device),)
                for key, value in ic_model_state_dict.items()
            }
        )

    def set_model_unet_function_wrapper(self):
        concat_conds = self.concat_conds
        def apply_c_concat(params: UnetParams) -> UnetParams:
            """Apply c_concat on unet call."""
            sample = params["input"]
            params["c"]["c_concat"] = torch.cat(
                ([concat_conds.to(sample.device)] * (sample.shape[0] // concat_conds.shape[0])),
                dim=0,
            )
            return params

        def unet_dummy_apply(unet_apply: Callable, params: UnetParams):
            """A dummy unet apply wrapper serving as the endpoint of wrapper chain."""
            return unet_apply(x=params["input"], t=params["timestep"], **params["c"])

        # Compose on existing `model_function_wrapper`.
        existing_wrapper =  self.work_model.model_options.get("model_function_wrapper", unet_dummy_apply)

        def wrapper_func(unet_apply: Callable, params: UnetParams):
            return existing_wrapper(unet_apply, params=apply_c_concat(params))

        self.work_model.set_model_unet_function_wrapper(wrapper_func)


class ICLightModelLoader:
    """
    Class for loading and managing the weights of ICLight models.
    """

    def __init__(self):
        """
        Initializes the ICLight model loader with default state.
        """
        self.iclight = {"model": None, "sd_dict": None, "fbc": False} 

    @classmethod
    def INPUT_TYPES(cls):
        """
        Class method defining the required input types for model loading.

        Returns:
            dict: A dictionary specifying the required input types for the model load operation.
        """
        return {"required": {"iclight_name": (folder_paths.get_filename_list(MODEL_TYPE_ICLIGHT),)}}

    RETURN_TYPES = ("ICLIGHT",)
    RETURN_NAMES = ("iclight",)
    FUNCTION = "load_iclight_model"
    CATEGORY = "gaffer"

    def load_iclight_model(self, iclight_name):
        """
        Loads the weights of an ICLight model.

        Parameters:
            iclight_name (str): The name of the ICLight model to load.

        Returns:
            tuple: A tuple containing the loaded ICLight model information.

        Raises:
            Exception: If the specified ICLight model file is not found.
        """
        LOGGER.info("Loading ICLight Model weights")

        # Safely retrieves the full path of the model and checks for its existence
        ckpt_path = folder_paths.get_full_path(MODEL_TYPE_ICLIGHT, iclight_name)
        if not os.path.exists(ckpt_path):
            error_message = f"Invalid ICLight model: {iclight_name}. File not found."
            LOGGER.error(error_message)
            raise Exception(error_message)

        # Loads the model securely
        model = load_torch_file(ckpt_path, safe_load=True)

        # converts it to a hardware-optimized precision
        sd_dict = convert_unet_state_dict(model)

        # Optimizes performance by converting the model's precision on demand
        sd_dict = {key: sd_dict[key].half() for key in sd_dict.keys()}

        # Updates the internal state to reflect the loaded model details
        self.iclight['model'] = model
        self.iclight['sd_dict'] = sd_dict
        self.iclight['fbc'] = 'fbc' in iclight_name.lower()

        LOGGER.info(f"ICLight model loaded from {iclight_name}")
        return (self.iclight,)
    

class ApplyICLight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "iclight": ("ICLIGHT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "fg_pixels": ("IMAGE",),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.001}),
            },
            "optional": {
                "bg_pixels": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "empty_latent")
    FUNCTION = "apply"
    CATEGORY = "gaffer"
    DESCRIPTION = """"""

    def apply(self, model: ModelPatcher, vae: VAE, iclight, positive, negative, fg_pixels, multiplier, bg_pixels=None):
        iclightPatcher = ICLightPatcher(model, vae, iclight)

        iclightPatcher.set_concat_conds(fg_pixels, bg_pixels)

        out_latent = torch.zeros_like(iclightPatcher.fg_samples)

        return (iclightPatcher.work_model, positive, negative, {"samples": out_latent})


class LightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "light_position": ([member.value for member in LightPosition],),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001}),
                "start_color": ("STRING", {"default": "#FFFFFF"}),
                "end_color": ("STRING", {"default": "#000000"}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "gaffer"
    DESCRIPTION = """
Generates a gradient image that can be used  
as a simple light source.  The color can be  
specified in RGB or hex format.  
"""

    def execute(self, light_position, multiplier, start_color, end_color, width, height):
        def toRgb(color):
            if color.startswith('#') and len(color) == 7:  # e.g. "#RRGGBB"
                color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            else:  # e.g. "255,255,255"
                color_rgb = tuple(int(i) for i in color.split(','))
            return color_rgb

        lightPosition = LightPosition(light_position)
        start_color_rgb = toRgb(start_color)
        end_color_rgb = toRgb(end_color)
        image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)


class CalculateNormalMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sigma": ("FLOAT", {"default": 10.0, "min": 0.01, "max": 100.0, "step": 0.01, }),
                "center_input_range": ("BOOLEAN", {"default": False, }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "execute"
    CATEGORY = "gaffer"
    DESCRIPTION = """
    Calculates normal map from different directional exposures.  
    Takes in 4 images as a batch:  
    left, right, bottom, top  
    """

    def execute(self, images, sigma, center_input_range, mask=None):
        """
        Executes an image processing algorithm to calculate and return a normal map.

        Parameters:
        images - A sequence of input images.
        sigma - The standard deviation for Gaussian blur, controlling the smoothness of the normal calculation.
        center_input_range - The range used to center the input images.
        mask - Optional, a mask image to specify the computation region.

        Returns:
        normal - The calculated normal map.
        """

        # Input validation
        self.validate_inputs(images, sigma, center_input_range, mask)

        # Centering images to adjust the brightness levels to have a mean closer to zero
        images = self.center_images(images, center_input_range)

        # Compute the normal based on image intensities and Gaussian blur
        normal = self.calculate_normal(images, sigma, mask)

        # Normalize and clamp the normal values to ensure they stay within valid ranges
        normal = self.normalize_and_clamp(normal)

        return (normal,)

    def validate_inputs(self, images, sigma, center_input_range, mask):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be a 4-dimensional torch.Tensor")
        if not (0.01 <= sigma <= 100.0):
            raise ValueError("sigma must be between 0.01 and 100.0")
        if not isinstance(center_input_range, bool):
            raise ValueError("center_input_range must be a boolean")
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.ndim != 3):
            raise ValueError("mask must be a 3-dimensional torch.Tensor")

    def center_images(self, images, center_input_range):
        if center_input_range:
            images = images * 0.5 + 0.5
        return images

    def calculate_normal(self, images, sigma, mask):
        images_np = images.numpy().astype(np.float32)
        left = images_np[0]
        right = images_np[1]
        bottom = images_np[2]
        top = images_np[3]
        ambient = (left + right + bottom + top) / 4.0
        height, width, _ = ambient.shape

        def safe_divide(a, b):
            epsilon = 1e-5
            return ((a + epsilon) / (b + epsilon)) - 1.0

        left = safe_divide(left, ambient)
        right = safe_divide(right, ambient)
        bottom = safe_divide(bottom, ambient)
        top = safe_divide(top, ambient)

        u = (right - left) * 0.5
        v = (top - bottom) * 0.5

        u = np.mean(u, axis=2)
        v = np.mean(v, axis=2)
        h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
        z = np.zeros_like(h)

        normal = np.stack([u, v, h], axis=2)
        normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5

        if mask is not None:
            matting = mask.numpy().astype(np.float32)
            matting = matting[..., np.newaxis]
            normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)
            normal = torch.from_numpy(normal)
        else:
            normal += np.stack([z, z, 1 - z], axis=2)
            normal = torch.from_numpy(normal).unsqueeze(0)

        return normal

    def normalize_and_clamp(self, normal):
        normal = (normal + 1.0) / 2.0
        normal = torch.clamp(normal, 0, 1)
        return normal


class GrayScaler:
    """Class to scale image regions to grey based on provided masks."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.001}),
            }
        }

    CATEGORY = "gaffer"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    def apply(self, image: torch.Tensor, mask: torch.Tensor, multiplier):
        """
        Applies grey scaling to image regions as indicated by the mask.

        Args:
            image: torch.Tensor: The input image tensor.
            mask: torch.Tensor: A mask tensor where 1 indicates areas to be converted to grey.
            multiplier: float: A value to control the intensity of the grey conversion.
        Returns:
            tuple: A tuple containing the image with masked regions turned grey.
        """
        # Validate inputs
        if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("image and mask must be torch.Tensor types.")
        if image.ndim != 4 or mask.ndim not in [3, 4]:
            raise ValueError("image must be a 4D tensor, and mask must be a 3D or 4D tensor.")

        # Adjust mask dimensions if necessary
        if mask.ndim == 3:
            # [B, H, W] => [B, H, W, C=1]
            mask = mask.unsqueeze(-1)

        grey_value = 0.5 * multiplier
        image = image * mask + (1 - mask) * grey_value

        return (image,)


NODE_CLASS_MAPPINGS = {
    "ICLightModelLoader": ICLightModelLoader,
    "ApplyICLight": ApplyICLight,
    "LightSource": LightSource,
    "CalculateNormalMap": CalculateNormalMap,
    "GrayScaler": GrayScaler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ICLightModelLoader": "Load ICLight Model",
    "ApplyICLight": "Apply ICLight",
    "LightSource": "Simple Light Source",
    "CalculateNormalMap": "Calculate Normal Map",
    "GrayScaler": "Gray Scaler",
}
