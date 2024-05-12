# ComfyUI Gaffer

comfyui's gaffer

## Important updates

* 2024-05-12: ComfyUI native implementation of [IC-Light](https://github.com/lllyasviel/IC-Light).

## Install

* Recommended use [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) for installation.

* Or open the cmd window in the plugin directory of ComfyUI, like ```ComfyUI\custom_nodes```，type    
  ```
  git clone https://github.com/huagetai/ComfyUI-Gaffer.git
  ```
* Or download the zip file and extracted, copy the resulting folder to ```ComfyUI\custom_ Nodes```
* Download models

  You can download models here: https://huggingface.co/lllyasviel/ic-light/tree/main
  There are 3 models:
  * iclight_sd15_fc.safetensors: Use this in FG workflows
  - iclight_sd15_fcon.safetensors: Use this in FG workflows
  - iclight_sd15_fbc.safetensors: Use this in BG workflows

  After you download these models, please put them under ComfyUI/models/iclight.

* Restart ComfyUI.

## Usage

![ALL Model](image/all.png)


* Text-Conditioned Model

  * refer to: [Text + foreground Conditioned](examples/workflow-gaffer-iclight-fc-text.json)

  * refer to: [Text + foreground + Lighting Preference Conditioned](examples/workflow-gaffer-iclight-fc.json)

* Background-Conditioned Model

  * refer to: [Text + foreground + background Conditioned](examples/workflow-gaffer-iclight-fbc.json)

* Calculate Normal
  
  * refer to: [Calculate Normal](examples/workflow-gaffer-iclight-fc-normal.json)

* tips

  Model description and input foreground and background images can be obtained [IC-Light](https://github.com/lllyasviel/IC-Light)

## Nodes


* **Load ICLight Model Node**

  Load ICLight Model

  ![Load ICLight Model](image/node_load-iclight-model.png)

  input:
  * iclight_name: The name of the ICLight model to load.
  
  output:
  * iclight: ICLight model information.


* **Apply ICLight Node**

  Apply ICLight

  ![Apply ICLight](image/node_apply-iclight.png)

  input:
  * model: base model
  * vae: VAE
  * iclight: ICLight model information.
  * position: position prompts
  * negative: negative prompts
  * fg_pixels: foreground image.The background of the foreground image needs to be removed, and it is recommended to set it to a grayscale background.
  * bg_pixels: background image(optional).Required for Background-Conditioned Model.Consistent with the width and height of the foreground image
  
  output:
  * model: model with ICLight
  * position: position prompts
  * negative: negative prompts
  * empty_latent: Empty Latent Image.The width and height are consistent with the foreground image.


* **Simple Light Source Node**
  
  Simple Light Source for Lighting Preference
  ![Simple Light Source Node](image/node_simple-light-source.png)

  input:
  * light_position: The name of the ICLight model to load.
  * multiplier: strength of Lighting Preference
  * start_color: start color of Lighting Preference
  * end_color: end color of Lighting Preference
  * width: The width of Lighting Preference
  * height: The height of Lighting Preference
  tips:Consistent with the width and height of the foreground image
  
  output:
  * image: Lighting Preference.
  

* **Calculate Normal Map Node**

  Calculate Normal Map

  ![Calculate Normal Map](image/node_calculate_normal_map.png)

  input:
  * images: The name of the ICLight model to load. 
  * mask: The name of the ICLight model to load.
  * sigma: The name of the ICLight model to load.
  * center_input_range:
  
  output:
  * normal: normal map
  
  **tips**:The node from [kijai-ComfyUI-IC-Light](https://github.com/kijai/ComfyUI-IC-Light) project

## Credits

* [IC-Light](https://github.com/lllyasviel/IC-Light)
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
* [kijai-ComfyUI-IC-Light](https://github.com/kijai/ComfyUI-IC-Light)
* [huchenlei-ComfyUI-IC-Light-Native](https://github.com/huchenlei/ComfyUI-IC-Light-Native)