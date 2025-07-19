import os
import io
import base64
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
from openai import OpenAI

class OpenAIImageFidelityFashion:
    """
    ComfyUI Node for OpenAI Image 1 with High Input Fidelity
    Specialized for fashion and product photography use cases
    """
    
    def __init__(self):
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize OpenAI client with API key"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Change the outfit color to blue while preserving all details and textures"
                }),
                "primary_image": ("IMAGE",),
                "input_fidelity": (["high", "low"], {"default": "high"}),
                "quality": (["auto", "low", "medium", "high"], {"default": "high"}),
                "size": (["auto", "1024x1024", "1024x1536", "1536x1024"], {"default": "auto"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "background": (["auto", "opaque", "transparent"], {"default": "auto"}),
                "fashion_preset": ([
                    "custom",
                    "outfit_variation", 
                    "accessory_addition",
                    "product_extraction",
                    "color_change",
                    "style_transfer",
                    "background_change"
                ], {"default": "custom"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "revised_prompt")
    FUNCTION = "generate_fashion_image"
    CATEGORY = "OpenAI/Fashion"
    
    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # Handle batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # Convert from tensor format [H, W, C] to numpy
        np_image = tensor.cpu().numpy()
        
        # Ensure values are in [0, 255] range
        if np_image.max() <= 1.0:
            np_image = np_image * 255.0
        
        np_image = np_image.astype(np.uint8)
        
        # Convert to PIL Image
        if np_image.shape[2] == 3:  # RGB
            pil_image = Image.fromarray(np_image, 'RGB')
        elif np_image.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(np_image, 'RGBA')
        else:
            # Handle grayscale or other formats
            pil_image = Image.fromarray(np_image[:,:,0], 'L')
        
        return pil_image
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        # Ensure RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor with batch dimension [1, H, W, C]
        tensor = torch.from_numpy(np_image)[None,]
        
        return tensor
    
    def pil_to_base64(self, pil_image, format="PNG"):
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def get_fashion_prompt(self, preset, custom_prompt):
        """Get optimized prompts for fashion use cases"""
        fashion_prompts = {
            "outfit_variation": "Change the outfit while preserving the model's pose, facial features, and body proportions. Maintain fabric textures and realistic lighting.",
            "accessory_addition": "Add the accessory to the outfit while maintaining the original pose, lighting, and all existing details of the clothing and model.",
            "product_extraction": "Extract this exact product/garment and place it on a clean, professional background while preserving all details, textures, and colors.",
            "color_change": "Change only the color of the specified garment while preserving all textures, patterns, fabric details, and the overall composition.",
            "style_transfer": "Transform the clothing style while maintaining the model's pose, facial features, and the overall composition of the image.",
            "background_change": "Change only the background while preserving the model, outfit, pose, lighting, and all clothing details exactly as they are."
        }
        
        if preset == "custom":
            return custom_prompt
        else:
            base_prompt = fashion_prompts.get(preset, custom_prompt)
            return f"{base_prompt}. {custom_prompt}" if custom_prompt.strip() else base_prompt
    
    def prepare_images_for_api(self, primary_image, reference_image=None, mask_image=None):
        """Prepare images for OpenAI API call"""
        images = []
        
        # Convert primary image
        primary_pil = self.tensor_to_pil(primary_image)
        primary_base64 = self.pil_to_base64(primary_pil, "PNG")
        images.append(f"data:image/png;base64,{primary_base64}")
        
        # Add reference image if provided
        if reference_image is not None:
            ref_pil = self.tensor_to_pil(reference_image)
            ref_base64 = self.pil_to_base64(ref_pil, "PNG")
            images.append(f"data:image/png;base64,{ref_base64}")
        
        return images, mask_image
    
    def generate_fashion_image(self, prompt, primary_image, input_fidelity="high", 
                              quality="high", size="auto", output_format="png", 
                              background="auto", fashion_preset="custom", 
                              reference_image=None, mask_image=None, api_key=""):
        
        # Use provided API key or environment variable
        if api_key.strip():
            try:
                client = OpenAI(api_key=api_key.strip())
            except Exception as e:
                raise Exception(f"Error with provided API key: {e}")
        else:
            if not self.client:
                raise Exception("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable or provide API key.")
            client = self.client
        
        try:
            # Get optimized prompt for fashion use case
            final_prompt = self.get_fashion_prompt(fashion_preset, prompt)
            
            # Prepare images
            images, processed_mask = self.prepare_images_for_api(primary_image, reference_image, mask_image)
            
            # Prepare API call parameters
            api_params = {
                "model": "gpt-image-1",
                "prompt": final_prompt,
                "input_fidelity": input_fidelity,
                "quality": quality,
                "size": size,
                "output_format": output_format,
                "response_format": "b64_json"
            }
            
            # Add background parameter if not auto
            if background != "auto":
                api_params["background"] = background
            
            # Handle single image edit vs multi-image generation
            if len(images) == 1 and processed_mask is None:
                # Simple image edit
                primary_pil = self.tensor_to_pil(primary_image)
                buffer = io.BytesIO()
                primary_pil.save(buffer, format="PNG")
                buffer.seek(0)
                
                response = client.images.edit(
                    image=buffer,
                    **api_params
                )
            else:
                # Multi-image or masked edit
                if processed_mask is not None:
                    mask_pil = self.tensor_to_pil(processed_mask)
                    mask_buffer = io.BytesIO()
                    mask_pil.save(mask_buffer, format="PNG")
                    mask_buffer.seek(0)
                    api_params["mask"] = mask_buffer
                
                primary_pil = self.tensor_to_pil(primary_image)
                buffer = io.BytesIO()
                primary_pil.save(buffer, format="PNG")
                buffer.seek(0)
                
                response = client.images.edit(
                    image=buffer,
                    **api_params
                )
            
            # Process response
            if hasattr(response, 'data') and len(response.data) > 0:
                result_data = response.data[0]
                
                # Get image data
                if hasattr(result_data, 'b64_json'):
                    image_base64 = result_data.b64_json
                else:
                    raise Exception("No image data in response")
                
                # Get revised prompt if available
                revised_prompt = getattr(result_data, 'revised_prompt', final_prompt)
                
                # Decode and convert image
                image_bytes = base64.b64decode(image_base64)
                result_image = Image.open(io.BytesIO(image_bytes))
                
                # Convert back to ComfyUI tensor
                result_tensor = self.pil_to_tensor(result_image)
                
                return (result_tensor, revised_prompt)
            else:
                raise Exception("No data received from OpenAI API")
                
        except Exception as e:
            print(f"Error in fashion image generation: {e}")
            # Return original image and error message
            return (primary_image, f"Error: {str(e)}")

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "OpenAIImageFidelityFashion": OpenAIImageFidelityFashion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageFidelityFashion": "OpenAI Image Fidelity (Fashion)"
}