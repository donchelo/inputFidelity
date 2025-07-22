import os
import io
import base64
import requests  # NUEVO: Para Airtable
import json      # NUEVO: Para Airtable
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
from openai import OpenAI

# NUEVO: Función para cargar config.env
def load_config_file():
    """Busca y carga config.env desde múltiples ubicaciones"""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'config.env'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.env'),
        os.path.expanduser('~/comfyui_config.env'),
    ]
    
    for config_path in possible_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                print(f"✅ Config cargado desde: {config_path}")
                return True
            except Exception as e:
                print(f"⚠️ Error leyendo {config_path}: {e}")
                continue
    
    # Intentar usar variables de entorno del sistema
    if os.getenv('OPENAI_API_KEY') and os.getenv('AIRTABLE_API_KEY'):
        print("✅ Usando variables de entorno del sistema")
        return True
        
    print("⚠️ No se encontró configuración")
    return False

class OpenAIImageFidelityFashion:
    """
    ComfyUI Node for OpenAI Image 1 with High Input Fidelity
    Specialized for fashion and product photography use cases
    NOW WITH ITERATION CONTROL FOR clienteTRUE
    """
    
    def __init__(self):
        self.client = None
        # NUEVO: Variables para control de iteraciones
        self.client_id = "clienteTRUE"
        self.table_name = "inputFidelity"
        # Cargar configuración
        load_config_file()
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
    
    # NUEVO: Función para control de iteraciones
    def check_and_consume_iteration(self, api_key, base_id):
        """Verifica y consume 1 iteración para clienteTRUE"""
        if not api_key or not base_id:
            return False, "❌ Credenciales de Airtable requeridas", 0
        
        try:
            url = f"https://api.airtable.com/v0/{base_id}/{self.table_name}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "filterByFormula": f"{{id_cliente}} = '{self.client_id}'"
            }
            
            # Consultar iteraciones actuales
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                return False, f"❌ Error consultando Airtable: {response.status_code}", 0
            
            data = response.json()
            
            if not data.get('records'):
                return False, f"❌ Cliente '{self.client_id}' no encontrado", 0
            
            # Obtener iteraciones actuales
            record = data['records'][0]
            record_id = record['id']
            current_iterations = record['fields'].get('iteraciones_restantes', 0)
            
            # Verificar disponibilidad
            if current_iterations <= 0:
                return False, "🚫 Sin iteraciones disponibles. Contacta al administrador para recargar.", 0
            
            # Decrementar iteraciones
            new_iterations = current_iterations - 1
            
            # Actualizar en Airtable
            update_url = f"{url}/{record_id}"
            update_data = {
                "fields": {
                    "iteraciones_restantes": new_iterations
                }
            }
            
            update_response = requests.patch(update_url, headers=headers, json=update_data)
            
            if update_response.status_code != 200:
                return False, f"❌ Error actualizando iteraciones: {update_response.status_code}", current_iterations
            
            return True, f"✅ Uso autorizado. Quedan {new_iterations} iteraciones", new_iterations
            
        except Exception as e:
            return False, f"❌ Error de conexión: {str(e)}", 0
    
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
                "api_method": (["images_api", "responses_api"], {"default": "images_api"}),
                
                # NUEVO: Control de iteraciones
                "enable_iterations_control": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "mask_image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                
                # NUEVO: Credenciales Airtable opcionales
                "airtable_api_key": ("STRING", {"default": ""}),
                "airtable_base_id": ("STRING", {"default": ""}),
            }
        }
    
    # NUEVO: Agregar remaining_iterations a las salidas
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "revised_prompt", "debug_info", "remaining_iterations")
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
    
    def edit_with_images_api(self, client, primary_image, mask_image, prompt, 
                           input_fidelity, quality, size, output_format, background):
        """Edit using OpenAI Images API - for single image editing"""
        try:
            # Convert primary image to buffer
            primary_pil = self.tensor_to_pil(primary_image)
            image_buffer = io.BytesIO()
            primary_pil.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            
            # Prepare API parameters
            params = {
                "model": "gpt-image-1",
                "image": image_buffer,
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "response_format": "b64_json"
            }
            
            # Add input_fidelity parameter
            if input_fidelity == "high":
                params["input_fidelity"] = "high"
            
            # Add output format if not auto
            if output_format != "auto":
                params["output_format"] = output_format
                
            # Add background if not auto
            if background != "auto":
                params["background"] = background
            
            # Add mask if provided
            if mask_image is not None:
                mask_pil = self.tensor_to_pil(mask_image)
                mask_buffer = io.BytesIO()
                mask_pil.save(mask_buffer, format="PNG")
                mask_buffer.seek(0)
                params["mask"] = mask_buffer
            
            # Make API call
            response = client.images.edit(**params)
            
            return response, "images_api"
            
        except Exception as e:
            raise Exception(f"Images API error: {str(e)}")
    
    def edit_with_responses_api(self, client, primary_image, reference_image, prompt,
                              input_fidelity, quality, size, output_format, background):
        """Edit using OpenAI Responses API - for multi-image scenarios"""
        try:
            # Prepare content array
            content = [
                {"type": "input_text", "text": prompt}
            ]
            
            # Add primary image
            primary_pil = self.tensor_to_pil(primary_image)
            primary_base64 = self.pil_to_base64(primary_pil, "PNG")
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{primary_base64}"
            })
            
            # Add reference image if provided
            if reference_image is not None:
                ref_pil = self.tensor_to_pil(reference_image)
                ref_base64 = self.pil_to_base64(ref_pil, "PNG")
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{ref_base64}"
                })
            
            # Prepare tool parameters
            tool_params = {"type": "image_generation"}
            
            if input_fidelity == "high":
                tool_params["input_fidelity"] = "high"
            if quality != "auto":
                tool_params["quality"] = quality
            if size != "auto":
                tool_params["size"] = size
            if output_format != "auto":
                tool_params["output_format"] = output_format
            if background != "auto":
                tool_params["background"] = background
            
            # Make API call
            response = client.responses.create(
                model="gpt-4.1",
                input=[{
                    "role": "user",
                    "content": content
                }],
                tools=[tool_params]
            )
            
            return response, "responses_api"
            
        except Exception as e:
            raise Exception(f"Responses API error: {str(e)}")
    
    def process_images_api_response(self, response):
        """Process response from Images API"""
        if hasattr(response, 'data') and len(response.data) > 0:
            result_data = response.data[0]
            
            # Get image data
            if hasattr(result_data, 'b64_json'):
                image_base64 = result_data.b64_json
            else:
                raise Exception("No image data in Images API response")
            
            # Get revised prompt if available
            revised_prompt = getattr(result_data, 'revised_prompt', "N/A (Images API)")
            
            return image_base64, revised_prompt
        else:
            raise Exception("No data received from Images API")
    
    def process_responses_api_response(self, response):
        """Process response from Responses API"""
        if hasattr(response, 'output') and response.output:
            # Find image generation calls
            image_data = [
                output.result
                for output in response.output
                if hasattr(output, 'type') and output.type == "image_generation_call"
            ]
            
            if image_data:
                image_base64 = image_data[0]
                
                # Try to get revised prompt
                revised_prompt = "Generated via Responses API"
                for output in response.output:
                    if hasattr(output, 'type') and output.type == "image_generation_call":
                        if hasattr(output, 'revised_prompt'):
                            revised_prompt = output.revised_prompt
                        break
                
                return image_base64, revised_prompt
            else:
                raise Exception("No image generation calls found in Responses API response")
        else:
            raise Exception("No output received from Responses API")
    
    # MODIFICADO: Añadir parámetros para control de iteraciones
    def generate_fashion_image(self, prompt, primary_image, input_fidelity="high", 
                              quality="high", size="auto", output_format="png", 
                              background="auto", fashion_preset="custom", 
                              api_method="images_api", enable_iterations_control=True,
                              reference_image=None, mask_image=None, api_key="",
                              airtable_api_key="", airtable_base_id=""):
        
        debug_info = []
        remaining_iterations = 0  # NUEVO: Variable para tracking
        
        # NUEVO: CONTROL DE ITERACIONES - LA PARTE CLAVE
        if enable_iterations_control:
            # Usar credenciales del nodo o del config.env
            airtable_key = airtable_api_key.strip() or os.getenv('AIRTABLE_API_KEY', '')
            airtable_base = airtable_base_id.strip() or os.getenv('AIRTABLE_BASE_ID', '')
            
            if not airtable_key or not airtable_base:
                error_msg = "❌ Se requieren credenciales de Airtable para control de iteraciones"
                return (primary_image, f"ERROR: {error_msg}", error_msg, 0)
            
            success, message, remaining_iterations = self.check_and_consume_iteration(
                airtable_key, airtable_base
            )
            
            if not success:
                debug_info.append(f"ACCESO BLOQUEADO: {message}")
                debug_str = " | ".join(debug_info)
                return (primary_image, f"ERROR: {message}", debug_str, remaining_iterations)
            
            debug_info.append(f"✅ {message}")
        else:
            debug_info.append("⚠️ Control de iteraciones deshabilitado")
        
        # RESTO DEL CÓDIGO ORIGINAL (sin cambios)
        # Use provided API key or environment variable
        if api_key.strip():
            try:
                client = OpenAI(api_key=api_key.strip())
                debug_info.append("Using provided API key")
            except Exception as e:
                return (primary_image, f"Error: {str(e)}", f"API key error: {str(e)}", remaining_iterations)
        else:
            if not self.client:
                error_msg = "OpenAI client not initialized. Please set OPENAI_API_KEY environment variable or provide API key."
                return (primary_image, f"Error: {error_msg}", error_msg, remaining_iterations)
            client = self.client
            debug_info.append("Using environment API key")
        
        try:
            # Get optimized prompt for fashion use case
            final_prompt = self.get_fashion_prompt(fashion_preset, prompt)
            debug_info.append(f"Fashion preset: {fashion_preset}")
            debug_info.append(f"Final prompt: {final_prompt[:100]}...")
            
            # Decide which API to use
            use_responses_api = (api_method == "responses_api" or 
                               reference_image is not None)
            
            if use_responses_api:
                debug_info.append("Using Responses API")
                response, api_used = self.edit_with_responses_api(
                    client, primary_image, reference_image, final_prompt,
                    input_fidelity, quality, size, output_format, background
                )
                image_base64, revised_prompt = self.process_responses_api_response(response)
            else:
                debug_info.append("Using Images API")
                response, api_used = self.edit_with_images_api(
                    client, primary_image, mask_image, final_prompt,
                    input_fidelity, quality, size, output_format, background
                )
                image_base64, revised_prompt = self.process_images_api_response(response)
            
            debug_info.append(f"API used: {api_used}")
            debug_info.append(f"Input fidelity: {input_fidelity}")
            
            # Decode and convert image
            image_bytes = base64.b64decode(image_base64)
            result_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert back to ComfyUI tensor
            result_tensor = self.pil_to_tensor(result_image)
            
            debug_info.append("Success: Image generated successfully")
            debug_str = " | ".join(debug_info)
            
            # MODIFICADO: Retornar también remaining_iterations
            return (result_tensor, revised_prompt, debug_str, remaining_iterations)
                
        except Exception as e:
            error_msg = f"Error in fashion image generation: {str(e)}"
            debug_info.append(error_msg)
            debug_str = " | ".join(debug_info)
            print(error_msg)
            
            # MODIFICADO: Return también remaining_iterations
            return (primary_image, f"Error: {str(e)}", debug_str, remaining_iterations)

# Node registration for ComfyUI (sin cambios)
NODE_CLASS_MAPPINGS = {
    "OpenAIImageFidelityFashion": OpenAIImageFidelityFashion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageFidelityFashion": "OpenAI Image Fidelity (Fashion)"
}