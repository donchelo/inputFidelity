# Autor: chelo
# Versión corregida con soporte para input_fidelity usando ambas APIs de OpenAI

import os
import io
import json
import base64
import logging
import hashlib
from typing import Any, Tuple, Optional, Dict, Union, List
from contextlib import contextmanager
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Archivos de configuración (SIN API keys hardcodeadas)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
API_KEY_FILE = os.path.join(os.path.dirname(__file__), "openai_api_key.txt")

# Configuración por defecto
DEFAULT_CONFIG = {
    "default_quality": "high",
    "default_fidelity": "high", 
    "default_output_format": "png",
    "max_image_size": 2048,
    "timeout": 120,  # Aumentado para input_fidelity
    "cache_enabled": True,
    "max_cache_size": 10,
    "error_image_color": [255, 0, 0],
    "combine_background_color": [255, 255, 255],
    "prefer_responses_api": True  # Nuevo: preferir Responses API
}

class OpenAIImageEditNode:
    """
    Nodo personalizado para ComfyUI que edita imágenes usando OpenAI API con alta fidelidad.
    Soporta tanto Image API como Responses API para máxima compatibilidad.
    """
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "execute"
    CATEGORY = "image/ai"
    DESCRIPTION = "Edita imágenes con OpenAI API usando input_fidelity='high' (Image API + Responses API)"

    def __init__(self):
        """Inicializa el nodo con configuración y detección de características."""
        self.config = self._load_config()
        self._cache = {} if self.config.get("cache_enabled", True) else None
        self._client = None
        self._openai_version = None
        self._api_capabilities = None
        logger.info("OpenAI Image Edit Node (Dual API) inicializado")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define los tipos de entrada para el nodo con opciones mejoradas."""
        return {
            "required": {
                "image_1": ("IMAGE", {
                    "tooltip": "Imagen principal a editar (preserva máximo detalle)"
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Describe la edición a realizar...",
                    "tooltip": "Descripción detallada de la edición deseada"
                }),
            },
            "optional": {
                "image_2": ("IMAGE", {
                    "tooltip": "Segunda imagen para combinación (opcional)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Clave API de OpenAI (se guarda de forma segura)"
                }),
                "input_fidelity": (["low", "high"], {
                    "default": "high",
                    "tooltip": "CRÍTICO: 'high' preserva detalles de caras, logos y texturas"
                }),
                "quality": (["standard", "high"], {
                    "default": "high",
                    "tooltip": "Calidad de imagen: 'high' para mejor resolución"
                }),
                "output_format": (["png", "jpeg", "webp"], {
                    "default": "png",
                    "tooltip": "Formato de salida (PNG recomendado para transparencias)"
                }),
                "api_method": (["auto", "responses_api", "image_api"], {
                    "default": "auto",
                    "tooltip": "API a usar: 'auto' detecta automáticamente, 'responses_api' para input_fidelity garantizado"
                }),
                "max_size": ("INT", {
                    "default": 1024, 
                    "min": 256, 
                    "max": 2048, 
                    "step": 64,
                    "tooltip": "Tamaño máximo en píxeles"
                }),
                "enable_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache para mejorar rendimiento"
                }),
                "force_update_client": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Forzar actualización del cliente OpenAI"
                }),
            }
        }

    def _detect_api_capabilities(self) -> Dict[str, bool]:
        """Detecta qué características están disponibles en la versión actual de OpenAI."""
        if self._api_capabilities is not None:
            return self._api_capabilities
            
        capabilities = {
            "image_api_input_fidelity": False,
            "responses_api_available": False,
            "gpt_image_1_available": False
        }
        
        try:
            import inspect
            
            # Verificar si Image API soporta input_fidelity
            try:
                sig = inspect.signature(self._client.images.edit)
                capabilities["image_api_input_fidelity"] = "input_fidelity" in sig.parameters
            except Exception as e:
                logger.warning(f"No se pudo inspeccionar images.edit: {e}")
            
            # Verificar si Responses API está disponible
            try:
                capabilities["responses_api_available"] = hasattr(self._client, 'responses')
            except Exception as e:
                logger.warning(f"No se pudo verificar responses API: {e}")
                
            # Verificar versión de librería
            try:
                import openai
                version = openai.__version__
                self._openai_version = version
                logger.info(f"OpenAI Python library version: {version}")
                
                # Versiones que soportan input_fidelity
                version_parts = version.split('.')
                if len(version_parts) >= 2:
                    major, minor = int(version_parts[0]), int(version_parts[1])
                    if major > 1 or (major == 1 and minor >= 50):  # Estimación
                        capabilities["gpt_image_1_available"] = True
                        
            except Exception as e:
                logger.warning(f"Error verificando versión OpenAI: {e}")
        
        except Exception as e:
            logger.error(f"Error detectando capacidades de API: {e}")
        
        self._api_capabilities = capabilities
        logger.info(f"🔍 Capacidades detectadas: {capabilities}")
        return capabilities

    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde archivo JSON."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    merged_config = DEFAULT_CONFIG.copy()
                    merged_config.update(config)
                    logger.info("Configuración cargada desde archivo")
                    return merged_config
                    
        except Exception as e:
            logger.warning(f"Error cargando configuración: {e}")
        
        logger.info("Usando configuración por defecto")
        return DEFAULT_CONFIG.copy()

    @staticmethod
    def _save_api_key(api_key: str) -> bool:
        """Guarda la clave API de forma segura."""
        try:
            os.makedirs(os.path.dirname(API_KEY_FILE), exist_ok=True)
            
            with open(API_KEY_FILE, "w", encoding='utf-8') as f:
                f.write(api_key.strip())
            
            if hasattr(os, 'chmod'):
                os.chmod(API_KEY_FILE, 0o600)
                
            logger.info("API key guardada de forma segura")
            return True
        except Exception as e:
            logger.warning(f"Error guardando API key: {e}")
            return False

    @staticmethod
    def _load_api_key() -> Optional[str]:
        """Carga la clave API desde archivo seguro."""
        try:
            if os.path.exists(API_KEY_FILE):
                with open(API_KEY_FILE, "r", encoding='utf-8') as f:
                    key = f.read().strip()
                    if key and key.startswith(('sk-', 'sk-proj-')):
                        logger.info("API key cargada desde archivo seguro")
                        return key
                    else:
                        logger.warning("API key en formato inválido")
        except Exception as e:
            logger.warning(f"Error cargando API key: {e}")
        return None

    def create_error_image(self, width: int = 512, height: int = 512, 
                          message: str = "ERROR") -> torch.Tensor:
        """Crea una imagen de error para casos fallidos."""
        try:
            color = tuple(self.config.get("error_image_color", [255, 0, 0]))
            error_img = Image.new('RGB', (width, height), color=color)
            
            try:
                draw = ImageDraw.Draw(error_img)
                font_size = min(width, height) // 20
                draw.text((width//2, height//2), message, 
                         fill=(255, 255, 255), anchor="mm")
            except Exception:
                pass
                
            return self.pil_to_tensor(error_img)
        except Exception as e:
            logger.error(f"Error creando imagen de error: {e}")
            error_tensor = torch.zeros(1, height, width, 3, dtype=torch.float32)
            error_tensor[:, :, :, 0] = 1.0  # Canal rojo para error
            return error_tensor

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convierte un tensor ComfyUI a una imagen PIL en formato RGBA."""
        try:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("El input debe ser un torch.Tensor")
            
            if tensor.ndim == 4:
                if tensor.shape[0] != 1:
                    raise ValueError(f"Batch size debe ser 1, es {tensor.shape[0]}")
                tensor = tensor[0]
            elif tensor.ndim == 3:
                pass
            else:
                raise ValueError(f"Tensor debe tener 3 o 4 dimensiones, tiene {tensor.ndim}")
            
            if tensor.shape[-1] not in [1, 3, 4]:
                if tensor.shape[0] in [1, 3, 4]:
                    tensor = tensor.permute(1, 2, 0)
                else:
                    raise ValueError(f"No se puede determinar el formato del tensor: {tensor.shape}")
            
            if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
                array = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
            else:
                array = tensor.clamp(0, 255).byte().cpu().numpy()
            
            if array.shape[-1] == 1:
                rgb_array = np.repeat(array, 3, axis=-1)
                rgba_array = np.concatenate([rgb_array, np.full((*array.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
                img = Image.fromarray(rgba_array, mode='RGBA')
            elif array.shape[-1] == 3:
                rgba_array = np.concatenate([array, np.full((*array.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
                img = Image.fromarray(rgba_array, mode='RGBA')
            elif array.shape[-1] == 4:
                img = Image.fromarray(array, mode='RGBA')
            else:
                raise ValueError(f"Número de canales no soportado: {array.shape[-1]}")
            
            return img
            
        except Exception as e:
            logger.error(f"Error en tensor_to_pil: {e}")
            logger.error(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            raise

    @staticmethod
    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """Convierte una imagen PIL a tensor ComfyUI."""
        try:
            if img is None:
                raise ValueError("La imagen no puede ser None")
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            array = np.array(img, dtype=np.float32)
            
            if array.max() > 1.0:
                array = array / 255.0
            
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Array debe ser [H, W, 3], es {array.shape}")
            
            tensor = torch.from_numpy(array).unsqueeze(0)
            
            assert tensor.ndim == 4, f"Tensor debe tener 4 dimensiones, tiene {tensor.ndim}"
            assert tensor.shape[0] == 1, f"Batch size debe ser 1, es {tensor.shape[0]}"
            assert tensor.shape[3] == 3, f"Debe tener 3 canales, tiene {tensor.shape[3]}"
            assert tensor.dtype == torch.float32, f"Debe ser float32, es {tensor.dtype}"
            
            logger.debug(f"Tensor creado con shape: {tensor.shape}, dtype: {tensor.dtype}")
            return tensor
            
        except Exception as e:
            logger.error(f"Error en pil_to_tensor: {e}")
            if img is not None:
                logger.error(f"PIL Image mode: {img.mode}, size: {img.size}")
            raise

    def combine_images_horizontally(self, img1: Image.Image, 
                                  img2: Image.Image) -> Image.Image:
        """Combina dos imágenes horizontalmente según patrón de documentación OpenAI."""
        try:
            if img1 is None or img2 is None:
                raise ValueError("Las imágenes no pueden ser None")
            
            if img1.mode != 'RGBA':
                img1 = img1.convert('RGBA')
            if img2.mode != 'RGBA':
                img2 = img2.convert('RGBA')
            
            target_height = img1.height
            
            if img2.height != target_height:
                ratio = target_height / img2.height
                new_width = int(img2.width * ratio)
                img2 = img2.resize((new_width, target_height), Image.LANCZOS)
            
            total_width = img1.width + img2.width
            bg_color = tuple(self.config.get("combine_background_color", [255, 255, 255]))
            
            combined = Image.new("RGBA", (total_width, target_height), color=bg_color + (255,))
            
            combined.paste(img1, (0, 0), img1)
            combined.paste(img2, (img1.width, 0), img2)
            
            logger.info(f"Imágenes combinadas (imagen1={img1.size}, imagen2={img2.size}) -> {total_width}x{target_height}")
            return combined
            
        except Exception as e:
            logger.error(f"Error en combine_images_horizontally: {e}")
            raise

    def _resize_image_if_needed(self, img: Image.Image, max_size: int) -> Image.Image:
        """Redimensiona imagen si excede el tamaño máximo, preservando el formato."""
        if max(img.width, img.height) > max_size:
            if img.width > img.height:
                ratio = max_size / img.width
                new_size = (max_size, int(img.height * ratio))
            else:
                ratio = max_size / img.height
                new_size = (int(img.width * ratio), max_size)
            
            img = img.resize(new_size, Image.LANCZOS)
            logger.info(f"Imagen redimensionada a {new_size}")
        
        return img

    def _get_openai_client(self, api_key: Optional[str] = None, force_update: bool = False) -> OpenAI:
        """Obtiene cliente OpenAI con gestión de API key."""
        if api_key and api_key.strip():
            self._save_api_key(api_key)
        
        final_api_key = (
            api_key.strip() if api_key and api_key.strip() 
            else self._load_api_key() 
            or os.getenv("OPENAI_API_KEY")
        )
        
        if not final_api_key:
            raise ValueError(
                "No se encontró API key de OpenAI. "
                "Proporciona una en el campo del nodo, guárdala en archivo, "
                "o configura la variable de entorno OPENAI_API_KEY"
            )
        
        if not final_api_key.startswith(('sk-', 'sk-proj-')):
            raise ValueError("API key no tiene el formato esperado de OpenAI")
        
        if self._client is None or getattr(self._client, '_api_key', None) != final_api_key or force_update:
            timeout = self.config.get("timeout", 120)
            self._client = OpenAI(
                api_key=final_api_key,
                timeout=timeout
            )
            logger.info("Cliente OpenAI inicializado/actualizado")
            
            # Detectar capacidades después de inicializar
            self._detect_api_capabilities()
        
        return self._client

    def _prepare_image_for_api(self, img: Image.Image, output_format: str) -> Tuple[str, bytes, str]:
        """Prepara imagen para API de OpenAI según especificaciones oficiales."""
        try:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
                logger.info("Imagen convertida a RGBA para compatibilidad OpenAI")
            
            format_map = {
                "png": ("PNG", "image/png"),
                "jpeg": ("JPEG", "image/jpeg"), 
                "webp": ("WEBP", "image/webp")
            }
            
            pil_format, mime_type = format_map.get(output_format, ("PNG", "image/png"))
            
            if pil_format == "JPEG" and img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
                logger.info("Imagen convertida a RGB para formato JPEG")
            
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=pil_format, quality=95 if pil_format == "JPEG" else None)
            img_bytes = img_buffer.getvalue()
            
            if len(img_bytes) == 0:
                raise ValueError("La imagen resultó vacía después de la conversión")
            
            if len(img_bytes) > 4 * 1024 * 1024:  # 4MB límite OpenAI
                raise ValueError(f"Imagen demasiado grande: {len(img_bytes)/1024/1024:.1f}MB > 4MB")
            
            filename = f"image.{output_format.lower()}"
            logger.info(f"Imagen preparada: {filename}, {len(img_bytes)/1024:.1f}KB, {mime_type}")
            
            return (filename, img_bytes, mime_type)
            
        except Exception as e:
            logger.error(f"Error preparando imagen para API: {e}")
            raise

    def _call_responses_api(self, client: OpenAI, image_tuple: Tuple[str, bytes, str], 
                           prompt: str, input_fidelity: str, quality: str, 
                           output_format: str) -> Any:
        """
        Usa la Responses API que garantiza soporte para input_fidelity.
        Según documentación oficial OpenAI.
        """
        try:
            # Crear file-like object para la imagen
            filename, img_bytes, mime_type = image_tuple
            
            # Codificar imagen en base64 para Responses API
            image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            data_url = f"data:{mime_type};base64,{image_b64}"
            
            # Configurar tool con input_fidelity (según documentación)
            tool_config = {
                "type": "image_generation",
                "input_fidelity": input_fidelity,
                "quality": quality
            }
            
            # Agregar output_format si es soportado
            if output_format in ["png", "jpeg", "webp"]:
                tool_config["output_format"] = output_format
            
            logger.info(f"🔄 Usando Responses API con configuración: {tool_config}")
            
            response = client.responses.create(
                model="gpt-4o",  # Modelo que soporta image_generation tool
                input=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "input_text", "text": prompt.strip()},
                            {"type": "input_image", "image_url": data_url}
                        ]
                    }
                ],
                tools=[tool_config]
            )
            
            logger.info("✅ Respuesta exitosa desde Responses API")
            return response
            
        except Exception as e:
            logger.error(f"Error en Responses API: {e}")
            raise

    def _call_image_api(self, client: OpenAI, image_tuple: Tuple[str, bytes, str], 
                       prompt: str, input_fidelity: str, quality: str, 
                       output_format: str) -> Any:
        """
        Usa la Image API tradicional con fallbacks inteligentes.
        """
        try:
            filename, img_bytes, mime_type = image_tuple
            
            # Configuración base
            base_params = {
                "model": "gpt-image-1",
                "image": (filename, img_bytes, mime_type),
                "prompt": prompt.strip()
            }
            
            # Detectar capacidades
            capabilities = self._detect_api_capabilities()
            
            if capabilities.get("image_api_input_fidelity", False):
                # Usar configuración completa si está soportada
                full_params = base_params.copy()
                full_params.update({
                    "input_fidelity": input_fidelity,
                    "quality": quality,
                    "output_format": output_format
                })
                
                logger.info(f"🎯 Usando Image API con input_fidelity soportado")
                response = client.images.edit(**full_params)
                
            else:
                # Fallback sin input_fidelity
                logger.warning("⚠️ input_fidelity no soportado en Image API, usando fallback")
                response = client.images.edit(**base_params)
                
            logger.info("✅ Respuesta exitosa desde Image API")
            return response
            
        except Exception as e:
            logger.error(f"Error en Image API: {e}")
            raise

    def _extract_image_from_responses_api(self, response: Any) -> bytes:
        """Extrae datos de imagen desde respuesta de Responses API."""
        try:
            image_generation_calls = [
                output for output in response.output
                if output.type == "image_generation_call"
            ]
            
            if not image_generation_calls:
                raise ValueError("No se encontraron llamadas de generación de imagen en la respuesta")
            
            first_call = image_generation_calls[0]
            
            if hasattr(first_call, 'result') and first_call.result:
                # La respuesta ya está en base64
                img_data = base64.b64decode(first_call.result)
                logger.info("📥 Imagen extraída desde Responses API (result)")
                return img_data
            else:
                raise ValueError("No se encontró resultado de imagen en la respuesta")
                
        except Exception as e:
            logger.error(f"Error extrayendo imagen de Responses API: {e}")
            raise

    def _extract_image_from_image_api(self, response: Any) -> bytes:
        """Extrae datos de imagen desde respuesta de Image API."""
        try:
            if not hasattr(response, 'data') or not response.data:
                raise ValueError("Respuesta sin datos válidos")
            
            first_item = response.data[0]
            
            if hasattr(first_item, 'b64_json') and first_item.b64_json:
                img_data = base64.b64decode(first_item.b64_json)
                logger.info("📥 Imagen extraída desde Image API (b64_json)")
                return img_data
            elif hasattr(first_item, 'url') and first_item.url:
                import requests
                url_response = requests.get(first_item.url, timeout=30)
                if url_response.status_code == 200:
                    img_data = url_response.content
                    logger.info("📥 Imagen extraída desde Image API (URL)")
                    return img_data
                else:
                    raise ValueError(f"Error descargando desde URL: {url_response.status_code}")
            else:
                raise ValueError("No se encontró imagen en la respuesta de Image API")
                
        except Exception as e:
            logger.error(f"Error extrayendo imagen de Image API: {e}")
            raise

    def execute(self,
                image_1: torch.Tensor,
                prompt: str,
                image_2: Optional[torch.Tensor] = None,
                api_key: Optional[str] = None,
                input_fidelity: str = "high",
                quality: str = "high",
                output_format: str = "png",
                api_method: str = "auto",
                max_size: int = 1024,
                enable_cache: bool = True,
                force_update_client: bool = False
                ) -> Tuple[torch.Tensor]:
        """
        Ejecuta la edición de imagen con input_fidelity usando la mejor API disponible.
        """
        logger.info("=== OpenAI Image Edit con Dual API Support ===")
        logger.info(f"Configuración: fidelity={input_fidelity}, quality={quality}, format={output_format}, api={api_method}")
        
        # 1. Validaciones básicas
        try:
            if not isinstance(image_1, torch.Tensor) or image_1.ndim not in (3, 4):
                raise ValueError("image_1 debe ser un tensor válido")
            if image_2 is not None and (not isinstance(image_2, torch.Tensor) or image_2.ndim not in (3, 4)):
                raise ValueError("image_2 debe ser un tensor válido o None")
            if not isinstance(prompt, str) or len(prompt.strip()) < 3:
                raise ValueError("El prompt debe tener al menos 3 caracteres")
            
            logger.info("✓ Validaciones básicas pasadas")
            
        except ValueError as e:
            logger.error(f"✗ Error de validación: {e}")
            return (self.create_error_image(message="INVALID INPUT"),)

        # 2. Procesar imágenes
        try:
            img1 = self.tensor_to_pil(image_1)
            img1 = self._resize_image_if_needed(img1, max_size)
            
            if image_2 is not None:
                img2 = self.tensor_to_pil(image_2)
                img2 = self._resize_image_if_needed(img2, max_size)
                combined_img = self.combine_images_horizontally(img1, img2)
                logger.info("✓ Imágenes combinadas")
            else:
                combined_img = img1
                logger.info("✓ Imagen única procesada")
                
        except Exception as e:
            logger.error(f"✗ Error procesando imágenes: {e}")
            return (self.create_error_image(message="IMAGE ERROR"),)

        # 3. Inicializar cliente OpenAI
        try:
            client = self._get_openai_client(api_key, force_update_client)
            capabilities = self._detect_api_capabilities()
            logger.info("✓ Cliente OpenAI inicializado")
        except Exception as e:
            logger.error(f"✗ Error con cliente OpenAI: {e}")
            return (self.create_error_image(message="API ERROR"),)

        # 4. Determinar qué API usar
        try:
            use_responses_api = False
            
            if api_method == "responses_api":
                use_responses_api = True
                logger.info("🔧 Forzando uso de Responses API")
            elif api_method == "image_api":
                use_responses_api = False
                logger.info("🔧 Forzando uso de Image API")
            else:  # auto
                if input_fidelity == "high":
                    if capabilities.get("responses_api_available", False):
                        use_responses_api = True
                        logger.info("🎯 Auto: Usando Responses API para input_fidelity='high'")
                    elif capabilities.get("image_api_input_fidelity", False):
                        use_responses_api = False
                        logger.info("🎯 Auto: Usando Image API con soporte input_fidelity")
                    else:
                        use_responses_api = False
                        logger.warning("⚠️ Auto: input_fidelity no disponible, usando Image API básica")
                else:
                    use_responses_api = False
                    logger.info("🎯 Auto: Usando Image API para input_fidelity='low'")
            
        except Exception as e:
            logger.error(f"✗ Error determinando API: {e}")
            return (self.create_error_image(message="API SELECTION ERROR"),)

        # 5. Preparar imagen y ejecutar
        try:
            image_tuple = self._prepare_image_for_api(combined_img, output_format)
            
            logger.info(f"⏳ Enviando a OpenAI {'Responses' if use_responses_api else 'Image'} API...")
            
            if use_responses_api:
                response = self._call_responses_api(
                    client, image_tuple, prompt, input_fidelity, quality, output_format
                )
                img_data = self._extract_image_from_responses_api(response)
            else:
                response = self._call_image_api(
                    client, image_tuple, prompt, input_fidelity, quality, output_format
                )
                img_data = self._extract_image_from_image_api(response)
            
            logger.info("✅ Respuesta recibida y procesada")
            
        except Exception as e:
            logger.error(f"✗ Error en llamada API: {e}")
            return (self.create_error_image(message="API ERROR"),)

        # 6. Procesar respuesta final
        try:
            result_img = Image.open(io.BytesIO(img_data))
            logger.info(f"📷 Imagen final: {result_img.mode}, {result_img.size}")
            
            result_tensor = self.pil_to_tensor(result_img)
            
            logger.info(f"✅ Procesamiento completado - Tensor: {result_tensor.shape}, dtype: {result_tensor.dtype}")
            
        except Exception as e:
            logger.error(f"✗ Error procesando respuesta: {e}")
            return (self.create_error_image(message="RESPONSE ERROR"),)

        # 7. Cleanup
        try:
            if 'combined_img' in locals():
                combined_img.close()
            if 'result_img' in locals():
                result_img.close()
        except Exception as e:
            logger.warning(f"Warning en cleanup: {e}")

        logger.info("=== ¡Ejecución con Dual API completada! ===")
        return (result_tensor,)

# --- Endpoints web opcionales ---
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/openai_image_edit/status")
    async def get_status(request):
        """Endpoint para verificar estado del nodo."""
        try:
            import openai
            return web.json_response({
                "status": "active",
                "message": "OpenAI Image Edit Node (Dual API) funcionando",
                "version": "4.0.0-dual_api",
                "openai_version": openai.__version__,
                "supports_dual_api": True,
                "config_loaded": os.path.exists(CONFIG_FILE),
                "api_key_configured": os.path.exists(API_KEY_FILE)
            })
        except Exception as e:
            return web.json_response({
                "status": "error",
                "message": str(e)
            }, status=500)
        
except ImportError:
    logger.info("Endpoints web no disponibles (normal en algunos entornos)")

# --- Exportación de nodos para ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "OpenAIImageEditNode": OpenAIImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageEditNode": "OpenAI Image Edit (Dual API + High Fidelity)",
}