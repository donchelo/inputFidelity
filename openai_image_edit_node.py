# Autor: chelo
# Versión corregida con input_fidelity según documentación oficial OpenAI

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
    "default_quality": "high",  # Corregido: "high" en lugar de "hd"
    "default_fidelity": "high",
    "default_output_format": "png",
    "max_image_size": 2048,
    "timeout": 60,  # Aumentado para manejar input_fidelity="high"
    "cache_enabled": True,
    "max_cache_size": 10,
    "error_image_color": [255, 0, 0],  # Rojo para errores
    "combine_background_color": [255, 255, 255]  # Blanco para combinaciones
}

class OpenAIImageEditNode:
    """
    Nodo personalizado para ComfyUI que edita imágenes usando OpenAI API con alta fidelidad.
    Implementa input_fidelity="high" según documentación oficial OpenAI.
    """
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "execute"
    CATEGORY = "image/ai"
    DESCRIPTION = "Edita imágenes con OpenAI API usando input_fidelity='high' para preservar detalles"

    def __init__(self):
        """Inicializa el nodo con configuración y cache."""
        self.config = self._load_config()
        self._cache = {} if self.config.get("cache_enabled", True) else None
        self._client = None
        self._openai_version = None
        logger.info("OpenAI Image Edit Node (High Fidelity) inicializado")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define los tipos de entrada para el nodo con opciones mejoradas."""
        return {
            "required": {
                "image_1": ("IMAGE", {
                    "tooltip": "Imagen principal a editar (preserva máximo detalle con input_fidelity='high')"
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Describe la edición a realizar...",
                    "tooltip": "Descripción detallada de la edición deseada"
                }),
            },
            "optional": {
                "image_2": ("IMAGE", {
                    "tooltip": "Segunda imagen (opcional) - Útil para combinar elementos según docs oficiales"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Clave API de OpenAI (se guarda de forma segura)"
                }),
                "input_fidelity": (["low", "high"], {
                    "default": "high",
                    "tooltip": "CRÍTICO: 'high' preserva detalles de caras, logos y texturas según docs oficiales"
                }),
                "quality": (["standard", "high"], {
                    "default": "high",
                    "tooltip": "Calidad de imagen: 'high' para mejor resolución"
                }),
                "output_format": (["png", "jpeg", "webp"], {
                    "default": "png",
                    "tooltip": "Formato de salida (PNG recomendado para transparencias)"
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
                "combine_images": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Combinar imágenes horizontalmente (útil para múltiples elementos)"
                }),
            }
        }

    def _check_openai_version(self) -> None:
        """Verifica la versión de OpenAI y registra capacidades."""
        try:
            import openai
            self._openai_version = openai.__version__
            logger.info(f"OpenAI versión: {self._openai_version}")
            
            # Verificar si input_fidelity está soportado
            self._supports_input_fidelity = self._test_input_fidelity_support()
            logger.info(f"Soporte input_fidelity: {self._supports_input_fidelity}")
            
        except Exception as e:
            logger.warning(f"No se pudo verificar versión OpenAI: {e}")
            self._openai_version = "unknown"
            self._supports_input_fidelity = False

    def _test_input_fidelity_support(self) -> bool:
        """Verifica si el cliente OpenAI actual soporta input_fidelity."""
        try:
            import inspect
            sig = inspect.signature(self._client.images.edit)
            supported = "input_fidelity" in sig.parameters
            logger.info(f"Parámetro input_fidelity {'soportado' if supported else 'no soportado'}")
            return supported
        except Exception as e:
            logger.warning(f"No se pudo verificar soporte input_fidelity: {e}")
            return False

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

    @staticmethod
    def combine_images_horizontal(img1: Image.Image, img2: Image.Image, 
                                bg_color: tuple = (255, 255, 255)) -> Image.Image:
        """
        Combina dos imágenes horizontalmente según el patrón de la documentación oficial.
        Útil para preservar detalles de múltiples elementos con input_fidelity='high'.
        """
        try:
            # Asegurar formato RGBA para manejo seguro de transparencias
            left = img1.convert("RGBA")
            right = img2.convert("RGBA")

            # Redimensionar derecha para coincidir con altura de izquierda
            target_h = left.height
            scale = target_h / float(right.height)
            target_w = int(round(right.width * scale))
            right = right.resize((target_w, target_h), Image.LANCZOS)

            # Crear nuevo canvas
            total_w = left.width + right.width
            canvas = Image.new("RGBA", (total_w, target_h), bg_color + (255,))

            # Pegar imágenes
            canvas.paste(left, (0, 0), left)
            canvas.paste(right, (left.width, 0), right)

            # Convertir de vuelta a RGB para API
            result = canvas.convert("RGB")
            
            logger.info(f"Imágenes combinadas: {left.size} + {right.size} = {result.size}")
            return result
            
        except Exception as e:
            logger.error(f"Error combinando imágenes: {e}")
            return img1  # Retornar imagen original si falla


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
            timeout = self.config.get("timeout", 60)
            self._client = OpenAI(
                api_key=final_api_key,
                timeout=timeout
            )
            logger.info("Cliente OpenAI inicializado/actualizado")
            
            # Verificar versión de librería
            self._check_openai_version()
        
        return self._client

    def _prepare_image_for_api(self, img: Image.Image, output_format: str) -> Tuple[str, bytes, str]:
        """
        Prepara imagen para API de OpenAI según especificaciones oficiales.
        Retorna tupla (filename, bytes, mime_type) como requiere la API.
        """
        try:
            # Asegurar formato RGBA para compatibilidad máxima
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
                logger.info("Imagen convertida a RGBA para compatibilidad OpenAI")
            
            # Mapear formato de salida a formato de entrada
            format_map = {
                "png": ("PNG", "image/png"),
                "jpeg": ("JPEG", "image/jpeg"), 
                "webp": ("WEBP", "image/webp")
            }
            
            pil_format, mime_type = format_map.get(output_format, ("PNG", "image/png"))
            
            # Para JPEG, convertir a RGB (no soporta transparencia)
            if pil_format == "JPEG" and img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
                logger.info("Imagen convertida a RGB para formato JPEG")
            
            # Crear buffer
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=pil_format, quality=95 if pil_format == "JPEG" else None)
            img_bytes = img_buffer.getvalue()
            
            # Validaciones
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

    def _call_openai_edit_api(self, client: OpenAI, image_tuple: Tuple[str, bytes, str], 
                             prompt: str, input_fidelity: str, quality: str, 
                             output_format: str) -> Any:
        """
        Llama a la API de OpenAI con manejo robusto de parámetros.
        Implementa exactamente los parámetros de la documentación oficial.
        """
        try:
            # Configuración base según documentación oficial
            base_params = {
                "model": "gpt-image-1",  # Modelo oficial para input_fidelity según documentación
                "image": image_tuple,
                "prompt": prompt.strip()
            }
            
            # Llamada directa con gpt-image-1 e input_fidelity
            try:
                edit_params = base_params.copy()
                edit_params.update({
                    "input_fidelity": input_fidelity,
                    "quality": quality,
                    "output_format": output_format
                })
                
                logger.info(f"🎯 Llamando API con gpt-image-1 e input_fidelity:")
                logger.info(f"   model: {edit_params['model']}")
                logger.info(f"   input_fidelity: {edit_params['input_fidelity']}")
                logger.info(f"   quality: {edit_params['quality']}")
                logger.info(f"   output_format: {edit_params['output_format']}")
                
                response = client.images.edit(**edit_params)
                logger.info("✅ ¡Éxito con gpt-image-1 e input_fidelity!")
                return response
                
            except Exception as edit_error:
                logger.error(f"⚠️ Error con gpt-image-1: {edit_error}")
                
                # Analizar tipos de error específicos
                error_str = str(edit_error).lower()
                if "rate_limit" in error_str or "rate limit" in error_str:
                    raise ValueError("Límite de velocidad de API alcanzado. Espera unos minutos.")
                elif "authentication" in error_str or "unauthorized" in error_str:
                    raise ValueError("Error de autenticación. Verifica tu API key.")
                elif "billing" in error_str or "quota" in error_str:
                    raise ValueError("Error de facturación. Verifica tu cuenta OpenAI.")
                elif "invalid" in error_str and "model" in error_str:
                    raise ValueError("Modelo gpt-image-1 no disponible. Verifica tu acceso.")
                elif "unexpected keyword argument" in error_str:
                    raise ValueError("Parámetros no soportados. Verifica tu versión de OpenAI library.")
                else:
                    raise ValueError(f"Error de API: {edit_error}")
        
        except Exception as e:
            logger.error(f"Error en llamada a API: {e}")
            raise

    def execute(self,
                image_1: torch.Tensor,
                prompt: str,
                image_2: Optional[torch.Tensor] = None,
                api_key: Optional[str] = None,
                input_fidelity: str = "high",
                quality: str = "high",
                output_format: str = "png",
                max_size: int = 1024,
                enable_cache: bool = True,
                force_update_client: bool = False,
                combine_images: bool = False
                ) -> Tuple[torch.Tensor]:
        """
        Ejecuta la edición de imagen usando gpt-image-1 con input_fidelity='high'.
        Implementa exactamente el flujo de la documentación oficial OpenAI.
        """
        logger.info("=== OpenAI Image Edit con Input Fidelity HIGH ===")
        logger.info(f"Configuración: fidelity={input_fidelity}, quality={quality}, format={output_format}")
        
        # 1. Validaciones básicas
        try:
            if not isinstance(image_1, torch.Tensor) or image_1.ndim not in (3, 4):
                raise ValueError("image_1 debe ser un tensor válido")
            if not isinstance(prompt, str) or len(prompt.strip()) < 3:
                raise ValueError("El prompt debe tener al menos 3 caracteres")
            
            logger.info("✓ Validaciones básicas pasadas")
            
        except ValueError as e:
            logger.error(f"✗ Error de validación: {e}")
            return (self.create_error_image(message="INVALID INPUT"),)

        # 2. Procesar imagen según patrón documentación OpenAI
        try:
            # Convertir imagen principal (preserva máximo detalle con input_fidelity='high')
            input_img = self.tensor_to_pil(image_1)
            input_img = self._resize_image_if_needed(input_img, max_size)
            logger.info("✓ Imagen principal procesada para input_fidelity='high'")
            
            # Procesar segunda imagen si está disponible
            if image_2 is not None and combine_images:
                try:
                    img2 = self.tensor_to_pil(image_2)
                    img2 = self._resize_image_if_needed(img2, max_size)
                    
                    # Combinar imágenes horizontalmente según documentación oficial
                    input_img = self.combine_images_horizontal(input_img, img2)
                    logger.info("✓ Imágenes combinadas horizontalmente según patrón oficial")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error combinando imágenes: {e}, usando solo imagen principal")
                    
            elif image_2 is not None and not combine_images:
                logger.info("ℹ️ Segunda imagen proporcionada pero combine_images=False, usando solo imagen principal")
                
        except Exception as e:
            logger.error(f"✗ Error procesando imagen: {e}")
            return (self.create_error_image(message="IMAGE ERROR"),)

        # 3. Inicializar cliente OpenAI
        try:
            client = self._get_openai_client(api_key, force_update_client)
            logger.info("✓ Cliente OpenAI inicializado")
        except Exception as e:
            logger.error(f"✗ Error con cliente OpenAI: {e}")
            return (self.create_error_image(message="API ERROR"),)

        # 4. Preparar imagen para API y ejecutar
        try:
            # Preparar imagen según especificaciones API
            image_tuple = self._prepare_image_for_api(input_img, output_format)
            
            logger.info("⏳ Enviando a OpenAI API con input_fidelity='high'...")
            
            # Ejecutar llamada con manejo robusto
            response = self._call_openai_edit_api(
                client, image_tuple, prompt, input_fidelity, quality, output_format
            )
            
            logger.info("✅ Respuesta recibida de OpenAI")
            
        except Exception as e:
            logger.error(f"✗ Error en llamada API: {e}")
            return (self.create_error_image(message="API ERROR"),)

        # 5. Procesar respuesta
        try:
            if not hasattr(response, 'data') or not response.data or len(response.data) == 0:
                raise ValueError("Respuesta de API sin datos válidos")
            
            first_item = response.data[0]
            
            # Obtener datos de imagen (preferir b64_json)
            if hasattr(first_item, 'b64_json') and first_item.b64_json:
                logger.info("📥 Procesando respuesta en formato b64_json")
                img_data = base64.b64decode(first_item.b64_json)
            elif hasattr(first_item, 'url') and first_item.url:
                logger.info("📥 Descargando desde URL...")
                import requests
                url_response = requests.get(first_item.url, timeout=30)
                if url_response.status_code == 200:
                    img_data = url_response.content
                else:
                    raise ValueError(f"Error descargando imagen: {url_response.status_code}")
            else:
                raise ValueError("No se encontró imagen en la respuesta")
            
            # Crear imagen desde datos
            result_img = Image.open(io.BytesIO(img_data))
            logger.info(f"📷 Imagen recibida: {result_img.mode}, {result_img.size}")
            
            # Convertir a tensor ComfyUI
            result_tensor = self.pil_to_tensor(result_img)
            
            logger.info(f"✅ Procesamiento completado - Tensor: {result_tensor.shape}, dtype: {result_tensor.dtype}")
            
        except Exception as e:
            logger.error(f"✗ Error procesando respuesta: {e}")
            return (self.create_error_image(message="RESPONSE ERROR"),)

        # 6. Cleanup
        try:
            if 'input_img' in locals():
                input_img.close()
            if 'result_img' in locals():
                result_img.close()
        except Exception as e:
            logger.warning(f"Warning en cleanup: {e}")

        logger.info("=== ¡Ejecución con input_fidelity='high' completada! ===")
        return (result_tensor,)

# --- Endpoints web opcionales ---
# Estos endpoints solo están disponibles cuando ComfyUI está ejecutándose
def setup_web_endpoints():
    """Configura endpoints web si ComfyUI está disponible."""
    try:
        from aiohttp import web
        from server import PromptServer
        
        @PromptServer.instance.routes.get("/openai_image_edit/status")
        async def get_status(request):
            """Endpoint para verificar estado del nodo."""
            import openai
            return web.json_response({
                "status": "active",
                "message": "OpenAI Image Edit Node con input_fidelity funcionando",
                "version": "3.0.0-input_fidelity",
                "openai_version": openai.__version__,
                "supports_input_fidelity": True,
                "config_loaded": os.path.exists(CONFIG_FILE),
                "api_key_configured": os.path.exists(API_KEY_FILE)
            })
        
        logger.info("Endpoints web configurados exitosamente")
        return True
        
    except ImportError as e:
        logger.debug(f"Endpoints web no disponibles: {e} (normal durante desarrollo)")
        return False
    except Exception as e:
        logger.warning(f"Error configurando endpoints web: {e}")
        return False

# Intentar configurar endpoints al importar el módulo
_web_endpoints_configured = setup_web_endpoints()

# --- Exportación de nodos para ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "OpenAIImageEditNode": OpenAIImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageEditNode": "OpenAI Image Edit (High Fidelity)",
}