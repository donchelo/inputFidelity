# Autor: chelo
# Versión corregida y mejorada del nodo OpenAI Image Edit

import os
import io
import json
import base64
import logging
import hashlib
from typing import Any, Tuple, Optional, Dict, Union
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
    "default_quality": "hd",
    "default_fidelity": "high",
    "default_output_format": "png",
    "max_image_size": 2048,
    "timeout": 30,
    "cache_enabled": True,
    "max_cache_size": 10,
    "error_image_color": [255, 0, 0],  # Rojo para errores
    "combine_background_color": [255, 255, 255]  # Blanco para combinaciones
}

class OpenAIImageEditNode:
    """
    Nodo personalizado para ComfyUI que edita imágenes usando la API de OpenAI.
    """
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "execute"
    CATEGORY = "image/ai"
    DESCRIPTION = "Edita imágenes usando OpenAI API con alta fidelidad y combinación de imágenes"

    def __init__(self):
        """Inicializa el nodo con configuración y cache."""
        self.config = self._load_config()
        self._cache = {} if self.config.get("cache_enabled", True) else None
        self._client = None
        logger.info("OpenAI Image Edit Node inicializado")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define los tipos de entrada para el nodo con opciones mejoradas."""
        return {
            "required": {
                "image_1": ("IMAGE", {
                    "tooltip": "Imagen principal a editar"
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Describe la edición a realizar...",
                    "tooltip": "Descripción detallada de la edición deseada"
                }),
            },
            "optional": {
                "image_2": ("IMAGE", {
                    "tooltip": "Segunda imagen para combinación horizontal (opcional)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Clave API de OpenAI (se guarda de forma segura)"
                }),
                "input_fidelity": (["low", "high"], {
                    "default": "high",
                    "tooltip": "Calidad de procesamiento: 'high' preserva más detalles"
                }),
                "quality": (["standard", "hd"], {
                    "default": "hd",
                    "tooltip": "Calidad de la imagen de salida"
                }),
                "output_format": (["png", "jpeg", "webp"], {
                    "default": "png",
                    "tooltip": "Formato de imagen de salida"
                }),
                "max_size": ("INT", {
                    "default": 1024, 
                    "min": 256, 
                    "max": 2048, 
                    "step": 64,
                    "tooltip": "Tamaño máximo de imagen en píxeles"
                }),
                "enable_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Habilitar cache para mejorar performance"
                }),
            }
        }

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

    @contextmanager
    def _image_buffer(self):
        """Context manager para manejo seguro de buffers de imagen."""
        buffer = io.BytesIO()
        try:
            yield buffer
        finally:
            buffer.close()

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
            # Fallback: crear tensor directamente en formato ComfyUI
            # [batch, height, width, channels] con valores float32 en [0,1]
            error_tensor = torch.zeros(1, height, width, 3, dtype=torch.float32)
            error_tensor[:, :, :, 0] = 1.0  # Canal rojo para error
            return error_tensor

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convierte un tensor ComfyUI a una imagen PIL en formato RGBA.
        ComfyUI usa formato [batch, height, width, channels] con float32 en [0,1]
        """
        try:
            # Validar entrada
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("El input debe ser un torch.Tensor")
            
            # Manejar diferentes formatos de tensor
            if tensor.ndim == 4:
                # Formato batch: [batch, height, width, channels]
                if tensor.shape[0] != 1:
                    raise ValueError(f"Batch size debe ser 1, es {tensor.shape[0]}")
                tensor = tensor[0]  # Remover dimensión de batch: [height, width, channels]
            elif tensor.ndim == 3:
                # Ya está en formato [height, width, channels]
                pass
            else:
                raise ValueError(f"Tensor debe tener 3 o 4 dimensiones, tiene {tensor.ndim}")
            
            # Verificar que tenemos el formato correcto [H, W, C]
            if tensor.shape[-1] not in [1, 3, 4]:
                # Podría estar en formato [C, H, W], intentar transponer
                if tensor.shape[0] in [1, 3, 4]:
                    tensor = tensor.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                else:
                    raise ValueError(f"No se puede determinar el formato del tensor: {tensor.shape}")
            
            # Convertir a numpy y asegurar rango [0, 255]
            if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
                # Asumir que está en rango [0, 1]
                array = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
            else:
                # Asumir que ya está en rango [0, 255]
                array = tensor.clamp(0, 255).byte().cpu().numpy()
            
            # Manejar diferentes números de canales y convertir a RGBA
            if array.shape[-1] == 1:
                # Grayscale -> RGBA
                rgb_array = np.repeat(array, 3, axis=-1)
                rgba_array = np.concatenate([rgb_array, np.full((*array.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
                img = Image.fromarray(rgba_array, mode='RGBA')
            elif array.shape[-1] == 3:
                # RGB -> RGBA (añadir canal alfa completamente opaco)
                rgba_array = np.concatenate([array, np.full((*array.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
                img = Image.fromarray(rgba_array, mode='RGBA')
            elif array.shape[-1] == 4:
                # Ya es RGBA
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
        """
        Convierte una imagen PIL a tensor ComfyUI.
        ComfyUI espera formato [batch, height, width, channels] con float32 en [0,1]
        Convierte RGBA a RGB para ComfyUI.
        """
        try:
            if img is None:
                raise ValueError("La imagen no puede ser None")
            
            # Si la imagen es RGBA, convertir a RGB con fondo blanco
            if img.mode == 'RGBA':
                # Crear fondo blanco
                background = Image.new('RGB', img.size, (255, 255, 255))
                # Componer la imagen RGBA sobre el fondo blanco
                background.paste(img, mask=img.split()[-1])  # Usar canal alfa como máscara
                img = background
            elif img.mode != 'RGB':
                # Convertir cualquier otro formato a RGB
                img = img.convert('RGB')
            
            # Convertir a numpy array
            array = np.array(img, dtype=np.float32)  # [H, W, 3]
            
            # Normalizar a rango [0, 1]
            if array.max() > 1.0:
                array = array / 255.0
            
            # Verificar dimensiones
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Array debe ser [H, W, 3], es {array.shape}")
            
            # Convertir a tensor y añadir dimensión de batch
            tensor = torch.from_numpy(array).unsqueeze(0)  # [1, H, W, 3]
            
            # Verificar que el tensor tiene el formato correcto
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
        """Combina dos imágenes PIL horizontalmente en formato RGBA."""
        try:
            if img1 is None or img2 is None:
                raise ValueError("Las imágenes no pueden ser None")
            
            # Asegurar formato RGBA para ambas imágenes
            if img1.mode != 'RGBA':
                img1 = img1.convert('RGBA')
            if img2.mode != 'RGBA':
                img2 = img2.convert('RGBA')
            
            # Igualar altura
            target_height = max(img1.height, img2.height)
            
            if img1.height != target_height:
                ratio = target_height / img1.height
                new_width = int(img1.width * ratio)
                img1 = img1.resize((new_width, target_height), Image.LANCZOS)
            
            if img2.height != target_height:
                ratio = target_height / img2.height
                new_width = int(img2.width * ratio)
                img2 = img2.resize((new_width, target_height), Image.LANCZOS)
            
            # Crear imagen combinada en RGBA
            total_width = img1.width + img2.width
            bg_color = tuple(self.config.get("combine_background_color", [255, 255, 255]))
            
            # Crear canvas RGBA con fondo opaco
            combined = Image.new("RGBA", (total_width, target_height), color=bg_color + (255,))
            combined.paste(img1, (0, 0), img1)  # Usar img1 como máscara para transparencia
            combined.paste(img2, (img1.width, 0), img2)  # Usar img2 como máscara para transparencia
            
            logger.info(f"Imágenes combinadas en RGBA: {total_width}x{target_height}")
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

    def _validate_inputs(self, image_1: torch.Tensor, image_2: Optional[torch.Tensor],
                        prompt: str, input_fidelity: str, quality: str,
                        output_format: str, max_size: int) -> None:
        """Valida todos los inputs del nodo."""
        # Validar imagen principal
        if not isinstance(image_1, torch.Tensor):
            raise ValueError("image_1 debe ser un torch.Tensor")
        if image_1.ndim not in (3, 4):
            raise ValueError("image_1 debe tener 3 o 4 dimensiones")
        
        # Validar imagen secundaria si existe
        if image_2 is not None:
            if not isinstance(image_2, torch.Tensor):
                raise ValueError("image_2 debe ser un torch.Tensor")
            if image_2.ndim not in (3, 4):
                raise ValueError("image_2 debe tener 3 o 4 dimensiones")
        
        # Validar prompt
        if not isinstance(prompt, str) or len(prompt.strip()) < 3:
            raise ValueError("El prompt debe ser una cadena de al menos 3 caracteres")
        
        # Validar parámetros
        if input_fidelity not in ["low", "high"]:
            raise ValueError("input_fidelity debe ser 'low' o 'high'")
        if quality not in ["standard", "hd"]:
            raise ValueError("quality debe ser 'standard' o 'hd'")
        if output_format not in ["png", "jpeg", "webp"]:
            raise ValueError("output_format debe ser 'png', 'jpeg' o 'webp'")
        if not isinstance(max_size, int) or max_size < 256 or max_size > 2048:
            raise ValueError("max_size debe ser un entero entre 256 y 2048")

    def _get_openai_client(self, api_key: Optional[str] = None) -> OpenAI:
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
        
        if self._client is None or getattr(self._client, '_api_key', None) != final_api_key:
            timeout = self.config.get("timeout", 30)
            self._client = OpenAI(
                api_key=final_api_key,
                timeout=timeout
            )
            logger.info("Cliente OpenAI inicializado")
            
            # Validación simple de API key
            try:
                logger.info("🔍 Validando API key...")
                # Usar una llamada simple que siempre funciona
                self._client.models.list()
                logger.info("✓ API key válida")
            except Exception as validation_error:
                # No fallar por errores de validación, solo avisar
                logger.warning(f"⚠️ No se pudo validar API key: {validation_error}")
                logger.info("Continuando con la ejecución...")
        
        return self._client

    def execute(self,
                image_1: torch.Tensor,
                prompt: str,
                image_2: Optional[torch.Tensor] = None,
                api_key: Optional[str] = None,
                input_fidelity: str = "high",
                quality: str = "hd",
                output_format: str = "png",
                max_size: int = 1024,
                enable_cache: bool = True
                ) -> Tuple[torch.Tensor]:
        """
        Ejecuta la edición de imagen usando la API de OpenAI.
        """
        logger.info("=== Iniciando OpenAI Image Edit ===")
        
        # 1. Validación de inputs
        try:
            self._validate_inputs(image_1, image_2, prompt, input_fidelity, 
                                quality, output_format, max_size)
            logger.info("✓ Inputs validados correctamente")
            logger.info(f"Input image_1 shape: {image_1.shape}, dtype: {image_1.dtype}")
            if image_2 is not None:
                logger.info(f"Input image_2 shape: {image_2.shape}, dtype: {image_2.dtype}")
        except ValueError as e:
            logger.error(f"✗ Error de validación: {e}")
            return (self.create_error_image(message="INVALID INPUT"),)

        # 2. Convertir tensores a PIL
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

        # 3. Obtener cliente OpenAI
        try:
            client = self._get_openai_client(api_key)
            logger.info("✓ Cliente OpenAI listo")
        except Exception as e:
            logger.error(f"✗ Error inicializando cliente: {e}")
            return (self.create_error_image(message="API ERROR"),)

        # 4. Preparar imagen para API y llamar a OpenAI
        try:
            # Asegurar que la imagen está en formato RGBA (requerido por OpenAI)
            if combined_img.mode != 'RGBA':
                combined_img = combined_img.convert('RGBA')
                logger.info(f"Imagen convertida a RGBA para compatibilidad con OpenAI API")
            
            # Crear buffer temporal
            img_buffer = io.BytesIO()
            combined_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            # Validar tamaño de imagen
            img_size = len(img_buffer.getvalue())
            logger.info(f"✓ Imagen preparada para API - Tamaño: {img_size/1024:.1f}KB")
            
            # Verificar que la imagen sea válida
            if img_size == 0:
                raise ValueError("La imagen está vacía")
            if img_size > 4 * 1024 * 1024:  # 4MB límite de OpenAI
                raise ValueError(f"Imagen demasiado grande: {img_size/1024/1024:.1f}MB > 4MB")
            
            img_buffer.seek(0)  # Reset posición
            
            logger.info("⏳ Enviando a OpenAI API...")
            
            # Preparar imagen con tipo MIME correcto
            img_data = img_buffer.getvalue()
            image_tuple = ("image.png", img_data, "image/png")
            
            # Preparar parámetros de API
            api_params = {
                "image": image_tuple,
                "prompt": prompt.strip()
            }
            
            # Añadir parámetros opcionales si están soportados
            try:
                # Intentar con parámetros modernos
                test_params = api_params.copy()
                if quality in ["standard", "hd"]:
                    test_params["quality"] = quality
                if input_fidelity in ["low", "high"]:
                    test_params["input_fidelity"] = input_fidelity
                
                logger.info(f"Intentando con parámetros: quality={quality}, input_fidelity={input_fidelity}")
                response = client.images.edit(**test_params)
                logger.info("✓ Respuesta recibida con parámetros avanzados")
                
            except Exception as modern_error:
                # Si fallan los parámetros modernos, usar solo los básicos
                error_str = str(modern_error).lower()
                if "unexpected keyword argument" in error_str or "input_fidelity" in error_str or "quality" in error_str:
                    logger.warning("⚠️ Parámetros avanzados no soportados, usando API básica...")
                    
                    # Solo usar parámetros básicos
                    basic_params = {
                        "image": image_tuple,
                        "prompt": prompt.strip()
                    }
                    
                    # Intentar añadir quality si está soportado
                    try:
                        if quality == "hd":
                            basic_params["quality"] = "hd"
                        response = client.images.edit(**basic_params)
                        logger.info("✓ Respuesta recibida con API básica + quality")
                    except Exception:
                        # Última opción: solo imagen y prompt
                        response = client.images.edit(
                            image=image_tuple,
                            prompt=prompt.strip()
                        )
                        logger.info("✓ Respuesta recibida con API básica")
                else:
                    # Error diferente, re-lanzar con manejo específico
                    error_type = type(modern_error).__name__
                    logger.error(f"✗ Error específico de OpenAI API ({error_type}): {str(modern_error)}")
                    
                    if "rate_limit" in error_str:
                        logger.error("Solución: Espera un momento antes de volver a intentar")
                        return (self.create_error_image(message="RATE LIMIT"),)
                    elif "invalid_request" in error_str:
                        logger.error("Solución: Verifica los parámetros de la solicitud")
                        return (self.create_error_image(message="INVALID REQUEST"),)
                    elif "authentication" in error_str:
                        logger.error("Solución: Verifica tu API key de OpenAI")
                        return (self.create_error_image(message="AUTH ERROR"),)
                    elif "billing" in error_str:
                        logger.error("Solución: Verifica tu saldo/plan de OpenAI")
                        return (self.create_error_image(message="BILLING ERROR"),)
                    else:
                        logger.error(f"Error no identificado: {modern_error}")
                        return (self.create_error_image(message="API ERROR"),)
                
        except Exception as e:
            logger.error(f"✗ Error preparando imagen para API: {e}")
            logger.error(f"Tipo de error: {type(e).__name__}")
            return (self.create_error_image(message="PREP ERROR"),)

       # 5. Procesar respuesta
        try:
            # Validar que la respuesta tiene la estructura esperada
            if not hasattr(response, 'data') or not response.data:
                logger.error("Respuesta no tiene datos")
                logger.error(f"Respuesta completa: {response}")
                raise ValueError("Respuesta de API sin datos")
            
            if len(response.data) == 0:
                logger.error("Lista de datos está vacía")
                raise ValueError("Respuesta de API con lista vacía")
            
            # Examinar el primer elemento de datos
            first_item = response.data[0]
            logger.info(f"Estructura del primer item: {dir(first_item)}")
            
            # Intentar obtener los datos de imagen en diferentes formatos
            img_data = None
            
            # Método 1: b64_json (más común)
            if hasattr(first_item, 'b64_json') and first_item.b64_json:
                logger.info("Usando b64_json")
                b64_data = first_item.b64_json
                img_data = base64.b64decode(b64_data)
            
            # Método 2: url (si la respuesta incluye URL)
            elif hasattr(first_item, 'url') and first_item.url:
                logger.info("Usando URL, descargando imagen...")
                import requests
                url_response = requests.get(first_item.url)
                if url_response.status_code == 200:
                    img_data = url_response.content
                else:
                    raise ValueError(f"Error descargando desde URL: {url_response.status_code}")
            
            # Método 3: revised_prompt (a veces OpenAI devuelve esto)
            elif hasattr(first_item, 'revised_prompt'):
                logger.warning("Respuesta contiene revised_prompt pero no imagen")
                logger.warning(f"Revised prompt: {first_item.revised_prompt}")
                raise ValueError("Respuesta contiene prompt pero no imagen")
            
            # Método 4: Inspección completa de la respuesta
            else:
                logger.error("No se encontró formato de imagen conocido")
                logger.error(f"Atributos disponibles: {[attr for attr in dir(first_item) if not attr.startswith('_')]}")
                
                # Intentar acceder a todos los atributos no privados
                for attr in dir(first_item):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(first_item, attr)
                            logger.error(f"  {attr}: {type(value)} = {str(value)[:100]}...")
                        except Exception as e:
                            logger.error(f"  {attr}: Error accediendo - {e}")
                
                raise ValueError("No se pudo extraer imagen de la respuesta")
            
            if img_data is None:
                raise ValueError("No se pudieron obtener datos de imagen")
            
            # Crear imagen desde los datos
            result_img = Image.open(io.BytesIO(img_data))
            
            # OpenAI puede devolver la imagen en diferentes formatos
            logger.info(f"Imagen recibida de OpenAI en formato: {result_img.mode}")
            logger.info(f"Tamaño de imagen: {result_img.size}")
            
            # Convertir a tensor usando nuestra función corregida
            result_tensor = self.pil_to_tensor(result_img)
            
            logger.info(f"✓ Imagen procesada exitosamente - Shape: {result_tensor.shape}, dtype: {result_tensor.dtype}")
            
        except Exception as e:
            logger.error(f"✗ Error procesando respuesta: {e}")
            logger.error(f"Tipo de error: {type(e).__name__}")
            
            # Debug adicional: intentar serializar la respuesta
            try:
                import json
                response_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
                logger.error(f"Respuesta completa: {json.dumps(response_dict, indent=2, default=str)[:500]}...")
            except Exception as debug_error:
                logger.error(f"Error en debug de respuesta: {debug_error}")
                logger.error(f"Tipo de respuesta: {type(response)}")
            
            return (self.create_error_image(message="RESPONSE ERROR"),)
        # 6. Cleanup
        try:
            if 'combined_img' in locals():
                combined_img.close()
            if 'result_img' in locals():
                result_img.close()
        except Exception as e:
            logger.warning(f"Warning en cleanup: {e}")

        logger.info("=== Ejecución completada exitosamente ===")
        return (result_tensor,)

# --- Endpoints web opcionales ---
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/openai_image_edit/status")
    async def get_status(request):
        """Endpoint para verificar estado del nodo."""
        return web.json_response({
            "status": "active",
            "message": "OpenAI Image Edit Node funcionando",
            "version": "2.1.0",
            "config_loaded": os.path.exists(CONFIG_FILE),
            "api_key_configured": os.path.exists(API_KEY_FILE)
        })
        
except ImportError:
    logger.info("Endpoints web no disponibles (normal en algunos entornos)")

# --- Exportación de nodos para ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "OpenAIImageEditNode": OpenAIImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageEditNode": "OpenAI Image Edit",
}