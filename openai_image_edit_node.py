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
    
    Funcionalidades principales:
    - Edita imágenes usando prompts de texto a través de OpenAI API
    - Puede combinar dos imágenes horizontalmente antes del procesamiento
    - Sistema seguro de gestión de claves API
    - Soporte para configuración personalizable
    - Sistema de cache para mejorar performance
    - Manejo robusto de errores con imágenes de fallback
    
    Seguridad:
    - No almacena API keys en código
    - Gestión segura de archivos de configuración
    - Validación exhaustiva de inputs
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
        """
        Define los tipos de entrada para el nodo con opciones mejoradas.
        
        Returns:
            Dict: Configuración completa de inputs
        """
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
        """
        Carga la configuración desde archivo JSON.
        
        Returns:
            Dict: Configuración cargada o configuración por defecto
        """
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge con configuración por defecto
                    merged_config = DEFAULT_CONFIG.copy()
                    merged_config.update(config)
                    logger.info("Configuración cargada desde archivo")
                    return merged_config
        except Exception as e:
            logger.warning(f"Error cargando configuración: {e}")
        
        logger.info("Usando configuración por defecto")
        return DEFAULT_CONFIG.copy()

    def _save_config(self) -> bool:
        """
        Guarda la configuración actual en archivo JSON.
        
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info("Configuración guardada exitosamente")
            return True
        except Exception as e:
            logger.warning(f"Error guardando configuración: {e}")
            return False

    @staticmethod
    def _save_api_key(api_key: str) -> bool:
        """
        Guarda la clave API de forma segura.
        
        Args:
            api_key: Clave API a guardar
            
        Returns:
            bool: True si se guardó exitosamente
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(API_KEY_FILE), exist_ok=True)
            
            with open(API_KEY_FILE, "w", encoding='utf-8') as f:
                f.write(api_key.strip())
            
            # Establecer permisos restrictivos en sistemas Unix
            if hasattr(os, 'chmod'):
                os.chmod(API_KEY_FILE, 0o600)
                
            logger.info("API key guardada de forma segura")
            return True
        except Exception as e:
            logger.warning(f"Error guardando API key: {e}")
            return False

    @staticmethod
    def _load_api_key() -> Optional[str]:
        """
        Carga la clave API desde archivo seguro.
        
        Returns:
            Optional[str]: Clave API si existe y es válida
        """
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

    def _get_cache_key(self, prompt: str, fidelity: str, quality: str, 
                      image_hash: str) -> str:
        """
        Genera clave única para cache basada en parámetros.
        
        Args:
            prompt: Prompt de edición
            fidelity: Configuración de fidelidad
            quality: Configuración de calidad
            image_hash: Hash de la imagen de entrada
            
        Returns:
            str: Clave única para cache
        """
        content = f"{prompt}_{fidelity}_{quality}_{image_hash}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _get_image_hash(self, image: Image.Image) -> str:
        """
        Calcula hash MD5 de una imagen.
        
        Args:
            image: Imagen PIL
            
        Returns:
            str: Hash MD5 de la imagen
        """
        with self._image_buffer() as buffer:
            image.save(buffer, format='PNG')
            return hashlib.md5(buffer.getvalue()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """
        Verifica si existe resultado en cache.
        
        Args:
            cache_key: Clave de cache a verificar
            
        Returns:
            Optional[torch.Tensor]: Resultado cached o None
        """
        if self._cache is None:
            return None
        return self._cache.get(cache_key)

    def _save_to_cache(self, cache_key: str, result: torch.Tensor) -> None:
        """
        Guarda resultado en cache con límite de tamaño.
        
        Args:
            cache_key: Clave para el cache
            result: Tensor resultado a guardar
        """
        if self._cache is None:
            return
            
        # Limpiar cache si está lleno
        max_size = self.config.get("max_cache_size", 10)
        if len(self._cache) >= max_size:
            # Remover entrada más antigua
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[cache_key] = result.clone()
        logger.debug(f"Resultado guardado en cache: {cache_key}")

    def create_error_image(self, width: int = 512, height: int = 512, 
                          message: str = "ERROR") -> torch.Tensor:
        """
        Crea una imagen de error para casos fallidos.
        
        Args:
            width: Ancho de la imagen
            height: Alto de la imagen
            message: Mensaje de error a mostrar
            
        Returns:
            torch.Tensor: Tensor con imagen de error
        """
        try:
            color = tuple(self.config.get("error_image_color", [255, 0, 0]))
            error_img = Image.new('RGB', (width, height), color=color)
            
            # Añadir texto de error si es posible
            try:
                draw = ImageDraw.Draw(error_img)
                # Usar fuente por defecto
                font_size = min(width, height) // 20
                draw.text((width//2, height//2), message, 
                         fill=(255, 255, 255), anchor="mm")
            except Exception:
                pass  # Si no se puede añadir texto, continuar sin él
                
            return self.pil_to_tensor(error_img)
        except Exception as e:
            logger.error(f"Error creando imagen de error: {e}")
            # Fallback: tensor de ceros
            return torch.zeros(1, 3, height, width)

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convierte un tensor ComfyUI a una imagen PIL con validación mejorada.
        
        Args:
            tensor: Tensor de ComfyUI con formato [C, H, W] o [1, C, H, W]
            
        Returns:
            Image.Image: Imagen PIL en formato RGB
            
        Raises:
            ValueError: Si el tensor no tiene el formato esperado
        """
        try:
            # Normalizar dimensiones
            if tensor.ndim == 4:
                if tensor.shape[0] != 1:
                    raise ValueError(f"Batch size debe ser 1, es {tensor.shape[0]}")
                tensor = tensor[0]
            elif tensor.ndim != 3:
                raise ValueError(f"Tensor debe tener 3 o 4 dimensiones, tiene {tensor.ndim}")
            
            # Validar formato de canales
            if tensor.shape[0] not in [1, 3]:
                raise ValueError(f"Tensor debe tener 1 o 3 canales, tiene {tensor.shape[0]}")
            
            # Convertir a numpy con clipping seguro
            array = tensor.mul(255).clamp(0, 255).byte().cpu().numpy()
            
            # Manejar grayscale
            if array.shape[0] == 1:
                array = np.repeat(array, 3, axis=0)
            
            # Transponer a formato PIL (H, W, C)
            array = np.transpose(array, (1, 2, 0))
            return Image.fromarray(array, mode='RGB')
            
        except Exception as e:
            logger.error(f"Error en tensor_to_pil: {e}")
            raise

    @staticmethod
    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """
        Convierte una imagen PIL a tensor ComfyUI con validación mejorada.
        
        Args:
            img: Imagen PIL a convertir
            
        Returns:
            torch.Tensor: Tensor en formato [1, C, H, W] con valores en rango 0-1
            
        Raises:
            ValueError: Si la imagen no es válida
        """
        try:
            if img is None:
                raise ValueError("La imagen no puede ser None")
            
            # Asegurar formato RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convertir a array numpy
            array = np.array(img, dtype=np.float32) / 255.0
            
            # Validar dimensiones
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"Imagen debe ser RGB con 3 canales, tiene shape {array.shape}")
            
            # Convertir a formato tensor (C, H, W)
            array = np.transpose(array, (2, 0, 1))
            
            # Crear tensor y añadir dimensión batch
            return torch.from_numpy(array).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Error en pil_to_tensor: {e}")
            raise

    def combine_images_horizontally(self, img1: Image.Image, 
                                  img2: Image.Image) -> Image.Image:
        """
        Combina dos imágenes PIL horizontalmente con configuración personalizable.
        
        Args:
            img1: Primera imagen (lado izquierdo)
            img2: Segunda imagen (lado derecho)
            
        Returns:
            Image.Image: Imagen combinada horizontalmente
            
        Raises:
            ValueError: Si alguna imagen es None o inválida
        """
        try:
            if img1 is None or img2 is None:
                raise ValueError("Las imágenes no pueden ser None")
            
            # Convertir a RGB si es necesario
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Calcular dimensiones - igualar altura
            target_height = max(img1.height, img2.height)
            
            # Redimensionar manteniendo aspecto
            if img1.height != target_height:
                ratio = target_height / img1.height
                new_width = int(img1.width * ratio)
                img1 = img1.resize((new_width, target_height), Image.LANCZOS)
            
            if img2.height != target_height:
                ratio = target_height / img2.height
                new_width = int(img2.width * ratio)
                img2 = img2.resize((new_width, target_height), Image.LANCZOS)
            
            # Crear imagen combinada
            total_width = img1.width + img2.width
            bg_color = tuple(self.config.get("combine_background_color", [255, 255, 255]))
            
            combined = Image.new("RGB", (total_width, target_height), color=bg_color)
            combined.paste(img1, (0, 0))
            combined.paste(img2, (img1.width, 0))
            
            logger.info(f"Imágenes combinadas: {total_width}x{target_height}")
            return combined
            
        except Exception as e:
            logger.error(f"Error en combine_images_horizontally: {e}")
            raise

    def _resize_image_if_needed(self, img: Image.Image, max_size: int) -> Image.Image:
        """
        Redimensiona imagen si excede el tamaño máximo.
        
        Args:
            img: Imagen a redimensionar
            max_size: Tamaño máximo permitido
            
        Returns:
            Image.Image: Imagen redimensionada si era necesario
        """
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
        """
        Valida todos los inputs del nodo con verificaciones exhaustivas.
        
        Args:
            image_1: Tensor de imagen principal
            image_2: Tensor de imagen secundaria (opcional)
            prompt: Prompt de edición
            input_fidelity: Configuración de fidelidad
            quality: Configuración de calidad
            output_format: Formato de salida
            max_size: Tamaño máximo
            
        Raises:
            ValueError: Si algún input no es válido
        """
        # Validar imagen principal
        if not isinstance(image_1, torch.Tensor):
            raise ValueError("image_1 debe ser un torch.Tensor")
        if image_1.ndim not in (3, 4):
            raise ValueError("image_1 debe tener 3 o 4 dimensiones")
        if image_1.shape[-3] not in (1, 3):
            raise ValueError("image_1 debe tener 1 o 3 canales")
        
        # Validar imagen secundaria si existe
        if image_2 is not None:
            if not isinstance(image_2, torch.Tensor):
                raise ValueError("image_2 debe ser un torch.Tensor")
            if image_2.ndim not in (3, 4):
                raise ValueError("image_2 debe tener 3 o 4 dimensiones")
            if image_2.shape[-3] not in (1, 3):
                raise ValueError("image_2 debe tener 1 o 3 canales")
        
        # Validar prompt
        if not isinstance(prompt, str) or len(prompt.strip()) < 3:
            raise ValueError("El prompt debe ser una cadena de al menos 3 caracteres")
        
        # Validar parámetros de configuración
        if input_fidelity not in ["low", "high"]:
            raise ValueError("input_fidelity debe ser 'low' o 'high'")
        if quality not in ["standard", "hd"]:
            raise ValueError("quality debe ser 'standard' o 'hd'")
        if output_format not in ["png", "jpeg", "webp"]:
            raise ValueError("output_format debe ser 'png', 'jpeg' o 'webp'")
        if not isinstance(max_size, int) or max_size < 256 or max_size > 2048:
            raise ValueError("max_size debe ser un entero entre 256 y 2048")

    def _get_openai_client(self, api_key: Optional[str] = None) -> OpenAI:
        """
        Obtiene cliente OpenAI con gestión de API key.
        
        Args:
            api_key: API key opcional para guardar
            
        Returns:
            OpenAI: Cliente configurado
            
        Raises:
            ValueError: Si no se puede obtener API key válida
        """
        # Guardar nueva API key si se proporciona
        if api_key and api_key.strip():
            self._save_api_key(api_key)
        
        # Obtener API key en orden de prioridad
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
        
        # Crear o reutilizar cliente
        if self._client is None or getattr(self._client, '_api_key', None) != final_api_key:
            timeout = self.config.get("timeout", 30)
            self._client = OpenAI(
                api_key=final_api_key,
                timeout=timeout
            )
            logger.info("Cliente OpenAI inicializado")
        
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
        Ejecuta la edición de imagen usando la API de OpenAI con todas las mejoras.
        
        Args:
            image_1: Tensor de imagen principal
            prompt: Descripción de la edición deseada
            image_2: Tensor de imagen secundaria (opcional)
            api_key: Clave API de OpenAI (opcional)
            input_fidelity: Calidad de procesamiento ("low" o "high")
            quality: Calidad de salida ("standard" o "hd")
            output_format: Formato de salida ("png", "jpeg", "webp")
            max_size: Tamaño máximo en píxeles
            enable_cache: Habilitar sistema de cache
            
        Returns:
            Tuple[torch.Tensor]: Tupla con la imagen editada
        """
        logger.info("=== Iniciando OpenAI Image Edit ===")
        
        # 1. Validación de inputs
        try:
            self._validate_inputs(image_1, image_2, prompt, input_fidelity, 
                                quality, output_format, max_size)
            logger.info("✓ Inputs validados correctamente")
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

        # 3. Verificar cache si está habilitado
        if enable_cache and self._cache is not None:
            try:
                image_hash = self._get_image_hash(combined_img)
                cache_key = self._get_cache_key(prompt, input_fidelity, quality, image_hash)
                cached_result = self._check_cache(cache_key)
                
                if cached_result is not None:
                    logger.info("✓ Resultado encontrado en cache")
                    return (cached_result,)
                    
            except Exception as e:
                logger.warning(f"Error en cache: {e}")

        # 4. Obtener cliente OpenAI
        try:
            client = self._get_openai_client(api_key)
            logger.info("✓ Cliente OpenAI listo")
        except Exception as e:
            logger.error(f"✗ Error inicializando cliente: {e}")
            return (self.create_error_image(message="API ERROR"),)

        # 5. Preparar imagen para API
        try:
            with self._image_buffer() as img_bytes:
                combined_img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                
                logger.info("✓ Imagen preparada para API")
                
                # 6. Llamar a OpenAI API
                logger.info("⏳ Enviando a OpenAI API...")
                response = client.images.edit(
                    image=img_bytes,
                    prompt=prompt.strip(),
                    input_fidelity=input_fidelity,
                    quality=quality
                )
                logger.info("✓ Respuesta recibida de OpenAI")
                
        except Exception as e:
            logger.error(f"✗ Error en API call: {e}")
            return (self.create_error_image(message="API CALL ERROR"),)

        # 7. Procesar respuesta
        try:
            if not (hasattr(response, 'data') and response.data and 
                    hasattr(response.data[0], 'b64_json')):
                raise ValueError("Respuesta de API inválida")
            
            b64_data = response.data[0].b64_json
            img_data = base64.b64decode(b64_data)
            result_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Convertir a tensor
            result_tensor = self.pil_to_tensor(result_img)
            logger.info("✓ Imagen procesada exitosamente")
            
            # 8. Guardar en cache si está habilitado
            if enable_cache and self._cache is not None:
                try:
                    self._save_to_cache(cache_key, result_tensor)
                except Exception as e:
                    logger.warning(f"Error guardando en cache: {e}")
            
        except Exception as e:
            logger.error(f"✗ Error procesando respuesta: {e}")
            return (self.create_error_image(message="RESPONSE ERROR"),)

        # 9. Cleanup
        try:
            if 'combined_img' in locals():
                combined_img.close()
            if 'result_img' in locals():
                result_img.close()
        except Exception as e:
            logger.warning(f"Warning en cleanup: {e}")

        logger.info("=== Ejecución completada exitosamente ===")
        return (result_tensor,)

# --- Endpoint web opcional para configuración ---
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/openai_image_edit/status")
    async def get_status(request):
        """Endpoint para verificar estado del nodo."""
        return web.json_response({
            "status": "active",
            "message": "OpenAI Image Edit Node funcionando",
            "version": "2.0.0",
            "config_loaded": os.path.exists(CONFIG_FILE),
            "api_key_configured": os.path.exists(API_KEY_FILE)
        })
        
    @PromptServer.instance.routes.post("/openai_image_edit/clear_cache")
    async def clear_cache(request):
        """Endpoint para limpiar cache."""
        try:
            # Esto requeriría acceso a la instancia del nodo
            return web.json_response({"message": "Cache limpiado"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
            
except ImportError:
    logger.info("Endpoints web no disponibles (normal en algunos entornos)")

# --- Exportación de nodos para ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "OpenAIImageEditNode": OpenAIImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageEditNode": "OpenAI Image Edit",
}