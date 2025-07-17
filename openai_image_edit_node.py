# Autor: chelo
import os
import io
import base64
import logging
from typing import Any, Tuple, Optional, Dict
from PIL import Image
import numpy as np
import torch
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY_FILE = os.path.join(os.path.dirname(__file__), ".openai_api_key")

class OpenAIImageEditNode:
    """
    Nodo personalizado para ComfyUI que edita imágenes usando la API de OpenAI.
    
    Funcionalidad:
    - Edita imágenes usando prompts de texto a través de OpenAI API
    - Puede combinar dos imágenes horizontalmente antes del procesamiento
    - Maneja automáticamente la gestión de claves API
    - Soporte para configuración de calidad y fidelidad
    
    API Key Management:
    - Si se proporciona en el campo del nodo, se guarda automáticamente
    - Si el campo está vacío, se intenta leer la clave guardada
    - Fallback a variable de entorno OPENAI_API_KEY
    """
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "execute"
    CATEGORY = "image/ai"
    DESCRIPTION = "Edita imágenes usando OpenAI API con soporte para combinación de imágenes"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define los tipos de entrada para el nodo.
        
        Returns:
            Dict: Configuración de inputs con required y optional
        """
        return {
            "required": {
                "image_1": ("IMAGE", {
                    "tooltip": "Imagen principal a editar"
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Describe la edición a realizar...",
                    "tooltip": "Descripción de la edición deseada"
                }),
            },
            "optional": {
                "image_2": ("IMAGE", {
                    "tooltip": "Segunda imagen para combinación horizontal (opcional)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Clave API de OpenAI (se guarda automáticamente)"
                }),
                "input_fidelity": (["low", "high"], {
                    "default": "high",
                    "tooltip": "Calidad de procesamiento de entrada"
                }),
                "quality": (["standard", "hd"], {
                    "default": "hd",
                    "tooltip": "Calidad de la imagen de salida"
                }),
            }
        }

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convierte un tensor ComfyUI a una imagen PIL.
        
        Args:
            tensor: Tensor de ComfyUI con formato [C, H, W] o [1, C, H, W]
            
        Returns:
            Image.Image: Imagen PIL en formato RGB
            
        Raises:
            ValueError: Si el tensor no tiene el formato esperado
        """
        try:
            if tensor.ndim == 4:
                tensor = tensor[0]
            elif tensor.ndim != 3:
                raise ValueError(f"Tensor debe tener 3 o 4 dimensiones, tiene {tensor.ndim}")
                
            array = tensor.mul(255).clamp(0, 255).byte().cpu().numpy()
            
            # Convertir grayscale a RGB si es necesario
            if array.shape[0] == 1:
                array = np.repeat(array, 3, axis=0)
            elif array.shape[0] != 3:
                raise ValueError(f"Tensor debe tener 1 o 3 canales, tiene {array.shape[0]}")
                
            array = np.transpose(array, (1, 2, 0))
            return Image.fromarray(array)
        except Exception as e:
            logger.error(f"Error en tensor_to_pil: {e}")
            raise

    @staticmethod
    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """
        Convierte una imagen PIL a tensor ComfyUI.
        
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
                
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            array = np.array(img).astype(np.float32) / 255.0
            
            # Manejar diferentes formatos
            if array.ndim == 2:
                array = np.expand_dims(array, axis=-1)
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            elif array.shape[2] != 3:
                raise ValueError(f"Imagen debe tener 1 o 3 canales, tiene {array.shape[2]}")
                
            array = np.transpose(array, (2, 0, 1))
            return torch.from_numpy(array).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error en pil_to_tensor: {e}")
            raise

    @staticmethod
    def combine_images_horizontally(img1: Image.Image, img2: Image.Image) -> Image.Image:
        """
        Combina dos imágenes PIL horizontalmente.
        
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
                
            # Asegurar que ambas imágenes estén en RGB
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
                
            h = max(img1.height, img2.height)
            w = img1.width + img2.width
            
            new_img = Image.new("RGB", (w, h), color=(255, 255, 255))
            new_img.paste(img1, (0, 0))
            new_img.paste(img2, (img1.width, 0))
            
            return new_img
        except Exception as e:
            logger.error(f"Error en combine_images_horizontally: {e}")
            raise

    @staticmethod
    def _save_api_key(api_key: str) -> bool:
        """
        Guarda la clave API en un archivo local.
        
        Args:
            api_key: Clave API a guardar
            
        Returns:
            bool: True si se guardó exitosamente, False en caso contrario
        """
        try:
            with open(API_KEY_FILE, "w") as f:
                f.write(api_key.strip())
            logger.info("API key guardada exitosamente")
            return True
        except Exception as e:
            logger.warning(f"No se pudo guardar la API key: {e}")
            return False

    @staticmethod
    def _load_api_key() -> Optional[str]:
        """
        Carga la clave API desde el archivo local.
        
        Returns:
            Optional[str]: Clave API si existe, None en caso contrario
        """
        try:
            if os.path.exists(API_KEY_FILE):
                with open(API_KEY_FILE, "r") as f:
                    key = f.read().strip()
                    if key:
                        logger.info("API key cargada desde archivo")
                        return key
        except Exception as e:
            logger.warning(f"No se pudo cargar la API key: {e}")
        return None

    def _validate_inputs(self, 
                        image_1: torch.Tensor, 
                        image_2: Optional[torch.Tensor],
                        prompt: str,
                        input_fidelity: str,
                        quality: str) -> None:
        """
        Valida los inputs del nodo.
        
        Args:
            image_1: Tensor de imagen principal
            image_2: Tensor de imagen secundaria (opcional)
            prompt: Prompt de edición
            input_fidelity: Configuración de fidelidad
            quality: Configuración de calidad
            
        Raises:
            ValueError: Si algún input no es válido
        """
        # Validar imagen principal
        if not isinstance(image_1, torch.Tensor):
            raise ValueError("image_1 debe ser un torch.Tensor")
        if image_1.ndim not in (3, 4):
            raise ValueError("image_1 debe tener 3 o 4 dimensiones (C,H,W) o (1,C,H,W)")
        if image_1.shape[-3] not in (1, 3):
            raise ValueError("image_1 debe tener 1 o 3 canales (grayscale o RGB)")
            
        # Validar imagen secundaria si existe
        if image_2 is not None:
            if not isinstance(image_2, torch.Tensor):
                raise ValueError("image_2 debe ser un torch.Tensor si se proporciona")
            if image_2.ndim not in (3, 4):
                raise ValueError("image_2 debe tener 3 o 4 dimensiones (C,H,W) o (1,C,H,W)")
            if image_2.shape[-3] not in (1, 3):
                raise ValueError("image_2 debe tener 1 o 3 canales (grayscale o RGB)")
                
        # Validar prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("El prompt debe ser una cadena no vacía")
            
        # Validar parámetros de configuración
        if input_fidelity not in ["low", "high"]:
            raise ValueError("input_fidelity debe ser 'low' o 'high'")
        if quality not in ["standard", "hd"]:
            raise ValueError("quality debe ser 'standard' o 'hd'")

    def execute(self,
                image_1: torch.Tensor,
                prompt: str,
                image_2: Optional[torch.Tensor] = None,
                api_key: Optional[str] = None,
                input_fidelity: str = "high",
                quality: str = "hd"
                ) -> Tuple[torch.Tensor]:
        """
        Ejecuta la edición de imagen usando la API de OpenAI.
        
        Args:
            image_1: Tensor de imagen principal
            prompt: Descripción de la edición deseada
            image_2: Tensor de imagen secundaria (opcional)
            api_key: Clave API de OpenAI (opcional)
            input_fidelity: Calidad de procesamiento ("low" o "high")
            quality: Calidad de salida ("standard" o "hd")
            
        Returns:
            Tuple[torch.Tensor]: Tupla con la imagen editada
            
        Raises:
            ValueError: Si los inputs no son válidos
            RuntimeError: Si la API call falla
        """
        logger.info("Iniciando ejecución de OpenAI Image Edit")
        
        # --- Input validation ---
        try:
            self._validate_inputs(image_1, image_2, prompt, input_fidelity, quality)
        except ValueError as e:
            logger.error(f"Error de validación: {e}")
            return (None,)

        # 1. Obtener y guardar API key si es necesario
        try:
            if api_key and api_key.strip():
                self._save_api_key(api_key)
            api_key_final = api_key.strip() if api_key and api_key.strip() else self._load_api_key() or os.getenv("OPENAI_API_KEY")
            if not api_key_final:
                logger.error("No se proporcionó una clave API de OpenAI")
                return (None,)
            client = OpenAI(api_key=api_key_final)
            logger.info("Cliente OpenAI inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar cliente OpenAI: {e}")
            return (None,)

        # 2. Convertir tensores a PIL y procesar imágenes
        try:
            img1 = self.tensor_to_pil(image_1)
            if image_2 is not None:
                img2 = self.tensor_to_pil(image_2)
                # 3. Combinar imágenes horizontalmente
                combined_img = self.combine_images_horizontally(img1, img2)
                logger.info("Imágenes combinadas horizontalmente")
            else:
                combined_img = img1
                logger.info("Usando imagen única")

            # 4. Convertir a bytes para API
            img_bytes = io.BytesIO()
            combined_img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            logger.info("Imagen preparada para API")
        except Exception as e:
            logger.error(f"Error al procesar imágenes: {e}")
            return (None,)

        # 5. Llamar a la API de OpenAI con manejo de errores
        try:
            logger.info("Enviando solicitud a OpenAI API")
            response = client.images.edit(
                image=img_bytes,
                prompt=prompt,
                input_fidelity=input_fidelity,
                quality=quality
            )
            logger.info("Respuesta recibida de OpenAI API")
        except Exception as e:
            logger.error(f"Error en llamada a OpenAI API: {e}")
            return (None,)

        # 6. Procesar respuesta base64 a PIL con manejo de errores
        try:
            if not hasattr(response, 'data') or not response.data or not hasattr(response.data[0], 'b64_json'):
                logger.error("La respuesta de OpenAI API no tiene los campos esperados")
                return (None,)
            b64_data = response.data[0].b64_json
            img_data = base64.b64decode(b64_data)
            result_img = Image.open(io.BytesIO(img_data)).convert("RGB")
            logger.info("Imagen procesada exitosamente")
        except Exception as e:
            logger.error(f"Error al procesar respuesta de OpenAI API: {e}")
            return (None,)

        # 7. Convertir a tensor ComfyUI
        try:
            result_tensor = self.pil_to_tensor(result_img)
            logger.info("Imagen convertida a tensor ComfyUI")
        except Exception as e:
            logger.error(f"Error al convertir imagen a tensor: {e}")
            return (None,)

        # 8. Cleanup de recursos
        try:
            img_bytes.close()
            if 'combined_img' in locals():
                combined_img.close()
            if 'result_img' in locals():
                result_img.close()
        except Exception as e:
            logger.warning(f"Error en cleanup de recursos: {e}")

        # 9. Retornar imagen
        logger.info("Ejecución completada exitosamente")
        return (result_tensor,)

# --- Ejemplo de endpoint web opcional ---
try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/openai_image_edit/hello")
    async def get_hello(request):
        return web.json_response({"message": "Hola desde OpenAI Image Edit Node!"})
except Exception:
    pass

# --- Exportación de nodos para ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "OpenAIImageEditNode": OpenAIImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageEditNode": "OpenAI Image Edit (Combinado)"
} 