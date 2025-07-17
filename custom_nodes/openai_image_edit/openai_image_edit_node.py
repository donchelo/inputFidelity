import os
import io
import base64
from typing import Any, Tuple
from PIL import Image
import numpy as np
import torch
from openai import OpenAI

API_KEY_FILE = os.path.join(os.path.dirname(__file__), ".openai_api_key")

class OpenAIImageEditNode:
    """
    Nodo personalizado para ComfyUI que edita imágenes usando la API de OpenAI.
    Puede combinar dos imágenes horizontalmente o usar solo una, según los inputs.

    API Key:
    - Si el usuario la escribe en el campo del nodo, se guarda automáticamente en un archivo oculto interno.
    - Si el campo está vacío, se intenta leer la clave guardada.
    - Si no existe, se usa la variable de entorno OPENAI_API_KEY.
    """
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Imagen editada",)
    FUNCTION = "edit_image"
    CATEGORY = "image/ai"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Permite image_1 como requerida y image_2 como opcional.
        """
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe la edición a realizar..."}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "input_fidelity": (["low", "high"], {"default": "high"}),
                "quality": (["standard", "hd"], {"default": "hd"}),
            }
        }

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convierte un tensor ComfyUI [C, H, W] o [1, C, H, W] a una imagen PIL.
        """
        if tensor.ndim == 4:
            tensor = tensor[0]
        array = tensor.mul(255).clamp(0, 255).byte().cpu().numpy()
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        array = np.transpose(array, (1, 2, 0))
        return Image.fromarray(array)

    @staticmethod
    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """
        Convierte una imagen PIL a tensor ComfyUI [1, C, H, W] en rango 0-1.
        """
        array = np.array(img).astype(np.float32) / 255.0
        if array.ndim == 2:
            array = np.expand_dims(array, axis=-1)
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array).unsqueeze(0)

    @staticmethod
    def combine_images_horizontally(img1: Image.Image, img2: Image.Image) -> Image.Image:
        """
        Combina dos imágenes PIL horizontalmente.
        """
        h = max(img1.height, img2.height)
        w = img1.width + img2.width
        new_img = Image.new("RGB", (w, h))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
        return new_img

    @staticmethod
    def _save_api_key(api_key: str):
        try:
            with open(API_KEY_FILE, "w") as f:
                f.write(api_key.strip())
        except Exception:
            pass  # No interrumpir el flujo si no se puede guardar

    @staticmethod
    def _load_api_key() -> str:
        try:
            if os.path.exists(API_KEY_FILE):
                with open(API_KEY_FILE, "r") as f:
                    return f.read().strip()
        except Exception:
            pass
        return None

    @classmethod
    def edit_image(cls,
                   image_1: torch.Tensor,
                   prompt: str,
                   image_2: torch.Tensor = None,
                   api_key: str = None,
                   input_fidelity: str = "high",
                   quality: str = "hd"
                   ) -> Tuple[torch.Tensor]:
        """
        Edita la imagen (una o dos) usando la API de OpenAI y retorna el resultado como tensor ComfyUI.
        """
        # --- Input validation ---
        if not isinstance(image_1, torch.Tensor):
            raise ValueError("image_1 must be a torch.Tensor.")
        if image_1.ndim not in (3, 4):
            raise ValueError("image_1 must have 3 or 4 dimensions (C,H,W) or (1,C,H,W).")
        if image_1.shape[-3] not in (1, 3):
            raise ValueError("image_1 must have 1 or 3 channels (grayscale or RGB).")
        if image_2 is not None:
            if not isinstance(image_2, torch.Tensor):
                raise ValueError("image_2 must be a torch.Tensor if provided.")
            if image_2.ndim not in (3, 4):
                raise ValueError("image_2 must have 3 or 4 dimensions (C,H,W) or (1,C,H,W).")
            if image_2.shape[-3] not in (1, 3):
                raise ValueError("image_2 must have 1 or 3 channels (grayscale or RGB).")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        # 1. Obtener y guardar API key si es necesario
        if api_key and api_key.strip():
            cls._save_api_key(api_key)
        api_key_final = api_key.strip() if api_key and api_key.strip() else cls._load_api_key() or os.getenv("OPENAI_API_KEY")
        if not api_key_final:
            raise ValueError("OpenAI API key was not provided.")
        client = OpenAI(api_key=api_key_final)

        # 2. Convertir tensores a PIL
        img1 = cls.tensor_to_pil(image_1)
        if image_2 is not None:
            img2 = cls.tensor_to_pil(image_2)
            # 3. Combinar imágenes horizontalmente
            combined_img = cls.combine_images_horizontally(img1, img2)
        else:
            combined_img = img1

        # 4. Convertir a bytes para API
        img_bytes = io.BytesIO()
        combined_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # 5. Llamar a la API de OpenAI con manejo de errores
        try:
            response = client.images.edit(
                image=img_bytes,
                prompt=prompt,
                input_fidelity=input_fidelity,
                quality=quality
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

        # 6. Procesar respuesta base64 a PIL con manejo de errores
        try:
            if not hasattr(response, 'data') or not response.data or not hasattr(response.data[0], 'b64_json'):
                raise RuntimeError("OpenAI API response is missing expected fields.")
            b64_data = response.data[0].b64_json
            img_data = base64.b64decode(b64_data)
            result_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to process OpenAI API response: {str(e)}")

        # 7. Convertir a tensor ComfyUI
        result_tensor = cls.pil_to_tensor(result_img)

        # 8. Retornar imagen
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