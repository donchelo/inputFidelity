import os
import io
import base64
from typing import Any, Tuple
from PIL import Image
import numpy as np
import torch
from openai import OpenAI

class OpenAIImageEditNode:
    """
    Nodo personalizado para ComfyUI que edita imágenes usando la API de OpenAI.
    Puede combinar dos imágenes horizontalmente o usar solo una, según los inputs.
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
        # 1. Validar API key
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No se proporcionó la API key de OpenAI.")
        client = OpenAI(api_key=api_key)

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

        # 5. Llamar a la API de OpenAI
        response = client.images.edit(
            image=img_bytes,
            prompt=prompt,
            input_fidelity=input_fidelity,
            quality=quality
        )
        # 6. Procesar respuesta base64 a PIL
        b64_data = response.data[0].b64_json
        img_data = base64.b64decode(b64_data)
        result_img = Image.open(io.BytesIO(img_data)).convert("RGB")

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