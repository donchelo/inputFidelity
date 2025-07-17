# Nodo ComfyUI: OpenAI Image Edit

Este nodo permite editar imágenes usando la API de OpenAI con input_fidelity="high" y combinar dos imágenes como entrada.

## Instalación

1. Copia la carpeta `openai_image_edit` dentro de `custom_nodes/` de tu instalación de ComfyUI.
2. Instala las dependencias:
   ```bash
   pip install -r custom_nodes/openai_image_edit/requirements.txt
   ```
3. Asegúrate de tener la variable de entorno `OPENAI_API_KEY` configurada o proporciona la API key en el nodo.

## Uso

- Inputs requeridos:
  - `image_1`: Primera imagen (tensor ComfyUI)
  - `image_2`: Segunda imagen (tensor ComfyUI)
  - `prompt`: Descripción de la edición
- Inputs opcionales:
  - `api_key`: Tu clave de OpenAI (opcional si tienes la variable de entorno)
  - `input_fidelity`: "low" o "high" (por defecto: "high")
  - `quality`: "standard" o "hd" (por defecto: "hd")
- Output:
  - Imagen editada (tensor ComfyUI)

## Notas
- El nodo combina las dos imágenes horizontalmente antes de enviarlas a la API.
- Requiere una cuenta de OpenAI con acceso a la API de edición de imágenes. 