# OpenAI Image Edit Node para ComfyUI

Un nodo personalizado para ComfyUI que permite editar imágenes usando la API de OpenAI con soporte completo para `input_fidelity="high"` para preservar detalles de caras, logos y texturas.

## 🚀 Instalación

### Método 1: ComfyUI Manager (Recomendado)
1. Abre ComfyUI Manager
2. Busca "OpenAI Image Edit Node"
3. Haz clic en "Install"
4. Reinicia ComfyUI

### Método 2: Instalación Manual
1. Copia esta carpeta a `custom_nodes/` de tu instalación de ComfyUI
2. Instala las dependencias:
   ```bash
   # Para ComfyUI Portable:
   python_embeded\python.exe -m pip install -r requirements.txt
   
   # Para ComfyUI Desktop:
   pip install -r requirements.txt
   ```
3. Reinicia ComfyUI

## 📋 Dependencias

- `openai>=1.100.0` - Cliente oficial de OpenAI
- `pillow>=8.0.0` - Procesamiento de imágenes
- `torch>=1.13.0` - Tensores de PyTorch
- `numpy>=1.21.0` - Operaciones numéricas
- `requests>=2.25.0` - Peticiones HTTP

## 🎯 Uso

### Inputs Requeridos:
- **`image_1`**: Imagen principal a editar (tensor ComfyUI)
- **`prompt`**: Descripción detallada de la edición deseada

### Inputs Opcionales:
- **`image_2`**: Segunda imagen (tensor ComfyUI) - Útil para combinar elementos
- **`api_key`**: Tu clave de OpenAI (opcional si tienes la variable de entorno)
- **`input_fidelity`**: "low" o "high" (por defecto: "high") - **CRÍTICO** para preservar detalles
- **`quality`**: "standard" o "high" (por defecto: "high")
- **`output_format`**: "png", "jpeg", "webp" (por defecto: "png")
- **`max_size`**: Tamaño máximo en píxeles (256-2048, por defecto: 1024)
- **`enable_cache`**: Habilitar cache para mejorar rendimiento
- **`force_update_client`**: Forzar actualización del cliente OpenAI
- **`combine_images`**: Combinar imágenes horizontalmente (útil para múltiples elementos)

### Output:
- **`edited_image`**: Imagen editada (tensor ComfyUI)

## ⚙️ Configuración

### API Key
Puedes configurar tu API key de OpenAI de tres formas:
1. **Variable de entorno**: `OPENAI_API_KEY`
2. **Archivo seguro**: Se guarda automáticamente en `openai_api_key.txt`
3. **Input del nodo**: Proporcionar directamente en el nodo

### Archivo de Configuración
El nodo crea automáticamente un archivo `config.json` con configuraciones personalizables:
```json
{
    "default_quality": "high",
    "default_fidelity": "high",
    "default_output_format": "png",
    "max_image_size": 2048,
    "timeout": 60,
    "cache_enabled": true,
    "max_cache_size": 10
}
```

## 🔧 Características Avanzadas

### Sistema de Cache
- Cache automático para mejorar rendimiento
- Configurable desde el archivo de configuración
- Limpieza automática cuando se excede el tamaño máximo

### Manejo de Errores
- Imágenes de error informativas
- Logging detallado para debugging
- Validación de inputs y API responses

### Soporte para Diferentes Formatos
- PNG (recomendado para transparencias)
- JPEG (compresión)
- WebP (formato moderno)

## 🐛 Solución de Problemas

### Error: "API key not found"
1. Verifica que tu API key sea válida
2. Asegúrate de que tenga permisos para la API de edición de imágenes
3. Revisa que el formato sea correcto (debe empezar con `sk-`)

### Error: "Input fidelity not supported"
1. Actualiza la biblioteca OpenAI: `pip install --upgrade openai`
2. Verifica que tu versión de OpenAI sea >= 1.100.0

### Error: "Image too large"
1. Reduce el valor de `max_size`
2. ComfyUI redimensionará automáticamente la imagen si es necesario

## 📝 Notas Importantes

- **input_fidelity="high"**: Preserva detalles de caras, logos y texturas según la documentación oficial de OpenAI
- **Cache**: Se recomienda mantener habilitado para mejor rendimiento
- **Formato PNG**: Recomendado para mantener transparencias
- **Tamaño máximo**: 2048x2048 píxeles según limitaciones de la API

## 🎨 Ejemplos de Uso (Basados en Documentación Oficial)

### Edición Precisa
- **Cambio de elementos**: Cambiar el color de objetos específicos sin afectar el resto
- **Eliminación de elementos**: Remover objetos de forma limpia
- **Adición de elementos**: Insertar nuevos objetos de forma natural

### Preservación de Caras
- **Edición de fotos**: Editar fotos manteniendo rasgos faciales
- **Personalización**: Crear avatares que mantengan la identidad
- **Combinación de caras**: Fusionar caras de múltiples fotos

### Consistencia de Marca
- **Assets de marketing**: Generar banners con logos sin distorsión
- **Mockups**: Colocar logos en escenas realistas
- **Fotografía de productos**: Cambiar fondos manteniendo detalles del producto

### Retoque de Moda y Productos
- **Variaciones de outfit**: Cambiar colores o estilos de ropa
- **Adición de accesorios**: Agregar joyas, sombreros, etc.
- **Extracción de productos**: Mostrar productos en nuevos entornos

## 🤝 Contribuir

Si encuentras bugs o tienes sugerencias, por favor:
1. Abre un issue en GitHub
2. Incluye logs detallados del error
3. Describe los pasos para reproducir el problema

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo LICENSE para más detalles.

## 🙏 Agradecimientos

- ComfyUI por el framework
- OpenAI por la API de edición de imágenes
- La comunidad de ComfyUI por el soporte 