# Ejemplos de Uso - OpenAI Image Edit Node

Este archivo contiene ejemplos prácticos basados en la documentación oficial de OpenAI para `input_fidelity="high"`.

## 🎯 Edición Precisa

### Cambio de Color de Objetos
```
Prompt: "Make the mug olive green"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Cambiar el color de objetos específicos sin afectar el resto de la imagen.

### Eliminación de Elementos
```
Prompt: "Remove the mug from the desk"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Remover objetos de forma limpia sin afectar el fondo.

### Adición de Elementos
```
Prompt: "Add a post-it note saying 'Be right back!' to the monitor"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Insertar nuevos objetos de forma natural en la escena.

## 👤 Preservación de Caras

### Edición de Fotos con Preservación Facial
```
Prompt: "Add soft neon purple and lime green lighting and glowing backlighting"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Editar fotos manteniendo rasgos faciales intactos.

### Creación de Avatares
```
Prompt: "Generate an avatar of this person in digital art style, with vivid splash of colors"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Crear avatares que mantengan la identidad de la persona.

### Combinación de Múltiples Caras
```
Prompt: "Put these two women in the same picture, holding shoulders, as if part of the same photo"
Configuración: input_fidelity="high", quality="high", combine_images=True
```
**Uso**: Fusionar caras de múltiples fotos en una sola imagen.

## 🏢 Consistencia de Marca

### Assets de Marketing
```
Prompt: "Generate a beautiful, modern hero banner featuring this logo in the center. It should look futuristic, with blue & violet hues"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Generar banners con logos sin distorsión.

### Mockups Realistas
```
Prompt: "Generate a highly realistic picture of a hand holding a tilted iphone, with an app on the screen that showcases this logo in the center with a loading animation below"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Colocar logos en escenas realistas.

### Fotografía de Productos
```
Prompt: "Generate a beautiful ad with this bag in the center, on top of a dark background with a glowing halo emanating from the center, behind the bag"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Cambiar fondos manteniendo detalles del producto.

## 👗 Retoque de Moda y Productos

### Variaciones de Outfit
```
Prompt: "Edit this picture so that the model wears a blue tank top instead of the coat and sweater"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Cambiar colores o estilos de ropa manteniendo la pose.

### Adición de Accesorios
```
Prompt: "Add the crossbody bag to the outfit"
Configuración: input_fidelity="high", quality="high", combine_images=True
```
**Uso**: Agregar accesorios sin alterar la pose o cara del modelo.

### Extracción de Productos
```
Prompt: "Generate a picture of this exact same jacket on a white background"
Configuración: input_fidelity="high", quality="high"
```
**Uso**: Mostrar productos en nuevos entornos manteniendo detalles.

## ⚙️ Configuraciones Recomendadas

### Para Caras y Logos
```python
input_fidelity = "high"      # CRÍTICO para preservar detalles
quality = "high"             # Mejor resolución
output_format = "png"        # Preserva transparencias
```

### Para Productos y Moda
```python
input_fidelity = "high"      # Preserva texturas y detalles
quality = "high"             # Calidad profesional
output_format = "png"        # Sin pérdida de calidad
```

### Para Combinación de Imágenes
```python
combine_images = True        # Combina horizontalmente
input_fidelity = "high"      # Preserva detalles de ambas
quality = "high"             # Máxima calidad
```

## 💡 Consejos Importantes

1. **Primera imagen**: La primera imagen proporcionada preserva el máximo detalle
2. **Combinación**: Usa `combine_images=True` para múltiples elementos
3. **Prompts específicos**: Sé específico sobre qué cambiar y qué mantener
4. **Calidad**: Siempre usa `quality="high"` para resultados profesionales
5. **Formato**: PNG para transparencias, JPEG para compresión

## 🔧 Troubleshooting

### Si los detalles no se preservan:
- Verifica que `input_fidelity="high"`
- Asegúrate de que la imagen principal tenga buena resolución
- Usa prompts más específicos

### Si la combinación no funciona:
- Verifica que `combine_images=True`
- Asegúrate de que ambas imágenes tengan formatos compatibles
- Usa imágenes de tamaños similares

### Si hay errores de API:
- Verifica tu API key
- Asegúrate de tener créditos suficientes
- Verifica que tu cuenta tenga acceso a `gpt-image-1` 