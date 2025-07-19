# OpenAI Image Fidelity Fashion Node para ComfyUI

## 📋 Descripción
Este nodo personalizado permite usar la funcionalidad de High Input Fidelity de OpenAI Image 1 directamente en ComfyUI, específicamente optimizado para casos de uso de moda y fotografía de productos.

## 🚀 Características Principales
- **Alta Fidelidad de Entrada**: Preserva detalles finos como texturas, patrones y logos
- **Presets de Moda**: Configuraciones predefinidas para casos comunes
- **Soporte Multi-imagen**: Combina múltiples imágenes de referencia
- **Máscara Opcional**: Para ediciones precisas de áreas específicas
- **Fondos Transparentes**: Para fotografía de productos

## 📦 Instalación

### Método 1: Manual (Recomendado)

1. **Navega a la carpeta de custom nodes de ComfyUI:**
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. **Copia esta carpeta completa:**
   Copia la carpeta `inputFidelity` a `ComfyUI/custom_nodes/`

3. **Instala las dependencias:**
   
   **Para ComfyUI Portable:**
   ```bash
   # Desde el directorio ComfyUI_windows_portable
   python_embeded\python.exe -m pip install -r custom_nodes\inputFidelity\requirements.txt
   ```
   
   **Para ComfyUI Desktop/Custom Python:**
   ```bash
   cd ComfyUI/custom_nodes/inputFidelity
   pip install -r requirements.txt
   ```

4. **Configura tu API Key de OpenAI:**
   
   **Opción A - Variable de entorno (Recomendado):**
   ```bash
   # Windows
   set OPENAI_API_KEY=tu_api_key_aqui
   
   # Linux/Mac
   export OPENAI_API_KEY=tu_api_key_aqui
   ```
   
   **Opción B - En el nodo directamente:**
   Introduce tu API key en el campo "api_key" del nodo

5. **Reinicia ComfyUI**

### Método 2: Via ComfyUI Manager
1. Abre ComfyUI Manager
2. Busca "OpenAI Image Fidelity Fashion"
3. Instala y reinicia ComfyUI

## 🎯 Casos de Uso Especializados

### 1. **Variación de Outfit**
- Cambia colores, estilos o piezas de ropa
- Preserva pose, rostro y proporciones del modelo
- Mantiene texturas y iluminación realista

### 2. **Adición de Accesorios**
- Añade bolsos, joyas, sombreros, etc.
- Mantiene la pose y detalles originales
- Iluminación coherente con la imagen original

### 3. **Extracción de Producto**
- Extrae prendas a fondos limpios
- Ideal para catálogos de e-commerce
- Preserva todos los detalles del producto

### 4. **Cambio de Color**
- Modifica solo el color de prendas específicas
- Mantiene texturas, patrones y detalles
- Conserva la composición general

### 5. **Transferencia de Estilo**
- Transforma el estilo de la ropa
- Preserva características del modelo
- Mantiene la coherencia visual

### 6. **Cambio de Fondo**
- Solo modifica el fondo
- Preserva modelo, outfit y pose exactamente
- Mantiene la iluminación original

## ⚙️ Parámetros del Nodo

### Parámetros Requeridos:
- **prompt**: Descripción de la edición deseada
- **primary_image**: Imagen principal a editar
- **input_fidelity**: "high" (recomendado) o "low"
- **quality**: "auto", "low", "medium", "high"
- **size**: "auto", "1024x1024", "1024x1536", "1536x1024"
- **output_format**: "png", "jpeg", "webp"
- **background**: "auto", "opaque", "transparent"
- **fashion_preset**: Preset especializado o "custom"

### Parámetros Opcionales:
- **reference_image**: Imagen de referencia adicional
- **mask_image**: Máscara para ediciones precisas
- **api_key**: API key de OpenAI (si no está en variables de entorno)

## 💡 Consejos de Uso

### Para Mejores Resultados:
1. **Usa "high" input fidelity** para preservar detalles importantes
2. **La primera imagen** siempre tiene la máxima preservación de detalles
3. **Coloca rostros en la primera imagen** cuando uses múltiples imágenes
4. **Usa prompts específicos** sobre qué mantener y qué cambiar
5. **Quality "high"** es recomendado para fotografía de moda

### Ejemplos de Prompts Efectivos:
```
"Change the dress to emerald green while preserving all fabric textures and the model's pose"

"Add gold jewelry including earrings and necklace while maintaining the original lighting"

"Extract this exact jacket and place it on a pure white background for catalog use"

"Change only the shirt color to navy blue, keeping all other elements unchanged"
```

## 🔧 Resolución de Problemas

### Error: "OpenAI client not initialized"
- Verifica que tu API key esté configurada correctamente
- Asegúrate de tener créditos disponibles en tu cuenta de OpenAI

### Error: "No data received from OpenAI API"
- Verifica tu conexión a internet
- Comprueba el status de la API de OpenAI
- Revisa que tu prompt no viole las políticas de contenido

### Calidad Baja en Resultados:
- Usa `input_fidelity="high"`
- Incrementa `quality` a "high"
- Asegúrate de que las imágenes de entrada tengan buena resolución

### Dependencias Faltantes:
```bash
# Reinstala las dependencias
pip install --upgrade openai pillow torch numpy
```

## 📊 Costos y Consideraciones

- **Input Fidelity "high"** consume más tokens que "low"
- **La primera imagen** en multi-imagen consume más tokens
- **Quality "high"** genera más tokens de salida
- Revisa la [página de precios de OpenAI](https://openai.com/pricing) para costos actuales

## 🎯 Presets de Moda Disponibles

1. **outfit_variation**: Para cambiar prendas completas
2. **accessory_addition**: Para añadir accesorios
3. **product_extraction**: Para extraer productos a fondos limpios
4. **color_change**: Para cambios de color específicos
5. **style_transfer**: Para transferir estilos entre prendas
6. **background_change**: Para cambiar solo el fondo
7. **custom**: Para prompts personalizados

## 🤝 Contribuciones

Si encuentras bugs o tienes sugerencias de mejora, por favor crea un issue o pull request en el repositorio del proyecto.

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo LICENSE para más detalles.

## 📞 Soporte

Para problemas específicos o preguntas:
1. Revisa esta documentación
2. Verifica que todas las dependencias estén instaladas
3. Confirma que tu API key de OpenAI es válida
4. Prueba con imágenes de menor resolución si hay problemas de memoria