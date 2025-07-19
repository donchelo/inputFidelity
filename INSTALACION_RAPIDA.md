# 🚀 Instalación Rápida - OpenAI Image Fidelity Fashion Node

## ⚡ Instalación en 3 Pasos

### 1️⃣ Copiar a ComfyUI
Copia toda esta carpeta `inputFidelity` a:
```
ComfyUI/custom_nodes/inputFidelity
```

### 2️⃣ Instalar Dependencias

**ComfyUI Portable:**
```bash
# Desde la carpeta ComfyUI_windows_portable
python_embeded\python.exe -m pip install -r custom_nodes\inputFidelity\requirements.txt
```

**ComfyUI Standard:**
```bash
cd ComfyUI/custom_nodes/inputFidelity
pip install -r requirements.txt
```

**O usa el instalador automático:**
- Ejecuta `instalar.bat` (Windows)
- Sigue las instrucciones en pantalla

### 3️⃣ Configurar API Key

**Opción A - Variable de entorno:**
```bash
set OPENAI_API_KEY=tu_api_key_aqui
```

**Opción B - En el nodo:**
Introduce tu API key directamente en el campo "api_key" del nodo

## ✅ Verificar Instalación

1. Reinicia ComfyUI
2. Busca "OpenAI Image Fidelity (Fashion)" en la categoría "OpenAI/Fashion"
3. Si aparece el nodo, ¡la instalación fue exitosa!

## 📚 Archivos Incluidos

- `openai_image_fidelity_fashion.py` - Código principal del nodo
- `__init__.py` - Registro del nodo para ComfyUI
- `requirements.txt` - Dependencias necesarias
- `README.md` - Documentación completa
- `PROMPTS_EJEMPLOS.md` - Ejemplos de prompts para moda
- `example_workflow.json` - Workflow de ejemplo
- `instalar.bat` - Instalador automático para Windows
- `INSTALACION_RAPIDA.md` - Este archivo

## 🆘 ¿Problemas?

1. **Error "module not found"**: Reinstala dependencias
2. **Error "API key"**: Verifica tu API key de OpenAI
3. **Nodo no aparece**: Reinicia ComfyUI completamente

## 💡 Primer Uso

1. Carga una imagen de moda/ropa
2. Conecta a "primary_image"
3. Escribe tu prompt (ej: "Change dress to blue")
4. Selecciona preset "color_change"
5. ¡Genera!

¡Listo para crear increíbles ediciones de moda con alta fidelidad! 🎉