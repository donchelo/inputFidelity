"""
OpenAI Image Edit Node para ComfyUI
===================================

Un nodo personalizado que permite editar imágenes usando la API de OpenAI
con soporte completo para input_fidelity="high" para preservar detalles.

Autor: chelo
Versión: 1.0.0
"""

import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .openai_image_edit_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    # Verificar que las importaciones sean exitosas
    if not NODE_CLASS_MAPPINGS:
        raise ImportError("NODE_CLASS_MAPPINGS está vacío")
    
    logger.info("OpenAI Image Edit Node cargado exitosamente")
    
except ImportError as e:
    logger.error(f"Error importando OpenAI Image Edit Node: {e}")
    # Crear mappings vacíos para evitar errores
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

except Exception as e:
    logger.error(f"Error inesperado cargando OpenAI Image Edit Node: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Exportar las variables requeridas por ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 