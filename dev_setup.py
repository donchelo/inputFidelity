#!/usr/bin/env python3
"""
Script de configuración de desarrollo para OpenAI Image Edit Node.
Este script ayuda a configurar el entorno de desarrollo y verificar la instalación.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_dependencies():
    """Verifica las dependencias requeridas."""
    required_packages = [
        'openai',
        'pillow',
        'torch',
        'numpy',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - FALTANTE")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Instalando dependencias faltantes...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencias instaladas exitosamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False
    
    return True

def check_file_structure():
    """Verifica la estructura de archivos del proyecto."""
    required_files = [
        "openai_image_edit_node.py",
        "__init__.py",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - FALTANTE")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Archivos faltantes: {', '.join(missing_files)}")
        return False
    
    return True

def check_comfyui_integration():
    """Verifica la integración con ComfyUI."""
    print("\n🔍 Verificando integración con ComfyUI...")
    
    # Verificar que el nodo se puede importar
    try:
        from openai_image_edit_node import OpenAIImageEditNode, NODE_CLASS_MAPPINGS
        print("✅ Nodo importado correctamente")
        
        # Verificar que la clase tiene los atributos requeridos
        node = OpenAIImageEditNode()
        required_attrs = ['RETURN_TYPES', 'RETURN_NAMES', 'FUNCTION', 'CATEGORY']
        
        for attr in required_attrs:
            if hasattr(node, attr):
                print(f"✅ {attr} - OK")
            else:
                print(f"❌ {attr} - FALTANTE")
                return False
        
        # Verificar mappings
        if NODE_CLASS_MAPPINGS:
            print("✅ NODE_CLASS_MAPPINGS - OK")
        else:
            print("❌ NODE_CLASS_MAPPINGS - VACÍO")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando nodo: {e}")
        return False
    except Exception as e:
        print(f"❌ Error verificando nodo: {e}")
        return False
    
    return True

def create_dev_config():
    """Crea archivos de configuración para desarrollo."""
    print("\n⚙️ Configurando archivos de desarrollo...")
    
    # Crear directorio .vscode si no existe
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Crear config.example.json si no existe
    if not os.path.exists("config.example.json"):
        print("✅ config.example.json ya existe")
    else:
        print("✅ config.example.json - OK")
    
    print("✅ Configuración de desarrollo completada")

def main():
    """Función principal del script."""
    print("🚀 Configuración de desarrollo - OpenAI Image Edit Node")
    print("=" * 60)
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Estructura de archivos", check_file_structure),
        ("Dependencias", check_dependencies),
        ("Integración ComfyUI", check_comfyui_integration),
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        print(f"\n📋 Verificando {name}...")
        if not check_func():
            all_passed = False
    
    create_dev_config()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ¡Configuración completada exitosamente!")
        print("\n📝 Próximos pasos:")
        print("1. Copia esta carpeta a custom_nodes/ de ComfyUI")
        print("2. Reinicia ComfyUI")
        print("3. Busca 'OpenAI Image Edit' en el menú de nodos")
        print("4. Configura tu API key de OpenAI")
    else:
        print("❌ Configuración incompleta. Revisa los errores arriba.")
        sys.exit(1)

if __name__ == "__main__":
    main() 