@echo off
echo ========================================
echo Instalador de OpenAI Image Edit Node
echo ========================================
echo.

REM Verificar si estamos en el directorio correcto
if not exist "requirements.txt" (
    echo Error: No se encontró requirements.txt
    echo Asegúrate de ejecutar este script desde el directorio del nodo
    pause
    exit /b 1
)

echo Instalando dependencias...
echo.

REM Intentar diferentes métodos de instalación
echo Método 1: Usando python del sistema...
python -m pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo Dependencias instaladas exitosamente usando python del sistema
    goto :success
)

echo.
echo Método 2: Usando python3...
python3 -m pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo Dependencias instaladas exitosamente usando python3
    goto :success
)

echo.
echo Método 3: Usando pip directamente...
pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo Dependencias instaladas exitosamente usando pip
    goto :success
)

echo.
echo Error: No se pudieron instalar las dependencias automáticamente
echo.
echo Instrucciones manuales:
echo 1. Abre una terminal en este directorio
echo 2. Ejecuta: pip install -r requirements.txt
echo 3. Si usas ComfyUI Portable, usa: python_embeded\python.exe -m pip install -r requirements.txt
echo.
pause
exit /b 1

:success
echo.
echo ========================================
echo Instalación completada exitosamente!
echo ========================================
echo.
echo Para usar el nodo:
echo 1. Copia esta carpeta a custom_nodes/ de ComfyUI
echo 2. Reinicia ComfyUI
echo 3. Busca "OpenAI Image Edit" en el menú de nodos
echo.
echo Para configurar tu API key:
echo - Variable de entorno: OPENAI_API_KEY
echo - Archivo: openai_api_key.txt (se crea automáticamente)
echo - Input del nodo: Proporcionar directamente
echo.
pause 