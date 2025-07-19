@echo off
echo ====================================
echo OpenAI Image Fidelity Fashion Node
echo Instalador para ComfyUI
echo ====================================
echo.

:: Verificar si estamos en la carpeta correcta
if not exist "openai_image_fidelity_fashion.py" (
    echo ERROR: No se encontro el archivo principal del nodo.
    echo Asegurate de ejecutar este script desde la carpeta del nodo.
    pause
    exit /b 1
)

:: Solicitar ruta de ComfyUI
set /p COMFYUI_PATH="Introduce la ruta completa de tu instalacion de ComfyUI (ej: C:\ComfyUI): "

:: Verificar que la ruta existe
if not exist "%COMFYUI_PATH%" (
    echo ERROR: La ruta especificada no existe.
    pause
    exit /b 1
)

:: Verificar que es una instalacion de ComfyUI valida
if not exist "%COMFYUI_PATH%\custom_nodes" (
    echo ERROR: No se encontro la carpeta custom_nodes en la ruta especificada.
    echo Asegurate de que la ruta apunte a la instalacion de ComfyUI.
    pause
    exit /b 1
)

:: Crear carpeta del nodo
set NODE_PATH=%COMFYUI_PATH%\custom_nodes\inputFidelity
echo Creando carpeta del nodo en: %NODE_PATH%
mkdir "%NODE_PATH%" 2>nul

:: Copiar archivos
echo Copiando archivos del nodo...
copy "openai_image_fidelity_fashion.py" "%NODE_PATH%\"
copy "__init__.py" "%NODE_PATH%\"
copy "requirements.txt" "%NODE_PATH%\"
copy "README.md" "%NODE_PATH%\"
copy "example_workflow.json" "%NODE_PATH%\"

:: Detectar tipo de instalacion de ComfyUI
if exist "%COMFYUI_PATH%\python_embeded" (
    echo Detectada instalacion Portable de ComfyUI
    set PYTHON_PATH=%COMFYUI_PATH%\python_embeded\python.exe
) else (
    echo Detectada instalacion Standard de ComfyUI
    set PYTHON_PATH=python
)

:: Instalar dependencias
echo.
echo Instalando dependencias...
echo Ejecutando: %PYTHON_PATH% -m pip install -r "%NODE_PATH%\requirements.txt"
"%PYTHON_PATH%" -m pip install -r "%NODE_PATH%\requirements.txt"

if errorlevel 1 (
    echo.
    echo ERROR: Hubo un problema instalando las dependencias.
    echo Intenta instalarlas manualmente con:
    echo %PYTHON_PATH% -m pip install openai pillow torch numpy
    pause
    exit /b 1
)

:: Solicitar API Key
echo.
echo ====================================
echo Configuracion de API Key
echo ====================================
echo.
echo Para usar este nodo necesitas una API Key de OpenAI.
echo Puedes obtenerla en: https://platform.openai.com/api-keys
echo.
set /p SETUP_API_KEY="¿Quieres configurar tu API Key ahora? (s/n): "

if /i "%SETUP_API_KEY%"=="s" (
    set /p API_KEY="Introduce tu API Key de OpenAI: "
    
    echo.
    echo Configurando variable de entorno...
    setx OPENAI_API_KEY "!API_KEY!"
    
    echo.
    echo IMPORTANTE: Reinicia ComfyUI para que la variable de entorno tenga efecto.
    echo O puedes introducir la API Key directamente en el nodo.
) else (
    echo.
    echo Recuerda configurar tu API Key de OpenAI antes de usar el nodo.
    echo Puedes hacerlo:
    echo 1. Como variable de entorno: set OPENAI_API_KEY=tu_api_key
    echo 2. Directamente en el campo "api_key" del nodo
)

echo.
echo ====================================
echo Instalacion Completada
echo ====================================
echo.
echo El nodo ha sido instalado exitosamente en:
echo %NODE_PATH%
echo.
echo Pasos siguientes:
echo 1. Reinicia ComfyUI
echo 2. Busca el nodo "OpenAI Image Fidelity (Fashion)" en la categoria "OpenAI/Fashion"
echo 3. Configura tu API Key si no lo hiciste ya
echo 4. ¡Disfruta editando imagenes de moda con alta fidelidad!
echo.
echo Documentacion completa en: %NODE_PATH%\README.md
echo Workflow de ejemplo en: %NODE_PATH%\example_workflow.json
echo.
pause