@echo off
REM Instala lo que falte (requirements.txt) y abre la app en el navegador
cd /d "%~dp0"
echo Revisando librerias...
python -m pip install -q --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo No se pudieron instalar las librerias. Revise que Python este instalado y en el PATH.
    pause
    exit /b 1
)
python -m streamlit run app.py
pause
