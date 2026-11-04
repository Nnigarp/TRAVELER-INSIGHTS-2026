@echo off
cd /d "%~dp0"
if not exist .venv (py -3.11 -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install -e ".[ui,test]"
if errorlevel 1 (echo Error instalando dependencias. & pause & exit /b 1)
python app.py
pause
