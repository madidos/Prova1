@echo off
cd /d "%~dp0"
echo Installazione dipendenze...
pip install -r requirements.txt
echo.
echo Avvio app ETF...
streamlit run app.py
pause
