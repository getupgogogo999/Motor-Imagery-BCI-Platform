@echo off
cd /d "G:\BCICIV_2a_all_patients.csv"
echo Project: %CD%
echo Stopping old Streamlit on port 8501...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PY%" set PY=python
echo Using Python: %PY%
"%PY%" -m streamlit run app.py --server.port 8501
pause
