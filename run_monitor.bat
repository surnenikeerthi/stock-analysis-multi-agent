@echo off
cd /d "D:\project\python_projects\capstone"
set PYTHONUTF8=1
if not exist logs mkdir logs
".venv\Scripts\python.exe" portfolio_monitor.py >> logs\monitor.log 2>&1
