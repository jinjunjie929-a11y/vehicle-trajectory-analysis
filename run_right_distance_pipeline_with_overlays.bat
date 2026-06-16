@echo off
chcp 65001 >nul
cd /d %~dp0
echo ========================================
echo Vehicle trajectory analysis pipeline
echo ========================================
python main.py
pause
