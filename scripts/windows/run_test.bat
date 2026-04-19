@echo off
set "PROJECT_ROOT=%~dp0..\.."
cd /d "%PROJECT_ROOT%"
python scripts\tests\test_enhanced_counterfactual.py
pause
