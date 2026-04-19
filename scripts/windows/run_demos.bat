@echo off
set "PROJECT_ROOT=%~dp0..\.."
echo ================================================================================
echo  RUNNING ENHANCED COUNTERFACTUAL + RECOVERY ORCHESTRATOR DEMO
echo ================================================================================
echo.

cd /d "%PROJECT_ROOT%"

echo [1/2] Running Enhanced Counterfactual Demo...
echo.
python scripts\demos\demo_counterfactual_showcase.py
echo.

echo.
echo ================================================================================
echo [2/2] Running Complete Recovery Pipeline Demo...
echo ================================================================================
echo.
python scripts\demos\demo_recovery_pipeline.py

echo.
echo ================================================================================
echo  DEMO COMPLETE!
echo ================================================================================
pause
