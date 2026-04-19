@echo off
set "PROJECT_ROOT=%~dp0..\.."
REM ============================================================================
REM  START SERVICES FOR TRACE & LOG INTEGRATION TESTING
REM ============================================================================
REM
REM This script helps you start all required services to test trace/log integration
REM
REM What it does:
REM 1. Checks if Docker is running
REM 2. Starts Jaeger (for traces)
REM 3. Provides instructions for starting your microservice
REM 4. Shows how to verify data collection
REM ============================================================================

echo.
echo ================================================================================
echo  STARTING OBSERVABILITY INFRASTRUCTURE
echo ================================================================================
echo.

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found! Install Docker Desktop first.
    pause
    exit /b 1
)

echo [1/3] Starting Jaeger for distributed tracing...
echo.

cd /d "%PROJECT_ROOT%"
docker-compose -f docker-compose-jaeger.yml up -d

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start Jaeger
    pause
    exit /b 1
)

echo.
echo [OK] Jaeger started successfully!
echo      UI: http://localhost:16686
echo      Agent: localhost:6831 (UDP)
echo.

timeout /t 3 >nul

echo ================================================================================
echo  NEXT STEPS
echo ================================================================================
echo.
echo [2/3] Start your microservice:
echo.
echo   Option A - Docker:
echo     cd services\notification-service
echo     docker build -t notification-service .
echo     docker run -p 8000:8000 --name notification-service notification-service
echo.
echo   Option B - Direct Python:
echo     cd services\notification-service
echo     pip install -r requirements.txt
echo     python notification_service.py
echo.
echo [3/3] Verify integration:
echo     python scripts\tests\test_integration_verification.py
echo.
echo ================================================================================
echo  VERIFY DATA COLLECTION
echo ================================================================================
echo.
echo 1. Check Jaeger UI: http://localhost:16686
echo    - Should see "notification-service" in service dropdown
echo    - Should see traces when you generate traffic
echo.
echo 2. Generate traffic:
echo    python scripts\tools\generate_load.py
echo.
echo 3. Capture logs (in separate terminal):
echo    python scripts\tools\capture_service_logs.py notification-service 2
echo.
echo 4. Run verification:
echo    python scripts\tests\test_integration_verification.py
echo.
echo ================================================================================
echo  DEMO COMMANDS
echo ================================================================================
echo.
echo After data is flowing, run:
echo.
echo   python scripts\demos\demo_integrated_rca.py    # Full RCA with traces + logs
echo   python scripts\demos\demo_full_pipeline.py     # Complete detection pipeline
echo.
echo ================================================================================
echo.

pause
