@echo off
echo ===================================================
echo [Gemini Quant] Weekend Trainer Auto-Scheduler
echo ===================================================
echo.
echo 스케줄러를 백그라운드에서 실행합니다. 이 창을 닫아도 계속 실행되도록 하려면
echo pythonw 를 사용할 수 있지만, 현재는 실행 상태를 보기 위해 일반 모드로 실행합니다.
echo 중지하려면 이 창을 닫거나 Ctrl+C를 누르세요.
echo.

cd /d "%~dp0"
python trainer_scheduler.py

pause
