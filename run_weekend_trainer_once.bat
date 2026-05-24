@echo off
cd /d D:\workspace_py\gemini_quant_trainer
echo 🚀 [주말 퀀트 최적화] KOSPI 국면 판독 및 파라미터 최적화 시작...
python weekend_trainer.py
echo ✅ 최적화 및 텔레그램 송신이 완료되었습니다. 창은 10초 뒤에 닫힙니다.
timeout /t 10
