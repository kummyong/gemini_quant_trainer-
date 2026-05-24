import time
import logging
import schedule
from weekend_trainer import main as run_weekend_trainer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trainer_scheduler.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainerScheduler")

def job():
    logger.info("⏰ 주말 퀀트 최적화 스케줄러: 학습 프로세스를 시작합니다.")
    try:
        run_weekend_trainer()
        logger.info("✅ 주말 최적화 및 텔레그램 송신이 정상적으로 완료되었습니다.")
    except Exception as e:
        logger.error(f"❌ 스케줄러 실행 중 에러 발생: {e}")

def start_scheduler():
    logger.info("🚀 Trainer 스케줄러가 시작되었습니다. (매주 토요일 10:00 실행 대기 중)")
    
    # 매주 토요일 오전 10시에 학습 실행
    schedule.every().saturday.at("10:00").do(job)
    
    # 테스트용: schedule.every(1).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_scheduler()
