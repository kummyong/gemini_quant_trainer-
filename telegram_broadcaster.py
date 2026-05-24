import os
import asyncio
from telethon import TelegramClient

# 1단계에서 발급받은 정보 입력
API_ID = 20282225
API_HASH = "bfc1e2296e2e4c8daf9cb3b467ab889a"

# 명령을 받을 모바일 봇의 유저네임 (봇파더가 만들어준 봇 아이디, 예: @gemini_quant_bot)
BOT_USERNAME = "@kummyong_stock_bot"

# 세션 파일 이름 (이 이름으로 PC에 로그인 정보가 저장됩니다)
client = TelegramClient("quant_trainer_session", API_ID, API_HASH)


async def _send_to_bot_async(message: str):
    """비동기 통신으로 봇에게 메시지를 쏘는 핵심 로직"""
    # 클라이언트 시작 (최초 1회 인증 요구)
    await client.start()

    # 사람인 척 봇에게 메시지 전송
    await client.send_message(BOT_USERNAME, message)
    print(f"[Telethon] 봇({BOT_USERNAME})에게 시스템 첩보 송신 완료: {message}")

    # 전송 후 안전하게 연결 종료
    await client.disconnect()


def send_telegram_message(message: str) -> bool:
    """
    weekend_trainer.py에서 기존 코드 수정 없이 그대로 호출할 수 있도록
    비동기 함수를 동기적으로 래핑(Wrapping)한 함수입니다.
    """
    try:
        asyncio.run(_send_to_bot_async(message))
        return True
    except Exception as e:
        print(f"[Telethon] 송신 에러: {e}")
        return False


if __name__ == "__main__":
    # 최초 인증 및 통신 테스트용 단독 실행
    print("🚀 Telethon 클라이언트 테스트를 시작합니다...")
    send_telegram_message("[SYSTEM_UPDATE] RSI_BUY_THRES=27.0, BB_STD=2.1")
