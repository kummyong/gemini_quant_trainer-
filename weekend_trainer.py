import time
import logging
import itertools
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from typing import Dict, Tuple, List

# 기존에 만들어둔 텔레그램 브로드캐스터 임포트
from telegram_broadcaster import send_telegram_message

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("WeekendTrainer")


# ==========================================
# 모듈 1: 인메모리 데이터 허브 (진짜 데이터 로드)
# ==========================================
def load_real_market_data(
    tickers: List[str], start_date: str = "2023-01-01"
) -> Dict[str, pd.DataFrame]:
    """
    FinanceDataReader를 사용해 타겟 종목들의 과거 데이터를 로드하고,
    '반복 계산할 필요가 없는' 기본 지표(이동평균, 기본 RSI)를 미리 계산해 둡니다. (속도 최적화의 핵심)
    """
    logger.info(
        f"💾 [Data Loader] {len(tickers)}개 종목의 과거 데이터 로드 및 전처리 시작..."
    )

    market_data = {}

    for ticker in tickers:
        try:
            # 1. 일봉 데이터 다운로드
            df = fdr.DataReader(ticker, start_date)
            if df.empty:
                continue

            # 2. RSI 계산 (14일 기준 - Wilders Smoothing 방식)
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # 3. 볼린저 밴드의 '기본 재료' 계산 (표준편차 배수는 나중에 곱함)
            df["BB_Mid"] = df["Close"].rolling(window=20).mean()
            df["BB_Std"] = df["Close"].rolling(window=20).std()

            # 결측치 제거
            df.dropna(inplace=True)

            market_data[ticker] = df

        except Exception as e:
            logger.error(f"❌ [{ticker}] 데이터 로드 실패: {e}")

    logger.info(f"✅ [Data Loader] 총 {len(market_data)}개 종목 인메모리 적재 완료.")
    return market_data


# ==========================================
# 모듈 2: 초고속 벡터 백테스트 엔진 (실제 매매 로직)
# ==========================================
def run_real_backtest(
    market_data: Dict[str, pd.DataFrame], rsi_buy: float, bb_std: float
) -> Tuple[float, float, int]:
    """
    미리 로드된 데이터(market_data)에 특정 파라미터를 씌워 모의 투자를 수행합니다.
    """
    total_trades = 0
    portfolio_returns = []  # 각 종목별 누적 수익률 보관
    mdd_list = []  # 각 종목별 MDD 보관

    for ticker, df in market_data.items():
        # 1. 테스트할 볼린저 밴드 하단선 동적 계산 (벡터 연산으로 0.01초 컷)
        df["Test_BB_Lower"] = df["BB_Mid"] - (bb_std * df["BB_Std"])

        # 2. 매수 시그널 포착: RSI가 기준치 이하 AND 종가가 BB 하단선 이하
        buy_signals = (df["RSI"] <= rsi_buy) & (df["Close"] <= df["Test_BB_Lower"])
        trade_count = buy_signals.sum()
        total_trades += trade_count

        # 3. 간단한 수익률 시뮬레이션 (매수 다음 날 시가 매도한다고 가정 - 예시 로직)
        # ※ 실제로는 15% 비중 보유 등 더 정밀하게 짜야 하지만, 속도를 위해 '타점의 다음날 반등률'로 룰의 퀄리티를 평가합니다.
        if trade_count > 0:
            # 매수 시그널이 뜬 '다음 날'의 수익률 (종가 대비 다음날 종가)
            df["Next_Day_Return"] = df["Close"].pct_change().shift(-1)

            # 타점이 발생한 날들의 수익률만 모아서 누적 계산
            strategy_returns = df.loc[buy_signals, "Next_Day_Return"].fillna(0)

            if len(strategy_returns) > 0:
                cumulative_return = (1 + strategy_returns).prod() - 1
                portfolio_returns.append(cumulative_return)

                # MDD 계산 (타점 이후 발생한 최대 손실폭)
                roll_max = (1 + strategy_returns).cumprod().cummax()
                daily_drawdown = (1 + strategy_returns).cumprod() / roll_max - 1.0
                mdd_list.append(abs(daily_drawdown.min()))

    # 전체 종목 평균 성과 산출
    avg_return = np.mean(portfolio_returns) * 100 if portfolio_returns else 0.0
    avg_mdd = np.mean(mdd_list) * 100 if mdd_list else 1.0  # 0 나누기 방지

    return avg_return, avg_mdd, total_trades


# ==========================================
# 모듈 3: 최적화 루프 (Fitness Evaluation)
# ==========================================
def main():
    logger.info("🚀 [Trainer] 주말 퀀트 파라미터 최적화 엔진 기동")

    # 1. 데이터 로드 (PC의 넉넉한 RAM 활용)
    tickers = ["005930", "000660", "035420"] # 삼성전자, SK하이닉스, NAVER
    market_data = load_real_market_data(tickers, start_date="2023-01-01")

    # 2. 탐색 공간(Search Space) 정의 - 총 30개의 조합
    rsi_candidates = range(25, 36, 2)  # [25, 27, 29, 31, 33, 35]
    bb_candidates = [1.8, 2.0, 2.2, 2.4, 2.5]  # 5개

    best_score = -999.0
    best_params = {}

    logger.info("🧬 [Optimizer] 파라미터 조합 탐색 시작...")

    # 3. 브루트포스(Brute-force) 최적화 루프
    for rsi, bb in itertools.product(rsi_candidates, bb_candidates):

        # 백테스트 실행
        tot_return, mdd, trades = run_real_backtest(market_data, rsi, bb)

        # 💡 적합도 함수 (Fitness Function): 거래횟수가 너무 적으면 페널티
        if trades < 10:
            score = -1.0  # 과최적화(Curve fitting) 방지
        else:
            score = tot_return / max(0.1, mdd)

        if score > best_score:
            best_score = score
            best_params = {
                "RSI_BUY_THRES": rsi,
                "BB_STD": bb,
                "RETURN": tot_return,
                "MDD": mdd,
            }

    logger.info("🏆 [훈련 완료] 최적의 파라미터 유전자 발굴:")
    logger.info(
        f"   👉 RSI 매수선: {best_params['RSI_BUY_THRES']} | BB 표준편차: {best_params['BB_STD']}"
    )
    logger.info(
        f"   👉 예상 수익률: {best_params['RETURN']:.2f}% | MDD: {best_params['MDD']:.2f}% | 점수: {best_score:.2f}"
    )

    # 4. 모바일(엣지)로 텔레그램 전송
    update_msg = f"[SYSTEM_UPDATE] RSI_BUY_THRES={best_params['RSI_BUY_THRES']}"
    # bb_std도 전송하고 싶다면: f"[SYSTEM_UPDATE] RSI_BUY_THRES={best_params['RSI_BUY_THRES']}, BB_STD={best_params['BB_STD']}"

    if send_telegram_message(update_msg):
        logger.info("✨ [성공] 엣지 노드로 시스템 업데이트 지령을 송신했습니다.")
    else:
        logger.error("❌ [실패] 텔레그램 송신 에러")


if __name__ == "__main__":
    main()
