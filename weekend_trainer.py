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
# 모듈 1: 인메모리 데이터 허브 (국면 판독 포함)
# ==========================================
def load_real_market_data(
    tickers: List[str], start_date: str = "2023-01-01"
) -> Dict[str, pd.DataFrame]:
    """
    FinanceDataReader를 사용해 타겟 종목들과 KOSPI(KS11) 지수를 로드합니다.
    코스피 지수의 50일 이동평균선을 기준으로 BULL/BEAR 국면을 판독하여 각 종목 데이터에 병합합니다.
    """
    logger.info(f"💾 [Data Loader] KOSPI 지수 및 {len(tickers)}개 종목 데이터 로드 시작...")

    market_data = {}

    try:
        # KOSPI 데이터 로드 및 50일 이동평균선(50-MA) 계산
        kospi = fdr.DataReader("KS11", start_date)
        kospi["50_MA"] = kospi["Close"].rolling(window=50).mean()
        
        # 국면(Regime) 라벨링: 종가가 50일선 위에 있으면 BULL, 아래면 BEAR (벡터 연산)
        kospi["Regime"] = np.where(kospi["Close"] > kospi["50_MA"], "BULL", "BEAR")
        kospi_regime = kospi[["Regime"]]
        logger.info("📈 [Regime] 코스피(KS11) 기준 국면 판독 라벨링 완료.")
    except Exception as e:
        logger.error(f"❌ KOSPI 데이터 로드 및 국면 판독 실패: {e}")
        return {}

    for ticker in tickers:
        try:
            df = fdr.DataReader(ticker, start_date)
            if df.empty:
                continue

            # RSI 계산 (14일 기준 - Wilders Smoothing 방식)
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # 볼린저 밴드의 '기본 재료' 계산 (표준편차 배수는 나중에 곱함)
            df["BB_Mid"] = df["Close"].rolling(window=20).mean()
            df["BB_Std"] = df["Close"].rolling(window=20).std()

            # 코스피 국면(Regime) 데이터를 종목 데이터에 병합 (인덱스 날짜 기준 매칭)
            df = df.join(kospi_regime, how="left")
            
            # 결측치(초기 50일 등) 제거
            df.dropna(inplace=True)

            market_data[ticker] = df

        except Exception as e:
            logger.error(f"❌ [{ticker}] 데이터 로드 실패: {e}")

    logger.info(f"✅ [Data Loader] 총 {len(market_data)}개 종목 인메모리 적재 완료.")
    return market_data


# ==========================================
# 모듈 2: 초고속 벡터 백테스트 엔진 (국면 분리형)
# ==========================================
def run_real_backtest(
    market_data: Dict[str, pd.DataFrame], rsi_buy: float, bb_std: float, target_regime: str
) -> Tuple[float, float, int]:
    """
    미리 로드된 데이터(market_data)에 특정 파라미터와 특정 국면(target_regime)을 씌워 모의 투자를 수행합니다.
    """
    total_trades = 0
    portfolio_returns = []  
    mdd_list = []  

    for ticker, df in market_data.items():
        # 1. 테스트할 볼린저 밴드 하단선 동적 계산 (벡터 연산으로 0.01초 컷)
        df["Test_BB_Lower"] = df["BB_Mid"] - (bb_std * df["BB_Std"])

        # 2. 매수 시그널 포착: RSI 이하 AND 볼밴 하단 이하 AND 타겟 국면(BULL/BEAR) 일치
        buy_signals = (df["RSI"] <= rsi_buy) & (df["Close"] <= df["Test_BB_Lower"]) & (df["Regime"] == target_regime)
        trade_count = buy_signals.sum()
        total_trades += trade_count

        # 3. 간단한 수익률 시뮬레이션 (매수 다음 날 종가 매도한다고 가정)
        if trade_count > 0:
            df["Next_Day_Return"] = df["Close"].pct_change().shift(-1)
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
def optimize_for_regime(market_data, rsi_candidates, bb_candidates, regime: str):
    logger.info(f"🧬 [{regime} 장] 최적 파라미터 유전자 탐색 시작...")
    best_score = -999.0
    best_params = {"RSI": 30, "BB": 2.0, "RETURN": 0.0, "MDD": 0.0}
    
    # 브루트포스(Brute-force) 최적화 루프
    for rsi, bb in itertools.product(rsi_candidates, bb_candidates):
        tot_return, mdd, trades = run_real_backtest(market_data, rsi, bb, target_regime=regime)
        
        # 💡 적합도 함수: 거래횟수가 너무 적으면 페널티 (국면 분리 감안하여 5회로 완화)
        if trades < 5:
            score = -1.0  
        else:
            score = tot_return / max(0.1, mdd)

        if score > best_score:
            best_score = score
            best_params = {
                "RSI": rsi,
                "BB": bb,
                "RETURN": tot_return,
                "MDD": mdd,
                "TRADES": trades
            }
            
    logger.info(f"🏆 [{regime} 최적화 완료] RSI: {best_params['RSI']} | BB: {best_params['BB']} (수익: {best_params['RETURN']:.2f}%, MDD: {best_params['MDD']:.2f}%)")
    return best_params

def main():
    logger.info("🚀 [Trainer] 국면 전환(Regime Switching) 퀀트 파라미터 최적화 엔진 기동")

    # 1. 데이터 로드
    tickers = ["005930", "000660", "035420"] # 삼성전자, SK하이닉스, NAVER
    market_data = load_real_market_data(tickers, start_date="2023-01-01")

    # 2. 탐색 공간(Search Space) 정의
    rsi_candidates = range(20, 41, 2)
    bb_candidates = [1.8, 2.0, 2.2, 2.4, 2.5]

    # 3. 국면별 최적화 별도 수행
    bull_best = optimize_for_regime(market_data, rsi_candidates, bb_candidates, "BULL")
    bear_best = optimize_for_regime(market_data, rsi_candidates, bb_candidates, "BEAR")

    # 4. 모바일(엣지)로 텔레그램 전송 (Payload 포맷 고도화)
    update_msg = (
        f"[SYSTEM_UPDATE] BULL_RSI={bull_best['RSI']}, BULL_BB={bull_best['BB']}, "
        f"BEAR_RSI={bear_best['RSI']}, BEAR_BB={bear_best['BB']}"
    )

    if send_telegram_message(update_msg):
        logger.info("✨ [성공] 엣지 노드로 국면별 파라미터 업데이트 지령 송신 완료.")
    else:
        logger.error("❌ [실패] 텔레그램 송신 에러")


if __name__ == "__main__":
    main()
