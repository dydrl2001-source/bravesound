"""BraveSound stock screener.

CSV 파일을 입력받아 장중매매와 종가베팅 후보를 각각 점수화한다.
업로드한 급등주/패턴/데이기법 노트의 핵심 원칙을 반영해
거래대금, 체결강도, 눌림, 지지/저항, 리스크 플래그를 함께 본다.

Usage:
    python src/main.py data/sample_watchlist.csv --mode all --top 10
    python src/main.py data/sample_watchlist.csv --mode intraday --top 10
    python src/main.py data/sample_watchlist.csv --mode closing --top 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


Mode = Literal["all", "intraday", "closing"]

BASE_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "market_cap_억",
    "trading_value_rank_today",
    "trading_value_rank_yesterday",
    "high_gap_pct",
    "theme_count",
    "news_score",
    "leader_score",
    "close_above_ma5",
    "close_above_ma20",
    "volume_change_pct",
]

OPTIONAL_COLUMNS_WITH_DEFAULTS = {
    # 공통/주도주 확인
    "price_change_pct": 0.0,
    "top15_after_930": 0,
    "sector_leader_rank": 3,
    "first_big_bull_candle": 0,
    "breakout_high_6m": 0,
    "support_resistance_score": 5,
    "second_low_confirmed": 0,
    "loss_cut_ready": 1,
    "position_size_pct": 30.0,
    # 장중매매용
    "intraday_strength_score": None,
    "pullback_quality_score": 5,
    "vwap_reclaim": 0,
    "near_day_high": 0,
    "execution_window_score": 5,
    # 종가베팅용
    "closing_strength_score": 5,
    "next_day_catalyst_score": None,
    "closing_position_score": 5,
    # 위험 신호
    "long_upper_shadow_risk": 0,
    "volume_dry_up_risk": 0,
    "washout_risk": 0,
    "fomo_risk": 0,
}

GRADE_ORDER = {"A+": 0, "A": 1, "B": 2, "WATCH": 3, "EXCLUDE": 4}
RISK_COLUMNS = ["long_upper_shadow_risk", "volume_dry_up_risk", "washout_risk", "fomo_risk"]


@dataclass(frozen=True)
class ScreeningConfig:
    min_market_cap_억: int = 1_000
    max_market_cap_억: int = 40_000
    trading_value_rank_cutoff: int = 50
    trading_value_rank_strong_cutoff: int = 15
    max_gap_pct_intraday: float = 10.0
    max_gap_pct_closing: float = 12.0
    min_price_change_pct_for_daytrade: float = 5.0
    min_theme_count: int = 3
    min_news_score: int = 6
    min_leader_score: int = 6
    min_intraday_strength: int = 6
    min_pullback_quality: int = 6
    min_closing_strength: int = 6
    min_next_day_catalyst: int = 6
    max_position_size_pct: float = 50.0


def load_watchlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)
    missing = [column for column in BASE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    df = add_optional_columns(df)
    return df


def add_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, default in OPTIONAL_COLUMNS_WITH_DEFAULTS.items():
        if column not in df.columns:
            if column == "intraday_strength_score":
                df[column] = df["leader_score"]
            elif column == "next_day_catalyst_score":
                df[column] = df["news_score"]
            else:
                df[column] = default
    return df


def to_bool(value: Any) -> bool:
    return bool(int(value or 0))


def has_top_trading_value(row: pd.Series, config: ScreeningConfig, today_only: bool = False) -> bool:
    today = int(row["trading_value_rank_today"] or 0)
    yesterday = int(row["trading_value_rank_yesterday"] or 0)

    today_ok = 1 <= today <= config.trading_value_rank_cutoff
    yesterday_ok = 1 <= yesterday <= config.trading_value_rank_cutoff

    if today_only:
        return today_ok
    return today_ok or yesterday_ok


def has_strong_trading_rank(row: pd.Series, config: ScreeningConfig) -> bool:
    today = int(row["trading_value_rank_today"] or 0)
    return 1 <= today <= config.trading_value_rank_strong_cutoff or to_bool(row["top15_after_930"])


def risk_count(row: pd.Series) -> int:
    return sum(1 for column in RISK_COLUMNS if to_bool(row[column]))


def apply_risk_penalty(row: pd.Series, reasons: list[str], warnings: list[str]) -> int:
    penalty = 0

    if to_bool(row["long_upper_shadow_risk"]):
        penalty += 8
        warnings.append("윗꼬리/고점 분산 위험")
    if to_bool(row["volume_dry_up_risk"]):
        penalty += 10
        warnings.append("거래량·거래대금 이탈 위험")
    if to_bool(row["washout_risk"]):
        penalty += 12
        warnings.append("설거지 위험")
    if to_bool(row["fomo_risk"]):
        penalty += 8
        warnings.append("FOMO 추격 위험")

    if penalty == 0:
        reasons.append("주요 위험 플래그 없음")

    return penalty


def grade_from_score(score: int, hard_exclude: bool) -> str:
    if hard_exclude:
        return "EXCLUDE"
    if score >= 90:
        return "A+"
    if score >= 78:
        return "A"
    if score >= 65:
        return "B"
    return "WATCH"


def score_intraday(row: pd.Series, config: ScreeningConfig) -> dict[str, Any]:
    """장중매매 점수화.

    장중매매는 '지금 돈이 들어오는가'와 '추격이 아닌 눌림/재돌파인가'를 더 강하게 본다.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0

    market_cap = float(row["market_cap_억"])
    high_gap_pct = float(row["high_gap_pct"])
    price_change_pct = float(row["price_change_pct"])
    theme_count = int(row["theme_count"])
    news_score = int(row["news_score"])
    leader_score = int(row["leader_score"])
    sector_leader_rank = int(row["sector_leader_rank"] or 99)
    intraday_strength = int(row["intraday_strength_score"])
    pullback_quality = int(row["pullback_quality_score"])
    support_resistance_score = int(row["support_resistance_score"])
    execution_window_score = int(row["execution_window_score"])
    close_above_ma5 = to_bool(row["close_above_ma5"])
    close_above_ma20 = to_bool(row["close_above_ma20"])
    vwap_reclaim = to_bool(row["vwap_reclaim"])
    near_day_high = to_bool(row["near_day_high"])
    second_low_confirmed = to_bool(row["second_low_confirmed"])
    loss_cut_ready = to_bool(row["loss_cut_ready"])
    first_big_bull_candle = to_bool(row["first_big_bull_candle"])
    breakout_high_6m = to_bool(row["breakout_high_6m"])
    position_size_pct = float(row["position_size_pct"])
    volume_change_pct = float(row["volume_change_pct"])
    today_top_trading_value = has_top_trading_value(row, config, today_only=True)
    strong_trading_rank = has_strong_trading_rank(row, config)

    if config.min_market_cap_억 <= market_cap <= config.max_market_cap_억:
        score += 8
        reasons.append("시총 적정")
    else:
        warnings.append("시총 조건 이탈")

    if today_top_trading_value:
        score += 16
        reasons.append("당일 거래대금 상위권")
    else:
        warnings.append("당일 거래대금 부족")

    if strong_trading_rank:
        score += 8
        reasons.append("9:30 이후 거래대금 강세권")

    if price_change_pct >= config.min_price_change_pct_for_daytrade:
        score += 7
        reasons.append("5% 이상 상승 후보")
    else:
        warnings.append("상승률 모멘텀 약함")

    if theme_count >= config.min_theme_count:
        score += 9
        reasons.append("테마 3종목 이상 동반")
    else:
        warnings.append("테마 확산 부족")

    if sector_leader_rank == 1:
        score += 10
        reasons.append("섹터 대장주")
    elif sector_leader_rank <= 3:
        score += 6
        reasons.append("섹터 상위권")
    else:
        warnings.append("섹터 후발주 가능성")

    if leader_score >= config.min_leader_score:
        score += 9
        reasons.append("대장성 양호")
    else:
        warnings.append("대장성 약함")

    if news_score >= config.min_news_score:
        score += 8
        reasons.append("뉴스 반응 양호")
    else:
        warnings.append("뉴스 명분 약함")

    if high_gap_pct <= config.max_gap_pct_intraday:
        score += 7
        reasons.append("시초/고점 이격 관리 가능")
    else:
        warnings.append("장중 추격 위험")

    if intraday_strength >= config.min_intraday_strength:
        score += 10
        reasons.append("체결강도/장중 힘 양호")
    else:
        warnings.append("장중 힘 약함")

    if pullback_quality >= config.min_pullback_quality:
        score += 10
        reasons.append("눌림 질 양호")
    else:
        warnings.append("눌림보다 붕괴 가능성 확인")

    if support_resistance_score >= 6:
        score += 7
        reasons.append("지지/저항 구조 양호")
    else:
        warnings.append("지지·저항 근거 약함")

    if second_low_confirmed:
        score += 5
        reasons.append("두 번째 저점 확인")

    if first_big_bull_candle:
        score += 4
        reasons.append("첫 장대양봉 후보")

    if breakout_high_6m:
        score += 5
        reasons.append("전고/신고가 돌파")

    if close_above_ma5 or close_above_ma20:
        score += 6
        reasons.append("주요 이평선 위 또는 회복")
    else:
        warnings.append("주요 이평선 아래")

    if vwap_reclaim:
        score += 4
        reasons.append("평균단가/VWAP 회복")
    else:
        warnings.append("평균단가 회복 미확인")

    if near_day_high:
        score += 3
        reasons.append("당일 고가권 유지")

    if execution_window_score >= 6:
        score += 4
        reasons.append("시간대 적합")
    else:
        warnings.append("시간대 신뢰도 낮음")

    if volume_change_pct >= 50:
        score += 5
        reasons.append("거래량 증가")
    else:
        warnings.append("거래량 증가 약함")

    if loss_cut_ready:
        score += 4
        reasons.append("손절 기준 준비")
    else:
        warnings.append("손절 기준 없음")

    if position_size_pct <= config.max_position_size_pct:
        score += 3
        reasons.append("비중 과열 아님")
    else:
        warnings.append("비중 과다")

    score -= apply_risk_penalty(row, reasons, warnings)
    score = max(0, min(score, 100))

    hard_exclude = (
        market_cap < config.min_market_cap_억
        or not today_top_trading_value
        or high_gap_pct > 18
        or (not close_above_ma5 and not close_above_ma20)
        or not loss_cut_ready
        or risk_count(row) >= 2
        or position_size_pct > 70
    )

    return {
        "intraday_score": score,
        "intraday_grade": grade_from_score(score, hard_exclude),
        "intraday_reasons": "; ".join(reasons),
        "intraday_warnings": "; ".join(warnings),
    }


def score_closing(row: pd.Series, config: ScreeningConfig) -> dict[str, Any]:
    """종가베팅 점수화.

    종가베팅은 '오늘 강했고, 종가까지 버텼고, 내일도 해석될 명분이 있는가'를 더 강하게 본다.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0

    market_cap = float(row["market_cap_억"])
    high_gap_pct = float(row["high_gap_pct"])
    price_change_pct = float(row["price_change_pct"])
    theme_count = int(row["theme_count"])
    leader_score = int(row["leader_score"])
    sector_leader_rank = int(row["sector_leader_rank"] or 99)
    closing_strength = int(row["closing_strength_score"])
    next_day_catalyst = int(row["next_day_catalyst_score"])
    closing_position = int(row["closing_position_score"])
    support_resistance_score = int(row["support_resistance_score"])
    close_above_ma5 = to_bool(row["close_above_ma5"])
    close_above_ma20 = to_bool(row["close_above_ma20"])
    loss_cut_ready = to_bool(row["loss_cut_ready"])
    first_big_bull_candle = to_bool(row["first_big_bull_candle"])
    breakout_high_6m = to_bool(row["breakout_high_6m"])
    position_size_pct = float(row["position_size_pct"])
    volume_change_pct = float(row["volume_change_pct"])
    top_trading_value = has_top_trading_value(row, config, today_only=False)
    strong_trading_rank = has_strong_trading_rank(row, config)

    if config.min_market_cap_억 <= market_cap <= config.max_market_cap_억:
        score += 10
        reasons.append("시총 적정")
    else:
        warnings.append("시총 조건 이탈")

    if top_trading_value:
        score += 16
        reasons.append("전일/당일 거래대금 상위권")
    else:
        warnings.append("거래대금 상위 50위 이력 부족")

    if strong_trading_rank:
        score += 6
        reasons.append("거래대금 강세권")

    if price_change_pct >= config.min_price_change_pct_for_daytrade:
        score += 5
        reasons.append("당일 상승률 조건 충족")
    else:
        warnings.append("당일 상승률 약함")

    if theme_count >= config.min_theme_count:
        score += 10
        reasons.append("테마 3종목 이상 동반")
    else:
        warnings.append("테마 확산 부족")

    if sector_leader_rank == 1:
        score += 8
        reasons.append("섹터 대장주")
    elif sector_leader_rank <= 3:
        score += 5
        reasons.append("섹터 상위권")
    else:
        warnings.append("섹터 후발주 가능성")

    if leader_score >= config.min_leader_score:
        score += 7
        reasons.append("대장성 양호")
    else:
        warnings.append("대장성 약함")

    if next_day_catalyst >= config.min_next_day_catalyst:
        score += 16
        reasons.append("익일 재료 지속 가능")
    else:
        warnings.append("익일 재료 약함")

    if closing_strength >= config.min_closing_strength:
        score += 14
        reasons.append("종가 부근 힘 양호")
    else:
        warnings.append("종가 부근 힘 약함")

    if closing_position >= 6:
        score += 10
        reasons.append("종가 위치 양호")
    else:
        warnings.append("종가 위치 애매")

    if support_resistance_score >= 6:
        score += 6
        reasons.append("지지/저항 구조 양호")
    else:
        warnings.append("지지·저항 근거 약함")

    if first_big_bull_candle:
        score += 4
        reasons.append("첫 장대양봉 후보")

    if breakout_high_6m:
        score += 5
        reasons.append("전고/신고가 돌파")

    if close_above_ma5:
        score += 5
        reasons.append("5일선 위")
    else:
        warnings.append("5일선 아래")

    if close_above_ma20:
        score += 5
        reasons.append("20일선 위")
    else:
        warnings.append("20일선 아래")

    if high_gap_pct <= config.max_gap_pct_closing:
        score += 4
        reasons.append("고점 대비 이격 관리 가능")
    else:
        warnings.append("고점 대비 이격 과대")

    if volume_change_pct >= 50:
        score += 4
        reasons.append("거래량 증가")
    else:
        warnings.append("거래량 증가 약함")

    if loss_cut_ready:
        score += 4
        reasons.append("익일 손절 기준 준비")
    else:
        warnings.append("익일 손절 기준 없음")

    if position_size_pct <= config.max_position_size_pct:
        score += 3
        reasons.append("비중 과열 아님")
    else:
        warnings.append("오버나잇 비중 과다")

    score -= apply_risk_penalty(row, reasons, warnings)
    score = max(0, min(score, 100))

    hard_exclude = (
        market_cap < config.min_market_cap_억
        or not top_trading_value
        or (not close_above_ma5 and not close_above_ma20)
        or closing_strength < 4
        or next_day_catalyst < 4
        or not loss_cut_ready
        or risk_count(row) >= 2
        or position_size_pct > 50
    )

    return {
        "closing_score": score,
        "closing_grade": grade_from_score(score, hard_exclude),
        "closing_reasons": "; ".join(reasons),
        "closing_warnings": "; ".join(warnings),
    }


def add_sort_key(df: pd.DataFrame, grade_column: str) -> pd.DataFrame:
    output = df.copy()
    output["_grade_order"] = output[grade_column].map(GRADE_ORDER).fillna(99)
    return output


def screen(df: pd.DataFrame, config: ScreeningConfig, mode: Mode) -> pd.DataFrame:
    output = df.copy()

    if mode in ("all", "intraday"):
        intraday_results = output.apply(lambda row: score_intraday(row, config), axis=1, result_type="expand")
        output = pd.concat([output, intraday_results], axis=1)

    if mode in ("all", "closing"):
        closing_results = output.apply(lambda row: score_closing(row, config), axis=1, result_type="expand")
        output = pd.concat([output, closing_results], axis=1)

    if mode == "intraday":
        output = add_sort_key(output, "intraday_grade")
        return output.sort_values(["_grade_order", "intraday_score"], ascending=[True, False]).drop(columns=["_grade_order"])

    if mode == "closing":
        output = add_sort_key(output, "closing_grade")
        return output.sort_values(["_grade_order", "closing_score"], ascending=[True, False]).drop(columns=["_grade_order"])

    output["total_score"] = output["intraday_score"] + output["closing_score"]
    return output.sort_values("total_score", ascending=False)


def columns_for_mode(mode: Mode) -> list[str]:
    base = ["ticker", "name", "sector"]
    if mode == "intraday":
        return base + ["intraday_score", "intraday_grade", "intraday_reasons", "intraday_warnings"]
    if mode == "closing":
        return base + ["closing_score", "closing_grade", "closing_reasons", "closing_warnings"]
    return base + [
        "intraday_score",
        "intraday_grade",
        "closing_score",
        "closing_grade",
        "intraday_warnings",
        "closing_warnings",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="BraveSound 장중매매·종가베팅 후보 스크리너")
    parser.add_argument("csv_path", type=Path, help="분석할 watchlist CSV 경로")
    parser.add_argument("--mode", choices=["all", "intraday", "closing"], default="all", help="분석 모드")
    parser.add_argument("--top", type=int, default=20, help="출력할 상위 종목 수")
    args = parser.parse_args()

    config = ScreeningConfig()
    df = load_watchlist(args.csv_path)
    result = screen(df, config, args.mode).head(args.top)

    print(result[columns_for_mode(args.mode)].to_string(index=False))


if __name__ == "__main__":
    main()
