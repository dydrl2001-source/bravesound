"""BraveSound stock screener.

CSV 파일을 입력받아 장중매매와 종가베팅 후보를 각각 점수화한다.

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
    # 장중매매용
    "intraday_strength_score": None,      # 없으면 leader_score 사용
    "pullback_quality_score": 5,          # 눌림 질: 0~10
    "vwap_reclaim": 0,                    # VWAP/당일 평균단가 회복 여부: 0/1
    "near_day_high": 0,                   # 당일 고가권 유지 여부: 0/1
    # 종가베팅용
    "closing_strength_score": 5,          # 종가 부근 힘: 0~10
    "next_day_catalyst_score": None,      # 없으면 news_score 사용
    "closing_position_score": 5,          # 종가 위치: 0~10
}

GRADE_ORDER = {"A+": 0, "A": 1, "B": 2, "WATCH": 3, "EXCLUDE": 4}


@dataclass(frozen=True)
class ScreeningConfig:
    min_market_cap_억: int = 1_000
    max_market_cap_억: int = 40_000
    trading_value_rank_cutoff: int = 50
    max_gap_pct_intraday: float = 10.0
    max_gap_pct_closing: float = 12.0
    min_theme_count: int = 3
    min_news_score: int = 6
    min_leader_score: int = 6
    min_intraday_strength: int = 6
    min_pullback_quality: int = 6
    min_closing_strength: int = 6
    min_next_day_catalyst: int = 6


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
    theme_count = int(row["theme_count"])
    news_score = int(row["news_score"])
    leader_score = int(row["leader_score"])
    intraday_strength = int(row["intraday_strength_score"])
    pullback_quality = int(row["pullback_quality_score"])
    close_above_ma5 = to_bool(row["close_above_ma5"])
    close_above_ma20 = to_bool(row["close_above_ma20"])
    vwap_reclaim = to_bool(row["vwap_reclaim"])
    near_day_high = to_bool(row["near_day_high"])
    volume_change_pct = float(row["volume_change_pct"])
    today_top_trading_value = has_top_trading_value(row, config, today_only=True)

    if config.min_market_cap_억 <= market_cap <= config.max_market_cap_억:
        score += 10
        reasons.append("시총 적정")
    else:
        warnings.append("시총 조건 이탈")

    if today_top_trading_value:
        score += 20
        reasons.append("당일 거래대금 상위권")
    else:
        warnings.append("당일 거래대금 부족")

    if theme_count >= config.min_theme_count:
        score += 10
        reasons.append("테마 동반 상승")
    else:
        warnings.append("테마 확산 부족")

    if leader_score >= config.min_leader_score:
        score += 15
        reasons.append("대장주/대장 후보")
    else:
        warnings.append("대장성 약함")

    if news_score >= config.min_news_score:
        score += 10
        reasons.append("뉴스 명분 양호")
    else:
        warnings.append("뉴스 명분 약함")

    if high_gap_pct <= config.max_gap_pct_intraday:
        score += 10
        reasons.append("시초/고점 이격 관리 가능")
    else:
        warnings.append("장중 추격 위험")

    if intraday_strength >= config.min_intraday_strength:
        score += 15
        reasons.append("장중 힘 양호")
    else:
        warnings.append("장중 힘 약함")

    if pullback_quality >= config.min_pullback_quality:
        score += 15
        reasons.append("눌림 질 양호")
    else:
        warnings.append("눌림보다 붕괴 가능성 확인")

    if close_above_ma5 or close_above_ma20:
        score += 8
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

    if volume_change_pct >= 50:
        score += 5
        reasons.append("거래량 증가")
    else:
        warnings.append("거래량 증가 약함")

    hard_exclude = (
        market_cap < config.min_market_cap_억
        or not today_top_trading_value
        or high_gap_pct > 18
        or (not close_above_ma5 and not close_above_ma20)
    )

    return {
        "intraday_score": min(score, 100),
        "intraday_grade": grade_from_score(min(score, 100), hard_exclude),
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
    theme_count = int(row["theme_count"])
    leader_score = int(row["leader_score"])
    closing_strength = int(row["closing_strength_score"])
    next_day_catalyst = int(row["next_day_catalyst_score"])
    closing_position = int(row["closing_position_score"])
    close_above_ma5 = to_bool(row["close_above_ma5"])
    close_above_ma20 = to_bool(row["close_above_ma20"])
    volume_change_pct = float(row["volume_change_pct"])
    top_trading_value = has_top_trading_value(row, config, today_only=False)

    if config.min_market_cap_억 <= market_cap <= config.max_market_cap_억:
        score += 15
        reasons.append("시총 적정")
    else:
        warnings.append("시총 조건 이탈")

    if top_trading_value:
        score += 20
        reasons.append("전일/당일 거래대금 상위권")
    else:
        warnings.append("거래대금 상위 50위 이력 부족")

    if theme_count >= config.min_theme_count:
        score += 15
        reasons.append("테마 동반 상승")
    else:
        warnings.append("테마 확산 부족")

    if leader_score >= config.min_leader_score:
        score += 10
        reasons.append("대장주/대장 후보")
    else:
        warnings.append("대장성 약함")

    if next_day_catalyst >= config.min_next_day_catalyst:
        score += 20
        reasons.append("익일 재료 지속 가능")
    else:
        warnings.append("익일 재료 약함")

    if closing_strength >= config.min_closing_strength:
        score += 15
        reasons.append("종가 부근 힘 양호")
    else:
        warnings.append("종가 부근 힘 약함")

    if closing_position >= 6:
        score += 10
        reasons.append("종가 위치 양호")
    else:
        warnings.append("종가 위치 애매")

    if close_above_ma5:
        score += 7
        reasons.append("5일선 위")
    else:
        warnings.append("5일선 아래")

    if close_above_ma20:
        score += 8
        reasons.append("20일선 위")
    else:
        warnings.append("20일선 아래")

    if high_gap_pct <= config.max_gap_pct_closing:
        score += 5
        reasons.append("고점 대비 이격 관리 가능")
    else:
        warnings.append("고점 대비 이격 과대")

    if volume_change_pct >= 50:
        score += 5
        reasons.append("거래량 증가")
    else:
        warnings.append("거래량 증가 약함")

    hard_exclude = (
        market_cap < config.min_market_cap_억
        or not top_trading_value
        or (not close_above_ma5 and not close_above_ma20)
        or closing_strength < 4
        or next_day_catalyst < 4
    )

    return {
        "closing_score": min(score, 100),
        "closing_grade": grade_from_score(min(score, 100), hard_exclude),
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
