"""BraveSound stock screener.

CSV 파일을 입력받아 사용자의 단타·단기스윙 기준에 맞게
A+, A, B, WATCH, EXCLUDE 등급을 산출한다.

Usage:
    python src/main.py data/sample_watchlist.csv --top 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
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


@dataclass(frozen=True)
class ScreeningConfig:
    min_market_cap_억: int = 1_000
    max_market_cap_억: int = 40_000
    trading_value_rank_cutoff: int = 50
    max_high_gap_pct_for_a: float = 12.0
    min_theme_count: int = 3
    min_news_score: int = 6
    min_leader_score: int = 6


def load_watchlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    return df


def has_top_trading_value(row: pd.Series, config: ScreeningConfig) -> bool:
    today = int(row["trading_value_rank_today"] or 0)
    yesterday = int(row["trading_value_rank_yesterday"] or 0)

    return (
        1 <= today <= config.trading_value_rank_cutoff
        or 1 <= yesterday <= config.trading_value_rank_cutoff
    )


def score_row(row: pd.Series, config: ScreeningConfig) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0

    market_cap = float(row["market_cap_억"])
    high_gap_pct = float(row["high_gap_pct"])
    theme_count = int(row["theme_count"])
    news_score = int(row["news_score"])
    leader_score = int(row["leader_score"])
    close_above_ma5 = bool(int(row["close_above_ma5"]))
    close_above_ma20 = bool(int(row["close_above_ma20"]))
    volume_change_pct = float(row["volume_change_pct"])
    top_trading_value = has_top_trading_value(row, config)

    if not (config.min_market_cap_억 <= market_cap <= config.max_market_cap_억):
        warnings.append("시총 조건 이탈")
    else:
        score += 15
        reasons.append("시총 적정")

    if top_trading_value:
        score += 20
        reasons.append("거래대금 상위권")
    else:
        warnings.append("거래대금 상위 50위 이력 부족")

    if theme_count >= config.min_theme_count:
        score += 15
        reasons.append("테마 동반 상승")
    else:
        warnings.append("테마 확산 부족")

    if news_score >= config.min_news_score:
        score += 15
        reasons.append("뉴스 명분 양호")
    else:
        warnings.append("뉴스 지속성 약함")

    if leader_score >= config.min_leader_score:
        score += 15
        reasons.append("대장주/대장 후보")
    else:
        warnings.append("리더십 약함")

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

    if high_gap_pct <= config.max_high_gap_pct_for_a:
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
    )

    if hard_exclude:
        grade = "EXCLUDE"
    elif score >= 90:
        grade = "A+"
    elif score >= 78:
        grade = "A"
    elif score >= 65:
        grade = "B"
    else:
        grade = "WATCH"

    return {
        "score": score,
        "grade": grade,
        "reasons": "; ".join(reasons),
        "warnings": "; ".join(warnings),
    }


def screen(df: pd.DataFrame, config: ScreeningConfig) -> pd.DataFrame:
    results = df.apply(lambda row: score_row(row, config), axis=1, result_type="expand")
    output = pd.concat([df, results], axis=1)
    return output.sort_values(["grade", "score"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="BraveSound 주도주 후보 스크리너")
    parser.add_argument("csv_path", type=Path, help="분석할 watchlist CSV 경로")
    parser.add_argument("--top", type=int, default=20, help="출력할 상위 종목 수")
    args = parser.parse_args()

    config = ScreeningConfig()
    df = load_watchlist(args.csv_path)
    result = screen(df, config).head(args.top)

    columns = [
        "ticker",
        "name",
        "sector",
        "score",
        "grade",
        "reasons",
        "warnings",
    ]
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
