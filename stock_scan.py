# -*- coding: utf-8 -*-
"""
stock_scan.py

用途：
1. 从全市场股票池中筛选股票
2. 剔除 ST 股
3. 剔除市值 80 亿以下股票（根据本地文件）
4. 日线价格条件：
   - close > MA20
   - close 离 MA20 不远
   - MA20 向上
   - MA20 > MA60
5. 日线 MACD 条件：
   - 使用 Tushare 已计算好的因子
   - macd_dif_bfq > macd_dea_bfq > 0
6. 周线条件：
   - MA5 > MA10 > MA20
7. 月线条件：
   - MA5 > MA10 > MA20
8. 输出入选结果和失败结果到 output/ 目录

GitHub Actions 中需要设置环境变量：
- TUSHARE_TOKEN
"""

import os
import time
import pandas as pd
import numpy as np
import tushare as ts


# =========================
# 1. 基础配置
# =========================
TUSHARE_TOKEN = os.environ["TUSHARE_TOKEN"]

CODE_FILE = "data/a_share_codes_for_akshare.csv"
ST_FILE = "data/st_stocks.csv"
BELOW_8B_FILE = "data/a_share_below_8b.csv"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 调试时可改成 20 / 50，正式跑设为 None
TEST_LIMIT = None

# 请求间隔
SLEEP_SECONDS = 0.12

# 收盘价离 MA20 最多高出 10%
MAX_DAILY_DIST_TO_MA20 = 0.10

# 取数起始日期
DAILY_START_DATE = "20240101"
WEEKLY_START_DATE = "20230101"
MONTHLY_START_DATE = "20220101"
FACTOR_START_DATE = "20240101"

END_DATE = pd.Timestamp.today().strftime("%Y%m%d")


# =========================
# 2. 初始化 Tushare
# =========================
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


# =========================
# 3. 工具函数
# =========================
def normalize_symbol_to_6(x):
    if pd.isna(x):
        return None

    s = str(x).strip()
    if s.lower() == "nan" or s == "":
        return None

    digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) >= 6:
        return digits[-6:]

    return None


def to_ts_code(symbol6, exchange_cd=None):
    if pd.isna(symbol6):
        return None

    symbol6 = str(symbol6).strip()

    if symbol6.lower() == "nan" or symbol6 == "":
        return None

    symbol6 = "".join(ch for ch in symbol6 if ch.isdigit())

    if len(symbol6) != 6:
        return None

    ex = ""
    if exchange_cd is not None and not pd.isna(exchange_cd):
        ex = str(exchange_cd).strip().upper()

    if ex in {"XSHE", "SZ", "SZSE"}:
        return f"{symbol6}.SZ"
    if ex in {"XSHG", "SH", "SSE"}:
        return f"{symbol6}.SH"
    if ex in {"BSE", "BJ"}:
        return f"{symbol6}.BJ"

    if symbol6.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{symbol6}.SH"
    if symbol6.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{symbol6}.SZ"
    if symbol6.startswith(
        (
            "430", "431", "432", "433", "834", "835", "836", "837",
            "838", "839", "870", "871", "872", "873", "874", "875",
            "876", "877", "878", "879", "920"
        )
    ):
        return f"{symbol6}.BJ"

    return None


def add_ma(df, close_col, windows):
    df = df.copy()
    for w in windows:
        df[f"ma{w}"] = df[close_col].rolling(w).mean()
    return df


def read_universe_from_csv(code_file):
    df = pd.read_csv(code_file, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = None
    for c in ["ticker", "symbol", "code", "股票代码"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None and "secID" in df.columns:
        df["ticker"] = df["secID"].astype(str).str.extract(r"(\d{6})", expand=False)
        symbol_col = "ticker"

    if symbol_col is None:
        raise ValueError("股票池文件中无法识别股票代码列")

    exchange_col = "exchangeCD" if "exchangeCD" in df.columns else None

    df["symbol6"] = df[symbol_col].apply(normalize_symbol_to_6)

    # 先去掉无效代码，再转 ts_code
    df = df.dropna(subset=["symbol6"]).copy()

    df["ts_code"] = df.apply(
        lambda row: to_ts_code(
            row["symbol6"],
            row[exchange_col] if exchange_col else None
        ),
        axis=1
    )

    name_col = "secShortName" if "secShortName" in df.columns else None
    df["name"] = df[name_col] if name_col else ""

    df = df.dropna(subset=["ts_code"]).copy()
    df = df.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
    return df


def read_st_set(st_file):
    df = pd.read_csv(st_file, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = None
    for c in ["ticker", "symbol", "code", "股票代码"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None and "secID" in df.columns:
        df["ticker"] = df["secID"].astype(str).str.extract(r"(\d{6})", expand=False)
        symbol_col = "ticker"

    if symbol_col is None:
        raise ValueError("ST文件中无法识别股票代码列")

    st_set = set(
        df[symbol_col]
        .dropna()
        .astype(str)
        .apply(normalize_symbol_to_6)
        .dropna()
        .tolist()
    )
    return st_set


def read_below_8b_set(below_8b_file):
    df = pd.read_csv(below_8b_file, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = None
    for c in ["ticker", "symbol", "code", "股票代码"]:
        if c in df.columns:
            symbol_col = c
            break

    if symbol_col is None and "secID" in df.columns:
        df["ticker"] = df["secID"].astype(str).str.extract(r"(\d{6})", expand=False)
        symbol_col = "ticker"

    if symbol_col is None:
        raise ValueError("80亿以下股票文件中无法识别股票代码列")

    below_8b_set = set(
        df[symbol_col]
        .dropna()
        .astype(str)
        .apply(normalize_symbol_to_6)
        .dropna()
        .tolist()
    )
    return below_8b_set


# =========================
# 4. 数据获取
# =========================
def fetch_daily(ts_code):
    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=DAILY_START_DATE,
            end_date=END_DATE
        )
        if df is None or df.empty:
            return None, "daily_empty"

        if "trade_date" not in df.columns or "close" not in df.columns:
            return None, f"daily_bad_columns: {list(df.columns)}"

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)

        if df.empty:
            return None, "daily_after_clean_empty"

        return df, None
    except Exception as e:
        return None, f"daily_exception: {repr(e)}"


def fetch_weekly(ts_code):
    try:
        df = pro.weekly(
            ts_code=ts_code,
            start_date=WEEKLY_START_DATE,
            end_date=END_DATE
        )
        if df is None or df.empty:
            return None, "weekly_empty"

        if "trade_date" not in df.columns or "close" not in df.columns:
            return None, f"weekly_bad_columns: {list(df.columns)}"

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)

        if df.empty:
            return None, "weekly_after_clean_empty"

        df = df.rename(columns={"close": "close_used"})
        return df, None
    except Exception as e:
        return None, f"weekly_exception: {repr(e)}"


def fetch_monthly(ts_code):
    try:
        df = pro.monthly(
            ts_code=ts_code,
            start_date=MONTHLY_START_DATE,
            end_date=END_DATE
        )
        if df is None or df.empty:
            return None, "monthly_empty"

        if "trade_date" not in df.columns or "close" not in df.columns:
            return None, f"monthly_bad_columns: {list(df.columns)}"

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)

        if df.empty:
            return None, "monthly_after_clean_empty"

        df = df.rename(columns={"close": "close_used"})
        return df, None
    except Exception as e:
        return None, f"monthly_exception: {repr(e)}"


def fetch_daily_factor(ts_code):
    try:
        df = pro.stk_factor_pro(
            ts_code=ts_code,
            start_date=FACTOR_START_DATE,
            end_date=END_DATE
        )

        if df is None or df.empty:
            return None, "factor_empty"

        required_cols = ["trade_date", "macd_dif_bfq", "macd_dea_bfq"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"factor_missing_cols: {missing}, all_cols={list(df.columns)}"

        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["macd_dif_bfq"] = pd.to_numeric(df["macd_dif_bfq"], errors="coerce")
        df["macd_dea_bfq"] = pd.to_numeric(df["macd_dea_bfq"], errors="coerce")

        df = df.dropna(subset=["trade_date", "macd_dif_bfq", "macd_dea_bfq"])
        df = df.sort_values("trade_date").reset_index(drop=True)

        if df.empty:
            return None, "factor_after_clean_empty"

        return df, None
    except Exception as e:
        return None, f"factor_exception: {repr(e)}"


# =========================
# 5. 条件判断
# =========================
def check_daily_price_conditions(df_daily):
    if df_daily is None or len(df_daily) < 80:
        return False, {"reason": "daily_not_enough"}

    df = add_ma(df_daily, "close", [20, 60])

    if len(df) < 2:
        return False, {"reason": "daily_not_enough_2"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    needed_cols = ["close", "ma20", "ma60"]
    if any(pd.isna(last[c]) for c in needed_cols) or pd.isna(prev["ma20"]):
        return False, {"reason": "daily_ma_nan"}

    close_ = float(last["close"])
    ma20 = float(last["ma20"])
    ma60 = float(last["ma60"])
    prev_ma20 = float(prev["ma20"])

    cond1 = close_ > ma20
    dist = close_ / ma20 - 1 if ma20 > 0 else np.nan
    cond2 = pd.notna(dist) and dist <= MAX_DAILY_DIST_TO_MA20
    cond3 = ma20 > prev_ma20
    cond4 = ma20 > ma60

    ok = cond1 and cond2 and cond3 and cond4

    info = {
        "daily_trade_date": last["trade_date"].strftime("%Y-%m-%d"),
        "close": round(close_, 4),
        "ma20_d": round(ma20, 4),
        "ma60_d": round(ma60, 4),
        "dist_to_ma20": round(dist, 4),
    }

    if not ok:
        failed = []
        if not cond1:
            failed.append("close_not_above_ma20")
        if not cond2:
            failed.append("close_too_far_from_ma20")
        if not cond3:
            failed.append("ma20_not_up")
        if not cond4:
            failed.append("ma20_not_above_ma60")
        info["reason"] = "|".join(failed)

    return ok, info


def check_daily_macd_factor(df_factor):
    if df_factor is None or df_factor.empty:
        return False, {"reason": "factor_not_enough"}

    last = df_factor.iloc[-1]
    dif = float(last["macd_dif_bfq"])
    dea = float(last["macd_dea_bfq"])

    cond = (dif > dea) and (dea > 0)

    info = {
        "factor_trade_date": last["trade_date"].strftime("%Y-%m-%d"),
        "dif_d": round(dif, 4),
        "dea_d": round(dea, 4),
    }

    if not cond:
        info["reason"] = "macd_factor_fail"

    return cond, info


def check_ma_order(df, label):
    if df is None or len(df) < 20:
        return False, {"reason": f"{label}_not_enough"}

    dfx = add_ma(df, "close_used", [5, 10, 20])
    last = dfx.iloc[-1]

    needed = ["ma5", "ma10", "ma20"]
    if any(pd.isna(last[c]) for c in needed):
        return False, {"reason": f"{label}_indicator_nan"}

    ma5 = float(last["ma5"])
    ma10 = float(last["ma10"])
    ma20 = float(last["ma20"])

    cond = ma5 > ma10 > ma20

    info = {
        f"{label}_trade_date": last["trade_date"].strftime("%Y-%m-%d"),
        f"ma5_{label}": round(ma5, 4),
        f"ma10_{label}": round(ma10, 4),
        f"ma20_{label}": round(ma20, 4),
    }

    if not cond:
        info["reason"] = f"{label}_ma_order_fail"

    return cond, info


# =========================
# 6. 主流程
# =========================
def main():
    universe_df = read_universe_from_csv(CODE_FILE)
    st_set = read_st_set(ST_FILE)
    below_8b_set = read_below_8b_set(BELOW_8B_FILE)

    universe_df = universe_df[~universe_df["symbol6"].isin(st_set)].copy()
    universe_df = universe_df[~universe_df["symbol6"].isin(below_8b_set)].copy()
    universe_df = universe_df.reset_index(drop=True)

    if TEST_LIMIT is not None:
        universe_df = universe_df.head(TEST_LIMIT).copy()

    print(f"ST股票数量: {len(st_set)}")
    print(f"80亿以下股票数量: {len(below_8b_set)}")
    print(f"待扫描股票数（剔除ST和80亿以下后）: {len(universe_df)}")

    results = []
    fails = []

    for i, row in universe_df.iterrows():
        ts_code = row["ts_code"]
        symbol6 = row["symbol6"]
        name = row["name"]

        print(f"[{i+1}/{len(universe_df)}] {ts_code}")

        df_daily, err_daily = fetch_daily(ts_code)
        if df_daily is None:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                "reason": err_daily
            })
            time.sleep(SLEEP_SECONDS)
            continue

        ok_daily_price, info_daily = check_daily_price_conditions(df_daily)
        if not ok_daily_price:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                **info_daily
            })
            time.sleep(SLEEP_SECONDS)
            continue

        df_factor, err_factor = fetch_daily_factor(ts_code)
        if df_factor is None:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                **info_daily,
                "reason": err_factor
            })
            time.sleep(SLEEP_SECONDS)
            continue

        ok_factor, info_factor = check_daily_macd_factor(df_factor)
        if not ok_factor:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                **info_daily,
                **info_factor
            })
            time.sleep(SLEEP_SECONDS)
            continue

        df_week, err_week = fetch_weekly(ts_code)
        if df_week is None:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                **info_daily,
                **info_factor,
                "reason": err_week
            })
            time.sleep(SLEEP_SECONDS)
            continue

        ok_week, info_week = check_ma_order(df_week, "w")
        if not ok_week:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                **info_daily,
                **info_factor,
                **info_week
            })
            time.sleep(SLEEP_SECONDS)
            continue

        df_month, err_month = fetch_monthly(ts_code)
        if df_month is None:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                **info_daily,
                **info_factor,
                **info_week,
                "reason": err_month
            })
            time.sleep(SLEEP_SECONDS)
            continue

        ok_month, info_month = check_ma_order(df_month, "m")
        if not ok_month:
            fails.append({
                "ts_code": ts_code,
                "symbol6": symbol6,
                "name": name,
                **info_daily,
                **info_factor,
                **info_week,
                **info_month
            })
            time.sleep(SLEEP_SECONDS)
            continue

        results.append({
            "ts_code": ts_code,
            "symbol6": symbol6,
            "name": name,
            **info_daily,
            **info_factor,
            **info_week,
            **info_month
        })

        time.sleep(SLEEP_SECONDS)

    result_df = pd.DataFrame(results)
    fail_df = pd.DataFrame(fails)

    result_path = os.path.join(OUTPUT_DIR, "tushare_scheme2_selected_exclude_8b.csv")
    fail_path = os.path.join(OUTPUT_DIR, "tushare_scheme2_failed_exclude_8b.csv")

    if not result_df.empty:
        sort_cols = [c for c in ["dist_to_ma20", "dif_d"] if c in result_df.columns]
        if sort_cols:
            result_df = result_df.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)

    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    fail_df.to_csv(fail_path, index=False, encoding="utf-8-sig")

    print("\n扫描完成")
    print(f"入选数量: {len(result_df)}")
    print(f"失败/未通过数量: {len(fail_df)}")
    print(f"入选文件: {result_path}")
    print(f"失败文件: {fail_path}")

    if not result_df.empty:
        print("\n前20条入选结果：")
        print(result_df.head(20).to_string(index=False))
    else:
        print("\n本次没有股票满足条件。")

    if not fail_df.empty and "reason" in fail_df.columns:
        print("\n失败原因统计前20项：")
        print(fail_df["reason"].value_counts().head(20))


if __name__ == "__main__":
    main()
