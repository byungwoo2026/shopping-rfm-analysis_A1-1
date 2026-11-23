"""분석 도구 모음: DataAnalyzer 클래스.

CSV 표 하나를 불러와서 정리하고 분석하는 메서드를 한 클래스에 모아 둡니다.
같은 클래스로 거래표(Online Retail II)와 상품표(Fashion)를 각각 다룹니다.
(표마다 DataAnalyzer를 하나씩 만들므로 인스턴스가 2개가 됩니다.)
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Online Retail II 원본 컬럼명 → 분석에 쓸 이름(공백 없는 소문자_밑줄 형식)
# 'Customer ID'처럼 공백이 있으면 코드에서 쓰기 불편하고, 과제 요구 컬럼명(customer_id,
# order_date, amount)과도 맞지 않아서 읽자마자 한 번에 바꿉니다. (디스크의 CSV 파일은 그대로)
RETAIL_RENAME = {
    "Invoice": "invoice",           # 주문번호 ('C'로 시작하면 취소 주문)
    "StockCode": "stock_code",      # 상품코드
    "Description": "description",   # 상품명
    "Quantity": "quantity",         # 수량
    "InvoiceDate": "order_date",    # 주문 시각
    "Price": "price",               # 단가
    "Customer ID": "customer_id",   # 고객 번호
    "Country": "country",           # 국가
}


def _mode(values):
    """최빈값(가장 많이 나온 값) 하나를 돌려준다. 값이 하나도 없으면 NaN.

    동점이면 정렬했을 때 앞에 오는 값을 고른다(실행할 때마다 같은 결과가 나오게).
    """
    counts = values.mode()                          # 최빈값 목록 (빈칸은 자동으로 제외됨)
    return counts.iloc[0] if len(counts) else np.nan


# 결측 대치 전략 이름 → 그룹의 대표값을 구하는 방법
_STRATEGIES = {
    "group_mean": "mean",       # 평균 (숫자 컬럼용)
    "group_median": "median",   # 중앙값 (숫자 컬럼용, 극단값에 덜 흔들림)
    "group_mode": _mode,        # 최빈값 (글자·범주 컬럼용)
}


class DataAnalyzer:
    """표 하나를 불러와 정제·분석하는 도구.

    사용 순서:  analyzer = DataAnalyzer(경로)  →  analyzer.load_data(...)  →  각종 메서드
    분석 중인 표는 항상 analyzer.df 에 들어 있고, 메서드가 이 표를 직접 갱신합니다.
    """

    def __init__(self, path):
        self.path = Path(path)      # 읽을 CSV 파일 위치 (경로는 여기서만 받음)
        self.df = None              # 불러온 표. load_data()를 부르기 전에는 비어 있음
        self.cleaning_log = None    # clean_transactions()가 남기는 "단계별 제거 행 수" 표
        self.missing_log = []       # handle_missing_values()를 부를 때마다 한 줄(딕셔너리)씩 쌓임

    # ───────────── 불러오기 ─────────────
    def load_data(self, date_cols=None, rename=None, **read_csv_kwargs):
        """CSV를 읽어 self.df 에 저장한다.

        date_cols : 날짜로 변환할 컬럼 이름 목록 (CSV에 적힌 원래 이름으로 적는다)
        rename    : {원래 이름: 새 이름} 딕셔너리. 주면 읽은 직후 컬럼명을 바꾼다
        read_csv_kwargs : pd.read_csv에 그대로 넘길 옵션 (dtype=..., on_bad_lines=... 등)
        """
        if date_cols:
            read_csv_kwargs["parse_dates"] = date_cols   # 날짜 글자를 진짜 날짜 형식으로 변환
        df = pd.read_csv(self.path, **read_csv_kwargs)
        if rename:
            df = df.rename(columns=rename)               # 날짜 변환이 끝난 뒤에 이름을 바꾼다
        self.df = df

    # ───────────── 거래 데이터 정제 ─────────────
    def clean_transactions(self):
        """거래표에서 분석에 쓸 수 없는 행을 제거하고 amount 컬럼을 만든다.

        제거 순서: ① 완전히 같은 행(중복) → ② 취소 주문 → ③ 수량 0 이하 → ④ 단가 0 이하
        각 단계에서 몇 행을 지웠는지 표로 돌려주고 self.cleaning_log 에도 저장한다.
        (고객 번호가 비어 있는 행은 여기서 지우지 않는다. RFM 계산 때 따로 제외한다.)
        """
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        need = {"invoice", "quantity", "price"}
        missing = need - set(self.df.columns)
        if missing:
            raise KeyError(f"거래표에 필요한 컬럼이 없습니다: {sorted(missing)}")

        df = self.df
        steps = [("정제 전 (원본)", 0, len(df))]          # (단계 이름, 제거한 행, 남은 행)

        # ① 완전히 같은 행 제거: 엑셀 두 시트가 겹친 기간 + 시트 안에서 반복된 줄
        before = len(df)
        df = df.drop_duplicates()
        steps.append(("완전히 같은 행(중복) 제거", before - len(df), len(df)))

        # ② 취소 주문 제거: 주문번호가 'C'로 시작하면 취소(수량이 음수로 기록됨)
        before = len(df)
        is_cancel = df["invoice"].str.startswith("C", na=False)
        df = df[~is_cancel]                              # ~ 는 "반대로": 취소가 아닌 행만 남김
        steps.append(("취소 주문(주문번호 C…) 제거", before - len(df), len(df)))

        # ③ 수량이 0 이하인 행 제거: 취소 표시 없이 음수로 들어간 행(분실·파손 조정 등)
        before = len(df)
        df = df[df["quantity"] > 0]
        steps.append(("수량 0 이하 제거", before - len(df), len(df)))

        # ④ 단가가 0 이하인 행 제거: 무료 지급·회계 조정 행이라 판매가 아님
        before = len(df)
        df = df[df["price"] > 0]
        steps.append(("단가 0 이하 제거", before - len(df), len(df)))

        # ⑤ 한 줄의 금액 = 수량 × 단가 (RFM의 M, 매출 분석에 사용)
        df = df.assign(amount=df["quantity"] * df["price"])

        self.df = df.reset_index(drop=True)              # 지워진 자리 때문에 비어 버린 줄 번호를 0부터 다시 매김
        self.cleaning_log = pd.DataFrame(steps, columns=["단계", "제거한 행", "남은 행"])
        return self.cleaning_log

    # ───────────── 결측치 ─────────────
    def missing_report(self):
        """컬럼별로 빈칸이 몇 개, 몇 %인지 보여주는 현황 표 (빈칸이 많은 컬럼이 위로)."""
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        counts = self.df.isna().sum()                              # 컬럼별 빈칸 수
        report = pd.DataFrame({
            "컬럼": counts.index,
            "결측 수": counts.values,
            "결측 비율(%)": (counts.values / len(self.df) * 100).round(2),
        })
        return report.sort_values("결측 수", ascending=False, kind="stable").reset_index(drop=True)

    def group_consistency(self, target_col, group_cols):
        """어떤 기준으로 묶어야 최빈값으로 채운 값이 믿을 만한지 비교한다 (글자 컬럼용).

        이미 값이 있는 행마다 "내 그룹의 최빈값"을 구해 실제 값과 같은 비율(일치율)을 센다.
        일치율이 높을수록 그 기준으로 묶어 채운 값이 실제와 비슷하다는 뜻이다.
        '(묶지 않음)' 행은 그룹 없이 전체 최빈값 하나만 쓴 경우로, 기준선이 된다.
        주의: 그룹을 아주 잘게 나누면(1행짜리 그룹은 항상 100%) 일치율이 부풀려지므로 그룹 수를 함께 본다.
        """
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        known = self.df.dropna(subset=[target_col])                # 값이 있는 행만
        rows = [{"묶는 기준": "(묶지 않음)", "그룹 수": 1,
                 "일치율(%)": round((known[target_col] == _mode(known[target_col])).mean() * 100, 1)}]
        for group_col in group_cols:
            sub = known.dropna(subset=[group_col])
            table = sub.groupby(group_col)[target_col].agg(_mode)  # 그룹별 최빈값 표
            same = sub[target_col] == sub[group_col].map(table)    # 내 값 == 내 그룹의 최빈값 ?
            rows.append({"묶는 기준": group_col, "그룹 수": len(table),
                         "일치율(%)": round(same.mean() * 100, 1)})
        return pd.DataFrame(rows)

    def handle_missing_values(self, strategy="group_mean", group_col=None, target_col=None):
        """target_col 의 빈칸을 group_col 별 대표값으로 채운다.

        strategy   : "group_mean"(평균) | "group_median"(중앙값) | "group_mode"(최빈값)
                     평균·중앙값은 숫자 컬럼에만, 글자 컬럼은 group_mode를 쓴다
        group_col  : 같은 그룹끼리 대표값을 구할 기준 컬럼. None이면 전체를 한 그룹으로 본다
        target_col : 빈칸을 채울 컬럼 (필수)

        이미 값이 있는 칸은 건드리지 않는다. 그룹에 값이 하나도 없어 대표값을 못 구하는 경우
        (또는 그룹 컬럼 자체가 비어 있는 행)는 전체 대표값으로 채운다.
        이번 대치 결과를 한 줄짜리 표로 돌려주고, self.missing_log 에도 쌓는다.
        """
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy는 {list(_STRATEGIES)} 중 하나여야 합니다: {strategy!r}")
        if target_col is None or target_col not in self.df.columns:
            raise KeyError(f"채울 컬럼(target_col)이 표에 없습니다: {target_col!r}")
        if group_col is not None and group_col not in self.df.columns:
            raise KeyError(f"묶는 기준 컬럼(group_col)이 표에 없습니다: {group_col!r}")
        target = self.df[target_col]
        if strategy != "group_mode" and not pd.api.types.is_numeric_dtype(target):
            raise TypeError(f"{target_col!r}는 숫자 컬럼이 아니라 평균·중앙값을 쓸 수 없습니다. group_mode를 쓰세요.")

        stat = _STRATEGIES[strategy]                               # 대표값을 구하는 방법
        before = int(target.isna().sum())                          # 채우기 전 빈칸 수
        overall = target.agg(stat)                                 # 전체 대표값 (마지막 보루)

        # ① 그룹별 대표값을 행마다 붙인다
        if group_col is None:
            group_values = pd.Series(overall, index=self.df.index)             # 모두 같은 한 그룹
        else:
            table = target.groupby(self.df[group_col]).agg(stat)               # {그룹 → 대표값} 표
            group_values = self.df[group_col].map(table)                       # 각 행의 그룹에 맞는 값

        # ② 빈칸에만 채운다 (값이 있는 칸은 그대로)
        filled = target.fillna(group_values)
        left_after_group = int(filled.isna().sum())

        # ③ 그룹 대표값이 없어 아직 빈 칸은 전체 대표값으로 채운다
        filled = filled.fillna(overall)
        after = int(filled.isna().sum())

        self.df[target_col] = filled
        row = {
            "컬럼": target_col, "전략": strategy, "묶는 기준": group_col or "(전체)",
            "채우기 전 결측": before,
            "그룹 값으로 채움": before - left_after_group,
            "전체 값으로 채움": left_after_group - after,
            "채운 후 결측": after,
        }
        self.missing_log.append(row)
        return pd.DataFrame([row])
