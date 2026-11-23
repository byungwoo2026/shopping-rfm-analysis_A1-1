"""분석 도구 모음: DataAnalyzer 클래스.

CSV 표 하나를 불러와서 정리하고 분석하는 메서드를 한 클래스에 모아 둡니다.
같은 클래스로 거래표(Online Retail II)와 상품표(Fashion)를 각각 다룹니다.
(표마다 DataAnalyzer를 하나씩 만들므로 인스턴스가 2개가 됩니다.)
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg    # mpimg.imread(): 사진 파일을 숫자 배열로 읽는 함수 (plt.imread와 같은 함수)

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


# 상품이 아닌 상품코드(배송비·수기 입력·수수료·조정·테스트). 고객이 상품을 산 매출이 아니라서
# 매출·이상치를 볼 때 따로 구분해야 한다. (실제 상품인 DCGS…, gift_0001_… 코드는 넣지 않음)
NON_PRODUCT_CODES = {
    "POST": "배송비(우편)", "DOT": "배송비(닷컴)", "C2": "운송비",
    "M": "수기 입력", "m": "수기 입력",
    "ADJUST": "금액 조정", "ADJUST2": "금액 조정", "B": "부실채권 조정",
    "BANK CHARGES": "은행 수수료", "AMAZONFEE": "아마존 수수료",
    "D": "할인", "S": "샘플", "TEST001": "테스트", "TEST002": "테스트",
}


def iqr_bounds(values, threshold=1.5):
    """IQR 방식으로 '정상 범위'를 구한다.

    ① 값을 크기순으로 줄 세운 뒤 25% 지점 = Q1, 75% 지점 = Q3
    ② IQR = Q3 - Q1  (가운데 50% 데이터가 퍼져 있는 폭)
    ③ 정상 범위 = Q1 - threshold × IQR  ~  Q3 + threshold × IQR
       이 범위 밖의 값이 이상치다. threshold의 관례값은 1.5이고, 클수록 범위가 넓어져 이상치가 줄어든다.

    돌려주는 값: {"q1", "q3", "iqr", "lower"(하한), "upper"(상한)} 딕셔너리
    빈칸(NaN)은 계산에서 빠진다.
    """
    if threshold < 0:
        raise ValueError(f"threshold는 0 이상이어야 합니다: {threshold}")
    values = pd.Series(values).dropna()
    q1 = values.quantile(0.25)          # 25% 지점 (사이 값은 앞뒤 두 값을 비율로 섞어서 구함)
    q3 = values.quantile(0.75)          # 75% 지점
    iqr = q3 - q1
    return {"q1": q1, "q3": q3, "iqr": iqr,
            "lower": q1 - threshold * iqr, "upper": q3 + threshold * iqr}


def _mode(values):
    """최빈값(가장 많이 나온 값) 하나를 돌려준다. 값이 하나도 없으면 NaN.

    동점이면 정렬했을 때 앞에 오는 값을 고른다(실행할 때마다 같은 결과가 나오게).
    """
    counts = values.mode()                          # 최빈값 목록 (빈칸은 자동으로 제외됨)
    return counts.iloc[0] if len(counts) else np.nan


# 고객 세그먼트 규칙: {이름: 조건을 계산하는 함수}. 위에서부터 차례로 보고 먼저 맞는 이름을 붙인다.
# (순서가 중요: VIP 조건을 만족하는 고객이 Loyal로 먼저 빠지지 않도록 VIP를 맨 위에 둔다.)
# 함수는 RFM 표(rfm)를 받아 "그 세그먼트인 고객이면 True"인 판정표를 돌려준다.
# New는 점수가 아니라 실제 구매 횟수(frequency == 1)로 판정한다. F 점수는 동점을 행 순서로 가르기 때문에
# 같은 1회 구매 고객이 서로 다른 점수를 받을 수 있어서다.
DEFAULT_SEGMENT_RULES = {
    "VIP":     lambda rfm: (rfm["r_score"] >= 4) & (rfm["f_score"] >= 4) & (rfm["m_score"] >= 4),   # 최근에, 자주, 많이
    "Loyal":   lambda rfm: (rfm["r_score"] >= 3) & (rfm["f_score"] >= 3),                           # 꾸준히 오는 고객
    "New":     lambda rfm: (rfm["r_score"] >= 4) & (rfm["frequency"] == 1),                         # 최근에 처음 온 고객
    "At Risk": lambda rfm: (rfm["r_score"] <= 2) & (rfm["f_score"] >= 3),                           # 자주 왔는데 요즘 안 오는 고객
    "Churned": lambda rfm: rfm["r_score"] <= 2,                                                      # 오래 안 온 이탈 고객
}


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
        self.cancel_rows = None     # clean_transactions()가 지운 취소 전표(원 주문과 짝을 맞출 때 필요)
        self.cancelled_orders = None  # remove_cancelled_orders()가 지운 "취소로 사라진 원 주문" 행들
        self.image_log = None       # load_images()가 남기는 "사진을 몇 장 찾고 몇 장 썼는지" 기록
        self.rfm = None             # calculate_rfm()이 만드는 고객별 R·F·M 표 (segment_customers()가 세그먼트 컬럼을 더함)
        self.rfm_info = None        # calculate_rfm()의 기준일·제외한 행 같은 기록
        self.segment_order = None   # segment_customers()가 쓴 세그먼트 이름 순서 (요약표 정렬용)

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
        self.cancel_rows = df[is_cancel]                 # 지우기 전에 기억: 나중에 취소와 원 주문의 짝을 맞출 때 씀
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

    # ───────────── 이상치 ─────────────
    def _numeric_column(self, column):
        """이상치 계산 전에 표가 있는지, 숫자 컬럼이 맞는지 점검하고 그 컬럼을 돌려준다."""
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        if column not in self.df.columns:
            raise KeyError(f"표에 없는 컬럼입니다: {column!r}")
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"{column!r}는 숫자 컬럼이 아니라 계산할 수 없습니다.")
        return self.df[column]

    def detect_outliers(self, column, threshold=1.5):
        """column 값이 IQR 정상 범위를 벗어난 행들을 돌려준다. (self.df 자체는 바꾸지 않는다)

        threshold : IQR에 곱하는 배수. 기본 1.5. 크게 하면 더 극단적인 값만 이상치로 잡힌다.
        """
        values = self._numeric_column(column)
        bounds = iqr_bounds(values, threshold)
        is_outlier = (values < bounds["lower"]) | (values > bounds["upper"])   # 범위 밖이면 True
        return self.df[is_outlier]

    def outlier_summary(self, columns, threshold=1.5):
        """여러 컬럼의 Q1·Q3·IQR·정상 범위와 이상치 개수·비율을 한 표로 보여 준다."""
        rows = []
        for column in columns:
            values = self._numeric_column(column)
            bounds = iqr_bounds(values, threshold)
            below = int((values < bounds["lower"]).sum())     # 하한보다 작은 값
            above = int((values > bounds["upper"]).sum())     # 상한보다 큰 값
            rows.append({
                "컬럼": column, "Q1": bounds["q1"], "Q3": bounds["q3"], "IQR": bounds["iqr"],
                "하한": bounds["lower"], "상한": bounds["upper"],
                "아래쪽 이상치": below, "위쪽 이상치": above, "이상치 수": below + above,
                "비율(%)": round((below + above) / values.notna().sum() * 100, 2),
            })
        return pd.DataFrame(rows)

    # ───────────── 추가 정제 (이상치를 살펴보다 발견한 문제) ─────────────
    def _append_cleaning_log(self, label, removed):
        """cleaning_log 맨 아래에 "이 단계에서 removed행을 지웠고 지금 len(self.df)행 남았다"를 한 줄 더한다."""
        row = pd.DataFrame([(label, int(removed), len(self.df))], columns=["단계", "제거한 행", "남은 행"])
        self.cleaning_log = pd.concat([self.cleaning_log, row], ignore_index=True)
        return self.cleaning_log

    def remove_non_product_rows(self, codes=None):
        """상품이 아닌 코드(배송비·수기 입력·수수료·조정·테스트)의 행을 지운다.

        codes : 지울 상품코드 목록. 안 주면 NON_PRODUCT_CODES(14종)를 쓴다.
        clean_transactions() 다음에 부른다. 단계별 제거 표(누적)를 돌려준다.
        """
        if self.cleaning_log is None:
            raise ValueError("먼저 clean_transactions()를 실행하세요.")
        codes = NON_PRODUCT_CODES if codes is None else codes
        is_non_product = self.df["stock_code"].isin(codes)       # 비판매 코드이면 True
        removed = int(is_non_product.sum())
        self.df = self.df[~is_non_product].reset_index(drop=True)
        return self._append_cleaning_log(f"비판매 코드 제거 ({len(codes)}종)", removed)

    def remove_cancelled_orders(self, hours=24):
        """취소로 상쇄된 원 주문을 지운다.

        clean_transactions()는 취소 전표(주문번호 C…)만 지우므로, 이미 취소된 원 주문이 "판매"로 남아 있다.
        아래 조건을 모두 만족하는 취소 전표가 있으면 그 원 주문을 없던 일로 보고 지운다.
          · 같은 고객, 같은 상품, 같은 수량(전량 취소)
          · 취소가 원 주문 뒤 hours시간 이내
        취소 한 건은 가장 가까운 앞선 원 주문 한 건만 지운다. 고객 번호가 비어 있으면 짝을 맞출 수 없어 건너뛴다.
        지운 원 주문은 self.cancelled_orders 에 남긴다. 단계별 제거 표(누적)를 돌려준다.
        """
        if self.cancel_rows is None or self.cleaning_log is None:
            raise ValueError("먼저 clean_transactions()를 실행하세요 (취소 전표를 기억해 둬야 짝을 맞출 수 있습니다).")
        if hours <= 0:
            raise ValueError(f"hours는 0보다 커야 합니다: {hours}")
        key = ["customer_id", "stock_code", "quantity"]

        # 원 주문 후보: 고객 번호가 있는 행. 행 번호(row_id)를 같이 들고 다닌다.
        sales = self.df.loc[self.df["customer_id"].notna(), key + ["order_date"]].copy()
        sales["row_id"] = sales.index
        # 취소 전표: 수량은 음수(-5)로 적혀 있으니 부호를 바꿔 원 주문 수량(5)과 비교한다.
        cancels = self.cancel_rows.dropna(subset=["customer_id"])[key + ["order_date"]].copy()
        cancels["quantity"] = -cancels["quantity"]

        # 취소마다 "같은 (고객, 상품, 수량)이면서 취소 시각 이전 hours시간 안에서 가장 가까운 원 주문"을 찾는다.
        matched = pd.merge_asof(
            cancels.sort_values("order_date"), sales.sort_values("order_date"),
            on="order_date", by=key, direction="backward", tolerance=pd.Timedelta(hours=hours),
        )
        drop_ids = matched["row_id"].dropna().astype(int).unique()    # 짝이 맞은 원 주문의 행 번호
        self.cancelled_orders = self.df.loc[drop_ids]
        self.df = self.df.drop(index=drop_ids).reset_index(drop=True)
        return self._append_cleaning_log(f"취소와 짝이 맞는 원 주문 제거 (같은 고객·상품·수량, {hours}시간 이내)", len(drop_ids))

    # ───────────── 이상치 처리 ─────────────
    def treat_outliers(self, column, method="clip", threshold=1.5):
        """이상치를 처리한 "새 값 목록(Series)"을 돌려준다. self.df 와 원본 컬럼은 바뀌지 않는다.

        method="clip"   : 정상 범위 밖의 값을 하한/상한 값으로 자른다. (행·개수는 그대로)
        method="remove" : 정상 범위 밖의 값을 빈칸(NaN)으로 바꿔 평균 등 통계에서 제외한다. (행은 그대로)
        돌려받은 값은 df["amount_clipped"] = analyzer.treat_outliers("amount") 처럼 새 컬럼으로 붙여 쓴다.
        (원본 컬럼은 그대로 두므로, RFM처럼 원본이 필요한 계산에는 영향이 없다.)
        """
        if method not in ("clip", "remove"):
            raise ValueError(f"method는 'clip' 또는 'remove'여야 합니다: {method!r}")
        values = self._numeric_column(column)
        bounds = iqr_bounds(values, threshold)
        if method == "clip":
            treated = values.clip(lower=bounds["lower"], upper=bounds["upper"])         # 범위 밖은 경계값으로
            treated.name = f"{column}_clipped"
        else:
            in_range = (values >= bounds["lower"]) & (values <= bounds["upper"])
            treated = values.where(in_range)                                            # 범위 밖은 NaN
            treated.name = f"{column}_removed"
        return treated

    # ───────────── 텍스트 피처 ─────────────
    def add_text_features(self, text_col, prefix=None):
        """글자 컬럼에서 단어 수·글자 수를 세어 새 컬럼 두 개를 추가한다.

        text_col : 글자가 들어 있는 컬럼 (예: 상품명)
        prefix   : 새 컬럼 이름의 앞부분. 안 주면 text_col 이름을 쓴다 → "{prefix}_word_count", "{prefix}_char_count"
        단어 수 = 공백으로 나눈 조각 수 (연속 공백도 한 번의 구분으로 침),
        글자 수 = 앞뒤 공백을 뺀 길이 (가운데 공백은 포함).
        빈칸(NaN)인 행은 새 컬럼도 빈칸으로 둔다. 새로 만든 두 컬럼을 표로 돌려준다.
        """
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        if text_col not in self.df.columns:
            raise KeyError(f"표에 없는 컬럼입니다: {text_col!r}")
        if not pd.api.types.is_string_dtype(self.df[text_col]) and not pd.api.types.is_object_dtype(self.df[text_col]):
            raise TypeError(f"{text_col!r}는 글자 컬럼이 아닙니다.")
        prefix = text_col if prefix is None else prefix
        word_col, char_col = f"{prefix}_word_count", f"{prefix}_char_count"
        text = self.df[text_col]
        self.df[word_col] = text.str.split().str.len()         # 공백 기준으로 쪼갠 조각의 개수
        self.df[char_col] = text.str.strip().str.len()         # 앞뒤 공백을 뺀 글자 수
        return self.df[[word_col, char_col]]

    # ───────────── 이미지 피처 ─────────────
    def load_images(self, ids, image_dir, expected_shape=(80, 60, 3)):
        """상품 id 목록에 해당하는 사진 파일({id}.jpg)을 읽어, 크기가 같은 것만 하나의 배열로 쌓는다.

        expected_shape : 쌓을 사진의 크기 (세로, 가로, 색 3개). 이 크기와 다른 사진(흑백 등)은 건너뛴다.
        돌려주는 값: (images, used_ids)
          images   : (사진 수 N, 세로, 가로, 3) 모양의 uint8 배열. 값은 0(어두움)~255(밝음)
          used_ids : 실제로 쌓은 사진의 id 목록 (길이 N, images와 같은 순서)
        사진을 한 장씩 읽는 것은 파일 입출력이라 반복이 필요하다. 평균·표준편차 계산에는 반복을 쓰지 않는다.
        몇 장을 찾고 몇 장을 건너뛰었는지는 self.image_log 에 남긴다.
        """
        image_dir = Path(image_dir)
        expected_shape = tuple(expected_shape)
        found = [(i, image_dir / f"{i}.jpg") for i in ids if (image_dir / f"{i}.jpg").exists()]   # 파일이 있는 것만
        arrays = [(i, mpimg.imread(path)) for i, path in found]                                   # 한 장씩 읽기 (반복이 필요한 부분)
        same_size = [(i, array) for i, array in arrays if array.shape == expected_shape]         # 크기가 맞는 것만
        if not same_size:
            raise ValueError(f"크기가 {expected_shape}인 사진을 하나도 찾지 못했습니다.")
        used_ids = [i for i, _ in same_size]
        images = np.stack([array for _, array in same_size])      # 사진 N장을 (N, 세로, 가로, 3) 하나의 큰 상자로 쌓음
        self.image_log = {
            "요청한 id": len(ids),
            "파일을 찾음": len(found),
            "쌓은 사진": len(same_size),
            "크기가 달라 건너뜀": len(arrays) - len(same_size),
            "건너뛴 사진의 크기": dict(Counter(str(a.shape) for _, a in arrays if a.shape != expected_shape)),
        }
        return images, used_ids

    def extract_image_features(self, images, ids=None, step=1):
        """사진 배열에서 사진별 평균·표준편차를 구한다. 반복문 없이 배열 연산 한 번으로 계산한다.

        images : (사진 수 N, 세로, 가로[, 색]) 배열. 0번 축이 "사진 번호"
        ids    : images와 같은 순서의 상품 id 목록. 주면 결과를 self.df 의 image_mean, image_std 컬럼에도 넣는다
                 (사진이 없는 행은 빈칸). 안 주면 결과 표만 돌려주고 self.df 는 건드리지 않는다.
        step   : 다운샘플 간격. 2이면 가로·세로로 한 칸씩 건너뛰며 (4분의 1만) 계산해 더 빠르지만 근사값이 된다.
        image_mean = 사진 전체의 밝기(0~255), image_std = 밝은 부분과 어두운 부분의 차이(대비).
        """
        images = np.asarray(images)
        if images.ndim < 3:
            raise ValueError("images는 (사진 수, 세로, 가로[, 색]) 모양의 배열이어야 합니다.")
        if step < 1 or int(step) != step:
            raise ValueError(f"step은 1 이상의 정수여야 합니다: {step}")
        if ids is not None and len(ids) != len(images):
            raise ValueError(f"ids({len(ids)}개)와 사진 수({len(images)}장)가 다릅니다.")
        if ids is not None and self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")

        pixels = images[:, ::step, ::step]                         # 세로·가로를 step칸 간격으로 (step=1이면 전부)
        axes = tuple(range(1, pixels.ndim))                        # 0번 축(사진 번호)만 남기고 나머지 축 전부
        features = pd.DataFrame({
            "image_mean": pixels.mean(axis=axes),                  # 사진 N장의 평균이 한 번에 나옴 → 길이 N
            "image_std": pixels.std(axis=axes),                    # 사진 N장의 표준편차가 한 번에 나옴 → 길이 N
        })
        if ids is None:
            return features
        features.insert(0, "id", list(ids))
        self.df = self.df.drop(columns=["image_mean", "image_std"], errors="ignore").merge(features, on="id", how="left")
        return features

    # ───────────── 통계 요약 ─────────────
    def summary_stats(self, columns):
        """숫자 컬럼들의 기초 통계를 한 표로 보여 준다. (컬럼마다 한 줄)

        개수, 평균, 중앙값, 표준편차, 최솟값, Q1(25% 지점), Q3(75% 지점), 최댓값, 그리고
        '평균/중앙값'(1보다 훨씬 크면 큰 극단값이 평균을 끌어올린 오른쪽 꼬리 분포).
        표준편차는 pandas 방식(하나 뺀 값으로 나누는 표본 표준편차)이다.
        """
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        columns = list(columns)
        for column in columns:                                  # 표에 있는 숫자 컬럼인지 점검
            self._numeric_column(column)
        data = self.df[columns]
        table = pd.DataFrame({
            "개수": data.count(), "평균": data.mean(), "중앙값": data.median(), "표준편차": data.std(),
            "최솟값": data.min(), "Q1": data.quantile(0.25), "Q3": data.quantile(0.75), "최댓값": data.max(),
        })
        table["평균/중앙값"] = table["평균"] / table["중앙값"]
        return table

    # ───────────── RFM ─────────────
    @staticmethod
    def _quintile_score(values, labels):
        """값을 작은 순서로 줄 세워 5등분하고, 구간마다 labels[0]~labels[4] 점수를 준다.

        pd.qcut 은 같은 값이 너무 많으면 구간 경계가 겹쳐 오류가 난다. 그래서 먼저 rank(method="first")로
        같은 값에도 줄 번호를 따로 매긴 뒤 나눈다. (부작용: 같은 값이 행 순서에 따라 다른 점수를 받을 수 있다.)
        """
        ranks = values.rank(method="first")                       # 같은 값도 1, 2, 3 … 처럼 서로 다른 순위
        return pd.qcut(ranks, 5, labels=labels).astype(int)       # 순위 기준으로 5등분

    def calculate_rfm(self, customer_col, date_col, amount_col, invoice_col=None, ref_date=None):
        """고객별 R(최근성)·F(빈도)·M(금액)과 1~5점을 계산해 표로 돌려준다. (고객 번호가 인덱스)

        customer_col : 고객 번호 컬럼. 이 값이 빈 행은 누구인지 몰라서 계산에서 뺀다
        date_col     : 주문 시각 컬럼
        amount_col   : 한 줄의 금액 컬럼 (원본 금액. 극단값을 자른 컬럼이 아니다)
        invoice_col  : 주문번호 컬럼. 주면 F = 고유 주문번호 수, 안 주면 F = 행 수
                       (한 주문에 상품이 여러 줄이면 행 수는 방문 횟수가 아니라 장바구니 크기를 재게 된다)
        ref_date     : 기준일. 안 주면 "마지막 거래일 + 1일"을 쓴다.
                       +1일의 이유: 마지막 날 구매한 고객의 R이 0일이 되는 것을 피하고 모두 1일 이상으로 계산하려고

        recency   = 기준일 − 고객의 마지막 구매일 (일수, 작을수록 좋음)
        frequency = 주문 횟수 (클수록 좋음)
        monetary  = 구매 금액 합계 (클수록 좋음)
        r_score·f_score·m_score = 고객 순위를 5등분한 1~5점 (5점이 가장 좋음. R은 작을수록 좋아서 점수를 거꾸로 매김)
        """
        if self.df is None:
            raise ValueError("먼저 load_data()로 데이터를 불러오세요.")
        needed = [customer_col, date_col, amount_col] + ([invoice_col] if invoice_col else [])
        missing = [c for c in needed if c not in self.df.columns]
        if missing:
            raise KeyError(f"표에 없는 컬럼입니다: {missing}")

        data = self.df.dropna(subset=[customer_col])                     # 누구인지 아는 행만
        if data.empty:
            raise ValueError("고객 번호가 있는 행이 없습니다.")
        last_date = data[date_col].max()
        ref_date = last_date.normalize() + pd.Timedelta(days=1) if ref_date is None else pd.Timestamp(ref_date).normalize()
        if ref_date < last_date.normalize():
            raise ValueError(f"기준일({ref_date:%Y-%m-%d})이 마지막 거래일({last_date:%Y-%m-%d})보다 앞서면 R이 음수가 됩니다.")

        grouped = data.groupby(customer_col)                              # 고객별로 묶기
        rfm = pd.DataFrame({
            "recency": (ref_date - grouped[date_col].max().dt.normalize()).dt.days,   # 기준일 − 마지막 구매일 (일수)
            "frequency": grouped[invoice_col].nunique() if invoice_col else grouped[date_col].size(),
            "monetary": grouped[amount_col].sum(),
        })
        if len(rfm) < 5:
            raise ValueError(f"고객이 {len(rfm)}명뿐이라 5등분할 수 없습니다(최소 5명).")
        if pd.api.types.is_float_dtype(rfm.index) and (rfm.index == rfm.index.astype("int64")).all():
            rfm.index = rfm.index.astype("int64")                  # 빈칸 때문에 소수형이 된 고객 번호를 12346.0 대신 12346으로
        rfm["r_score"] = self._quintile_score(rfm["recency"], [5, 4, 3, 2, 1])     # R은 작을수록 좋으므로 점수를 거꾸로
        rfm["f_score"] = self._quintile_score(rfm["frequency"], [1, 2, 3, 4, 5])
        rfm["m_score"] = self._quintile_score(rfm["monetary"], [1, 2, 3, 4, 5])

        self.rfm = rfm
        self.rfm_info = {
            "기준일": ref_date, "마지막 거래 시각": last_date, "RFM 대상 고객 수": len(rfm),
            "고객 번호가 없어 제외한 행": len(self.df) - len(data),
            "제외한 행의 금액": float(self.df.loc[self.df[customer_col].isna(), amount_col].sum()),
        }
        return rfm

    def segment_customers(self, rules=None, default="Others"):
        """RFM 점수로 고객에게 세그먼트 이름을 붙여, rfm 표의 segment 컬럼에 넣고 표를 돌려준다.

        rules   : {세그먼트 이름: 조건 함수} 딕셔너리. 안 주면 DEFAULT_SEGMENT_RULES.
                  위에서부터 차례로 보고 먼저 맞는 이름을 붙인다 (순서가 중요).
        default : 어떤 규칙에도 안 맞는 고객의 이름
        """
        if self.rfm is None:
            raise ValueError("먼저 calculate_rfm()으로 RFM을 계산하세요.")
        rules = DEFAULT_SEGMENT_RULES if rules is None else rules
        conditions = [rule(self.rfm) for rule in rules.values()]          # 규칙마다 "맞는 고객이면 True"인 판정표
        # np.select: 판정표를 순서대로 보고 처음 True인 규칙의 이름을 고른다. 모두 False면 default
        self.rfm["segment"] = np.select(conditions, list(rules), default=default)
        self.segment_order = list(rules) + [default]
        return self.rfm

    def segment_summary(self):
        """세그먼트별 고객 수·평균 금액·매출 비중 요약표를 돌려준다. (segment_customers() 다음에 부른다)"""
        if self.rfm is None or "segment" not in self.rfm.columns:
            raise ValueError("먼저 segment_customers()로 세그먼트를 붙이세요.")
        grouped = self.rfm.groupby("segment")
        table = pd.DataFrame({
            "고객 수": grouped.size(),
            "평균 최근성 R(일)": grouped["recency"].mean(),
            "평균 구매 횟수 F": grouped["frequency"].mean(),
            "평균 금액 M": grouped["monetary"].mean(),
            "매출 합계": grouped["monetary"].sum(),
        })
        table.insert(1, "고객 비율(%)", table["고객 수"] / table["고객 수"].sum() * 100)
        table["매출 비중(%)"] = table["매출 합계"] / table["매출 합계"].sum() * 100
        order = [name for name in self.segment_order if name in table.index]    # 규칙 순서대로
        return table.loc[order]
