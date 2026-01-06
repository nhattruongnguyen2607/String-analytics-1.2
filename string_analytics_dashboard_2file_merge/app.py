# app.py
"""
Streamlit Merge Tool + Performance QA Dashboard
=============================================
1) Upload File A (DATA) + File B (CONFIG) (CSV/XLSX)
2) Preview 2 file song song
3) Merge theo key: label
4) Xem kết quả + download
5) Biểu đồ thống kê cho cột số
6) Dashboard: label có Performance thấp theo Plant
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------
# 0) App config
# ----------------------------
st.set_page_config(page_title="Merge & Performance Dashboard", layout="wide")


# ----------------------------
# 1) I/O: đọc file CSV/XLSX
# ----------------------------
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Đọc CSV robust theo một số encoding phổ biến."""
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(file_bytes))


def read_excel_bytes(file_bytes: bytes, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Đọc Excel và trả về (df, list_sheet_names)."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = xls.sheet_names
    sheet = sheet_name or sheets[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    return df, sheets


@st.cache_data(show_spinner=False)
def load_table(file_name: str, file_bytes: bytes, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Load CSV/XLSX -> (df, sheets_or_none)."""
    name = (file_name or "").lower()
    if name.endswith(".csv"):
        return read_csv_bytes(file_bytes), None
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df, sheets = read_excel_bytes(file_bytes, sheet_name=sheet_name)
        return df, sheets
    # fallback
    return read_csv_bytes(file_bytes), None


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def find_col(df: pd.DataFrame, canonical: str, aliases: List[str]) -> Optional[str]:
    """Tìm cột theo canonical/aliases (case-insensitive)."""
    norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in [canonical] + aliases:
        key = cand.strip().lower()
        if key in norm:
            return norm[key]
    return None


def to_numeric_safe(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def to_datetime_safe(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce", infer_datetime_format=True)
    return out


# ----------------------------
# 2) Merge logic
# ----------------------------
@dataclass
class MergeStats:
    rows_a: int
    rows_b: int
    rows_merged: int
    matched_rows: int
    unmatched_rows: int
    unique_key_a: int
    unique_key_b: int


def merge_on_label(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    how: str = "left",
    drop_dupe_config: bool = True,
) -> Tuple[pd.DataFrame, MergeStats, List[str]]:
    """
    Merge df_a (data) với df_b (config) theo label.
    - how: left (giữ all A) hoặc inner
    - drop_dupe_config: nếu config có label trùng -> giữ first để tránh nhân bản
    Returns: merged_df, stats, warnings(list)
    """
    warns = []
    a = df_a.copy()
    b = df_b.copy()

    a[key_a] = normalize_str_series(a[key_a])
    b[key_b] = normalize_str_series(b[key_b])

    # xử lý duplicate label trong config
    if b[key_b].duplicated().any():
        ndup = int(b[key_b].duplicated().sum())
        warns.append(f"CONFIG có {ndup:,} dòng label bị trùng.")
        if drop_dupe_config:
            b = b.drop_duplicates(subset=[key_b], keep="first")
            warns.append("Đã tự động drop duplicates trong CONFIG (giữ first).")

    merged = a.merge(b, left_on=key_a, right_on=key_b, how=how, suffixes=("", "_cfg"))

    # Nếu key khác tên, ưu tiên giữ tên label ở output
    if key_a != "label" and "label" not in merged.columns:
        merged = merged.rename(columns={key_a: "label"})
    if key_b != key_a:
        merged = merged.drop(columns=[key_b], errors="ignore")

    # matched = có ít nhất 1 cột từ config (trừ key) không NA
    b_cols = [c for c in b.columns if c != key_b]
    matched = int(merged[b_cols].notna().any(axis=1).sum()) if b_cols else 0
    total = int(len(merged))

    stats = MergeStats(
        rows_a=int(len(a)),
        rows_b=int(len(b)),
        rows_merged=total,
        matched_rows=matched,
        unmatched_rows=total - matched,
        unique_key_a=int(a[key_a].nunique(dropna=True)),
        unique_key_b=int(b[key_b].nunique(dropna=True)),
    )
    return merged, stats, warns


def download_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def download_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "merged") -> bytes:
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buff.seek(0)
    return buff.read()


# ----------------------------
# 3) UI blocks: preview + charts
# ----------------------------
def render_preview(title: str, df: pd.DataFrame, head_rows: int = 50):
    st.markdown(f"### {title}")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")

    with st.expander("Xem kiểu dữ liệu (dtypes)", expanded=False):
        st.dataframe(
            pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}),
            use_container_width=True,
        )

    st.dataframe(df.head(head_rows), use_container_width=True, height=380)


def render_basic_numeric_charts(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        st.info("Không có cột số (numeric) để vẽ biểu đồ.")
        return

    st.markdown("## Biểu đồ thống kê cơ bản")
    col1, col2 = st.columns([1, 1])

    with col1:
        metric_col = st.selectbox("Chọn cột số", options=num_cols, index=0)
    with col2:
        chart_type = st.selectbox("Loại biểu đồ", options=["Histogram", "Boxplot"], index=0)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=metric_col, nbins=30, title=f"Histogram: {metric_col}")
    else:
        if "Plant" in df.columns:
            fig = px.box(df, x="Plant", y=metric_col, points="outliers", title=f"Boxplot {metric_col} theo Plant")
        else:
            fig = px.box(df, y=metric_col, points="outliers", title=f"Boxplot: {metric_col}")

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Summary statistics")
    st.dataframe(df[num_cols].describe().T, use_container_width=True)


# ----------------------------
# 4) Dashboard: label performance thấp theo Plant
# ----------------------------
def low_performance_dashboard(df: pd.DataFrame):
    st.markdown("## Dashboard: kiểm tra label có Performance thấp theo từng Plant")

    label_col = find_col(df, "label", aliases=["lable", "Label", "LABEL"])
    plant_col = find_col(df, "Plant", aliases=["plant", "PLANT"])
    perf_col = find_col(df, "Performance", aliases=["performance", "PERFORMANCE", "perf"])
    date_col = find_col(df, "date", aliases=["Date", "DATE", "datetime", "time"])

    if not (label_col and plant_col and perf_col):
        st.info("Cần có cột **Plant**, **label**, **Performance** trong dữ liệu sau merge để chạy dashboard này.")
        return

    work = df.copy()
    work = to_numeric_safe(work, perf_col)
    if date_col:
        work = to_datetime_safe(work, date_col)

    # optional filters (nếu có)
    inverter_col = find_col(work, "Inverter", aliases=["inverter"])
    cap_col = find_col(work, "Capacity", aliases=["capacity"])
    az_col = find_col(work, "String Azimuth", aliases=["string azimuth", "azimuth"])

    with st.container(border=True):
        st.markdown("#### Bộ lọc")
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])

        plants = sorted(work[plant_col].dropna().astype(str).unique().tolist())
        with f1:
            plant_sel = st.multiselect("Plant", options=plants, default=plants[:1] if plants else [])
        with f2:
            bottom_n = st.number_input("Bottom N labels / Plant", min_value=3, max_value=50, value=10, step=1)

        with f3:
            inv_sel = []
            if inverter_col:
                inv_vals = sorted(work[inverter_col].dropna().astype(str).unique().tolist())
                inv_sel = st.multiselect("Inverter", options=inv_vals, default=inv_vals)

        with f4:
            az_sel = []
            if az_col:
                az_vals = sorted(work[az_col].dropna().astype(str).unique().tolist())
                az_sel = st.multiselect("String Azimuth", options=az_vals, default=az_vals)

        date_range = None
        if date_col and work[date_col].notna().any():
            dmin = work[date_col].min().date()
            dmax = work[date_col].max().date()
            date_range = st.date_input("Khoảng ngày", value=(dmin, dmax), min_value=dmin, max_value=dmax)

        cap_range = None
        cap_set = None
        if cap_col:
            if pd.api.types.is_numeric_dtype(work[cap_col]):
                cmin = float(np.nanmin(work[cap_col].values))
                cmax = float(np.nanmax(work[cap_col].values))
                cap_range = st.slider("Capacity (range)", min_value=cmin, max_value=cmax, value=(cmin, cmax))
            else:
                cap_vals = sorted(work[cap_col].dropna().astype(str).unique().tolist())
                cap_set = st.multiselect("Capacity", options=cap_vals, default=cap_vals)

    # Apply filters
    if plant_sel:
        work = work[work[plant_col].astype(str).isin([str(p) for p in plant_sel])]
    if inverter_col and inv_sel:
        work = work[work[inverter_col].astype(str).isin(inv_sel)]
    if az_col and az_sel:
        work = work[work[az_col].astype(str).isin(az_sel)]
    if cap_col:
        if cap_range is not None and pd.api.types.is_numeric_dtype(work[cap_col]):
            lo, hi = cap_range
            work = work[(work[cap_col] >= lo) & (work[cap_col] <= hi)]
        elif cap_set is not None:
            work = work[work[cap_col].astype(str).isin(cap_set)]
    if date_col and date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        work = work[(work[date_col].dt.date >= start) & (work[date_col].dt.date <= end)]

    if work.empty:
        st.warning("Không có dữ liệu sau khi lọc.")
        return

    # Aggregate avg performance by (Plant, label)
    agg = (
        work.dropna(subset=[plant_col, label_col])
        .groupby([plant_col, label_col], dropna=False)
        .agg(
            avg_performance=(perf_col, "mean"),
            min_performance=(perf_col, "min"),
            max_performance=(perf_col, "max"),
            n_points=(perf_col, "size"),
        )
        .reset_index()
    )

    # bottom N per plant
    agg = agg.sort_values([plant_col, "avg_performance"], ascending=[True, True])
    low = agg.groupby(plant_col, as_index=False).head(int(bottom_n))

    st.markdown("### Bảng label Performance thấp")
    st.dataframe(low, use_container_width=True)

    # Chart
    if low[plant_col].nunique() == 1:
        fig = px.bar(low.sort_values("avg_performance"), x=label_col, y="avg_performance", title="Bottom labels theo avg performance")
    else:
        fig = px.bar(
            low.sort_values("avg_performance"),
            x=label_col,
            y="avg_performance",
            color=plant_col,
            barmode="group",
            title="Bottom labels theo avg performance (so sánh nhiều plant)",
        )
    st.plotly_chart(fig, use_container_width=True)

    # Drill-down time series nếu có date
    if date_col and work[date_col].notna().any():
        st.markdown("### Drill-down: Performance day-by-day theo label")
        label_opts = low[label_col].astype(str).unique().tolist()
        sel_labels = st.multiselect("Chọn label để xem chi tiết", options=label_opts, default=label_opts[: min(3, len(label_opts))])
        if sel_labels:
            dd = work[work[label_col].astype(str).isin(sel_labels)].copy()
            dd = dd.dropna(subset=[date_col])
            dd["_day"] = dd[date_col].dt.to_period("D").dt.to_timestamp()
            ts = dd.groupby(["_day", label_col])[perf_col].mean().reset_index()

            fig2 = px.line(ts, x="_day", y=perf_col, color=label_col, markers=True, title="Performance day-by-day theo label")
            st.plotly_chart(fig2, use_container_width=True)


# ----------------------------
# 5) Sidebar: upload + merge settings
# ----------------------------
st.sidebar.title("📥 Upload & Merge")

file_a = st.sidebar.file_uploader("File A (DATA) - CSV/XLSX", type=["csv", "xlsx", "xls"], key="file_a")
file_b = st.sidebar.file_uploader("File B (CONFIG) - CSV/XLSX", type=["csv", "xlsx", "xls"], key="file_b")

st.sidebar.markdown("---")
st.sidebar.subheader("Merge Settings")
key_a = st.sidebar.text_input("Key column File A", value="label")
key_b = st.sidebar.text_input("Key column File B", value="label")
how = st.sidebar.selectbox("Merge type", options=["left", "inner"], index=0)
drop_dupe_cfg = st.sidebar.checkbox("CONFIG: drop duplicate label (keep first)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Pre-processing (optional)")
drop_na = st.sidebar.checkbox("Drop rows with NA", value=False)
dedup = st.sidebar.checkbox("Drop duplicated rows", value=False)


# ----------------------------
# 6) Main: Preview 2 cột + merged result + charts + dashboard
# ----------------------------
st.title("🔗 Streamlit Merge Tool + Performance Dashboard")
st.caption("Upload 2 file (data + config), preview song song, merge theo label, tải kết quả và kiểm tra label performance thấp theo plant.")

if not file_a or not file_b:
    st.info("Hãy upload **cả 2 file** ở sidebar để bắt đầu.")
    st.stop()

bytes_a = file_a.getvalue()
bytes_b = file_b.getvalue()

# Nếu excel: cho chọn sheet (đọc 1 lần để lấy sheet list)
df_a_tmp, sheets_a = load_table(file_a.name, bytes_a, sheet_name=None)
df_b_tmp, sheets_b = load_table(file_b.name, bytes_b, sheet_name=None)

sheet_a = None
sheet_b = None
if sheets_a:
    sheet_a = st.sidebar.selectbox("Sheet File A", options=sheets_a, index=0)
if sheets_b:
    sheet_b = st.sidebar.selectbox("Sheet File B", options=sheets_b, index=0)

df_a, _ = load_table(file_a.name, bytes_a, sheet_name=sheet_a)
df_b, _ = load_table(file_b.name, bytes_b, sheet_name=sheet_b)

df_a = clean_columns(df_a)
df_b = clean_columns(df_b)

if drop_na:
    df_a = df_a.dropna()
    df_b = df_b.dropna()
if dedup:
    df_a = df_a.drop_duplicates()
    df_b = df_b.drop_duplicates()

# Preview area: 2 columns
c1, c2 = st.columns(2)
with c1:
    render_preview("Preview File A (DATA)", df_a)
with c2:
    render_preview("Preview File B (CONFIG)", df_b)

# Validate key columns
if key_a not in df_a.columns:
    st.error(f"File A không có cột '{key_a}'. Cột hiện có: {list(df_a.columns)}")
    st.stop()
if key_b not in df_b.columns:
    st.error(f"File B không có cột '{key_b}'. Cột hiện có: {list(df_b.columns)}")
    st.stop()

# Merge
merged, stats, warns = merge_on_label(
    df_a, df_b, key_a=key_a, key_b=key_b, how=how, drop_dupe_config=drop_dupe_cfg
)

st.markdown("---")
st.markdown("## ✅ Kết quả merge")

if warns:
    for w in warns:
        st.warning(w)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows A", f"{stats.rows_a:,}")
k2.metric("Rows B", f"{stats.rows_b:,}")
k3.metric("Rows merged", f"{stats.rows_merged:,}")
k4.metric("Matched", f"{stats.matched_rows:,}")
k5.metric("Unmatched", f"{stats.unmatched_rows:,}")

if stats.unmatched_rows > 0:
    st.info("Unmatched = label trong File A không tìm thấy trong File B. Các cột config ở dòng đó sẽ bị trống.")

st.dataframe(merged.head(300), use_container_width=True, height=450)

# Download buttons
d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "⬇️ Download CSV",
        data=download_csv_bytes(merged),
        file_name="merged_result.csv",
        mime="text/csv",
    )
with d2:
    st.download_button(
        "⬇️ Download Excel",
        data=download_xlsx_bytes(merged),
        file_name="merged_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Charts
st.markdown("---")
render_basic_numeric_charts(merged)

# Low performance dashboard
st.markdown("---")
low_performance_dashboard(merged)
