# app.py
"""
Streamlit: Merge Tool + Performance QA Dashboard (Sidebar separated by tabs)
--------------------------------------------------------------------------
Yêu cầu:
- Gom (merge) chức năng upload + merge vào 1 sidebar (tab MERGE)
- Chức năng phân tích vào 1 sidebar riêng (tab ANALYSIS)
- Main area:
  - Preview File A & File B (2 columns)
  - Merged result + download
  - Charts + dashboard low-performance labels per Plant (dùng settings từ ANALYSIS tab)

Run:
  pip install -r requirements.txt
  streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# 0) App config
# =========================
st.set_page_config(page_title="Merge & Performance Dashboard", layout="wide")


# =========================
# 1) Data I/O utilities
# =========================
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Robust CSV reader with common encodings."""
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(file_bytes))


def read_excel_bytes(file_bytes: bytes, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Read Excel bytes; return (df, sheet_names)."""
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
    """Strip column names to reduce key mismatch."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_str_series(s: pd.Series) -> pd.Series:
    """Normalize join key: string + strip."""
    return s.astype(str).str.strip()


def find_col(df: pd.DataFrame, canonical: str, aliases: List[str]) -> Optional[str]:
    """Find a column (case-insensitive) from canonical + aliases."""
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


# =========================
# 2) Merge logic
# =========================
@dataclass
class MergeStats:
    rows_a: int
    rows_b: int
    rows_merged: int
    matched_rows: int
    unmatched_rows: int
    unique_key_a: int
    unique_key_b: int


def merge_on_key(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_a: str,
    key_b: str,
    how: str = "left",
    drop_dupe_config: bool = True,
) -> Tuple[pd.DataFrame, MergeStats, List[str]]:
    """
    Merge df_a (DATA) với df_b (CONFIG) theo key.
    - Default: LEFT JOIN (giữ toàn bộ rows của DATA)
    - Optional: drop duplicates trong CONFIG theo key để tránh nhân bản.
    Returns: (merged_df, stats, warnings)
    """
    warns: List[str] = []
    a = df_a.copy()
    b = df_b.copy()

    # Key normalize
    a[key_a] = normalize_str_series(a[key_a])
    b[key_b] = normalize_str_series(b[key_b])

    # Duplicates in config
    if b[key_b].duplicated().any():
        ndup = int(b[key_b].duplicated().sum())
        warns.append(f"CONFIG có {ndup:,} dòng '{key_b}' bị trùng.")
        if drop_dupe_config:
            b = b.drop_duplicates(subset=[key_b], keep="first")
            warns.append("Đã tự động xoá duplicates trong CONFIG (giữ dòng đầu tiên theo key).")

    merged = a.merge(b, left_on=key_a, right_on=key_b, how=how, suffixes=("", "_cfg"))

    # Keep output key clean
    if key_a != "label" and "label" not in merged.columns and key_a in merged.columns:
        merged = merged.rename(columns={key_a: "label"})
    if key_b != key_a:
        merged = merged.drop(columns=[key_b], errors="ignore")

    # matched rows: any non-null from config columns (excluding its key)
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


# =========================
# 3) Analytics computations (no UI here)
# =========================
def compute_bottom_labels_per_plant(
    df: pd.DataFrame,
    plant_col: str,
    label_col: str,
    perf_col: str,
    date_col: Optional[str] = None,
    bottom_n: int = 10,
) -> pd.DataFrame:
    """
    Compute avg/min/max/count per (Plant, label), then return bottom_n labels per Plant by avg_performance.
    """
    work = df.copy()
    work[perf_col] = pd.to_numeric(work[perf_col], errors="coerce")

    agg_map: Dict[str, Tuple[str, str]] = {
        "avg_performance": (perf_col, "mean"),
        "min_performance": (perf_col, "min"),
        "max_performance": (perf_col, "max"),
        "n_points": (perf_col, "size"),
    }
    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce", infer_datetime_format=True)
        agg_map["last_date"] = (date_col, "max")

    out = (
        work.dropna(subset=[plant_col, label_col])
        .groupby([plant_col, label_col], dropna=False)
        .agg(**agg_map)
        .reset_index()
        .sort_values([plant_col, "avg_performance"], ascending=[True, True])
    )
    return out.groupby(plant_col, as_index=False).head(int(bottom_n))


def apply_analysis_filters(
    df: pd.DataFrame,
    plant_col: str,
    label_col: str,
    perf_col: str,
    date_col: Optional[str],
    inverter_col: Optional[str],
    capacity_col: Optional[str],
    az_col: Optional[str],
    plant_sel: List[str],
    inv_sel: List[str],
    az_sel: List[str],
    cap_range: Optional[Tuple[float, float]],
    cap_set: Optional[List[str]],
    date_range: Optional[Tuple[Any, Any]],
) -> pd.DataFrame:
    """Apply filters from ANALYSIS sidebar to merged dataframe."""
    work = df.copy()
    work = to_numeric_safe(work, perf_col)
    if date_col:
        work = to_datetime_safe(work, date_col)

    # Plant
    if plant_sel:
        work = work[work[plant_col].astype(str).isin([str(p) for p in plant_sel])]

    # Inverter
    if inverter_col and inv_sel:
        work = work[work[inverter_col].astype(str).isin(inv_sel)]

    # Azimuth
    if az_col and az_sel:
        work = work[work[az_col].astype(str).isin(az_sel)]

    # Capacity
    if capacity_col:
        if cap_range is not None and pd.api.types.is_numeric_dtype(work[capacity_col]):
            lo, hi = cap_range
            work = work[(work[capacity_col] >= lo) & (work[capacity_col] <= hi)]
        elif cap_set is not None:
            work = work[work[capacity_col].astype(str).isin(cap_set)]

    # Date range
    if date_col and date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        # only filter if date parsing succeeded
        if work[date_col].notna().any():
            work = work[(work[date_col].dt.date >= start) & (work[date_col].dt.date <= end)]

    return work


# =========================
# 4) Main area renderers
# =========================
def render_preview(title: str, df: pd.DataFrame, head_rows: int = 50):
    st.markdown(f"### {title}")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")

    with st.expander("Dtypes", expanded=False):
        st.dataframe(
            pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}),
            use_container_width=True,
        )
    st.dataframe(df.head(head_rows), use_container_width=True, height=380)


def render_basic_numeric_chart(df: pd.DataFrame, metric_col: Optional[str], chart_type: str):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        st.info("Không có cột numeric để vẽ biểu đồ.")
        return

    if metric_col not in num_cols:
        metric_col = num_cols[0]

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


def render_low_performance_section(
    work: pd.DataFrame,
    plant_col: str,
    label_col: str,
    perf_col: str,
    date_col: Optional[str],
    bottom_n: int,
):
    st.markdown("## Dashboard: label có Performance thấp theo từng Plant")

    if work.empty:
        st.warning("Không có dữ liệu sau khi lọc (ANALYSIS).")
        return

    low = compute_bottom_labels_per_plant(
        work, plant_col=plant_col, label_col=label_col, perf_col=perf_col, date_col=date_col, bottom_n=bottom_n
    )

    st.markdown("### Bảng label Performance thấp")
    st.dataframe(low, use_container_width=True)

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

    # Optional drill-down time series
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


# =========================
# 5) Sidebar layout: 2 tabs (MERGE / ANALYSIS)
# =========================
st.sidebar.title("🧭 Điều khiển")
tab_merge, tab_analysis = st.sidebar.tabs(["MERGE", "ANALYSIS"])

# Session state for dataframes
if "df_a" not in st.session_state:
    st.session_state.df_a = None
if "df_b" not in st.session_state:
    st.session_state.df_b = None
if "merged" not in st.session_state:
    st.session_state.merged = None
if "merge_stats" not in st.session_state:
    st.session_state.merge_stats = None
if "merge_warns" not in st.session_state:
    st.session_state.merge_warns = []


# ---- MERGE sidebar (Upload + merge settings) ----
with tab_merge:
    st.subheader("Upload & Merge")

    file_a = st.file_uploader("File A (DATA) - CSV/XLSX", type=["csv", "xlsx", "xls"], key="file_a")
    file_b = st.file_uploader("File B (CONFIG) - CSV/XLSX", type=["csv", "xlsx", "xls"], key="file_b")

    st.markdown("---")
    st.caption("Merge Settings")
    key_a = st.text_input("Key column File A", value="label", key="key_a")
    key_b = st.text_input("Key column File B", value="label", key="key_b")
    how = st.selectbox("Merge type", options=["left", "inner"], index=0, key="how")
    drop_dupe_cfg = st.checkbox("CONFIG: drop duplicate key (keep first)", value=True, key="drop_dupe_cfg")

    st.markdown("---")
    st.caption("Pre-processing (optional)")
    drop_na = st.checkbox("Drop rows with NA", value=False, key="drop_na")
    dedup = st.checkbox("Drop duplicated rows", value=False, key="dedup")

    # Sheet selection (Excel only) - shown after upload
    sheet_a = None
    sheet_b = None
    if file_a is not None:
        df_a_tmp, sheets_a = load_table(file_a.name, file_a.getvalue(), sheet_name=None)
        if sheets_a:
            sheet_a = st.selectbox("Sheet File A", options=sheets_a, index=0, key="sheet_a")
    if file_b is not None:
        df_b_tmp, sheets_b = load_table(file_b.name, file_b.getvalue(), sheet_name=None)
        if sheets_b:
            sheet_b = st.selectbox("Sheet File B", options=sheets_b, index=0, key="sheet_b")

    st.markdown("---")
    run_merge = st.button("✅ Run Merge", type="primary", use_container_width=True)

    # Execute merge when button pressed
    if run_merge:
        if file_a is None or file_b is None:
            st.error("Bạn cần upload cả File A và File B.")
        else:
            # Read again with selected sheets
            df_a, _ = load_table(file_a.name, file_a.getvalue(), sheet_name=sheet_a)
            df_b, _ = load_table(file_b.name, file_b.getvalue(), sheet_name=sheet_b)

            df_a = clean_columns(df_a)
            df_b = clean_columns(df_b)

            if drop_na:
                df_a = df_a.dropna()
                df_b = df_b.dropna()
            if dedup:
                df_a = df_a.drop_duplicates()
                df_b = df_b.drop_duplicates()

            # Validate keys
            if key_a not in df_a.columns:
                st.error(f"File A không có cột '{key_a}'.")
            elif key_b not in df_b.columns:
                st.error(f"File B không có cột '{key_b}'.")
            else:
                merged, stats, warns = merge_on_key(
                    df_a, df_b, key_a=key_a, key_b=key_b, how=how, drop_dupe_config=drop_dupe_cfg
                )
                st.session_state.df_a = df_a
                st.session_state.df_b = df_b
                st.session_state.merged = merged
                st.session_state.merge_stats = stats
                st.session_state.merge_warns = warns
                st.success("Merge thành công! Chuyển qua tab ANALYSIS để cấu hình phân tích.")


# ---- ANALYSIS sidebar (only analysis settings) ----
analysis_settings: Dict[str, Any] = {}
with tab_analysis:
    st.subheader("Analysis Settings")

    merged = st.session_state.merged
    if merged is None or not isinstance(merged, pd.DataFrame) or merged.empty:
        st.info("Chưa có dữ liệu merged. Hãy qua tab MERGE và bấm **Run Merge** trước.")
    else:
        # Identify key columns for dashboard
        label_col = find_col(merged, "label", aliases=["lable", "Label", "LABEL"])
        plant_col = find_col(merged, "Plant", aliases=["plant", "PLANT"])
        perf_col = find_col(merged, "Performance", aliases=["performance", "PERFORMANCE", "perf"])
        date_col = find_col(merged, "date", aliases=["Date", "DATE", "datetime", "time"])

        inverter_col = find_col(merged, "Inverter", aliases=["inverter"])
        capacity_col = find_col(merged, "Capacity", aliases=["capacity"])
        az_col = find_col(merged, "String Azimuth", aliases=["string azimuth", "azimuth"])

        # Basic chart settings
        st.caption("Biểu đồ thống kê (numeric)")
        num_cols = merged.select_dtypes(include=["number"]).columns.tolist()
        metric_col = st.selectbox("Numeric column", options=(num_cols if num_cols else ["(none)"]), index=0, key="metric_col")
        chart_type = st.selectbox("Chart type", options=["Histogram", "Boxplot"], index=0, key="chart_type")

        st.markdown("---")
        st.caption("Low-performance dashboard filters")

        # Plant selector
        plant_sel: List[str] = []
        if plant_col:
            plants = sorted(merged[plant_col].dropna().astype(str).unique().tolist())
            plant_sel = st.multiselect("Plant", options=plants, default=plants[:1] if plants else [], key="plant_sel")

        bottom_n = st.number_input("Bottom N labels / Plant", min_value=3, max_value=50, value=10, step=1, key="bottom_n")

        # Inverter / Azimuth
        inv_sel: List[str] = []
        if inverter_col:
            inv_vals = sorted(merged[inverter_col].dropna().astype(str).unique().tolist())
            inv_sel = st.multiselect("Inverter", options=inv_vals, default=inv_vals, key="inv_sel")

        az_sel: List[str] = []
        if az_col:
            az_vals = sorted(merged[az_col].dropna().astype(str).unique().tolist())
            az_sel = st.multiselect("String Azimuth", options=az_vals, default=az_vals, key="az_sel")

        # Date range
        date_range = None
        if date_col:
            tmpd = pd.to_datetime(merged[date_col], errors="coerce", infer_datetime_format=True)
            if tmpd.notna().any():
                dmin = tmpd.min().date()
                dmax = tmpd.max().date()
                date_range = st.date_input("Khoảng ngày", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="date_range")

        # Capacity filter
        cap_range = None
        cap_set = None
        if capacity_col:
            if pd.api.types.is_numeric_dtype(merged[capacity_col]):
                cmin = float(np.nanmin(merged[capacity_col].values))
                cmax = float(np.nanmax(merged[capacity_col].values))
                cap_range = st.slider("Capacity (range)", min_value=cmin, max_value=cmax, value=(cmin, cmax), key="cap_range")
            else:
                cap_vals = sorted(merged[capacity_col].dropna().astype(str).unique().tolist())
                cap_set = st.multiselect("Capacity", options=cap_vals, default=cap_vals, key="cap_set")

        # Store settings
        analysis_settings = {
            "label_col": label_col,
            "plant_col": plant_col,
            "perf_col": perf_col,
            "date_col": date_col,
            "inverter_col": inverter_col,
            "capacity_col": capacity_col,
            "az_col": az_col,
            "metric_col": metric_col if metric_col != "(none)" else None,
            "chart_type": chart_type,
            "plant_sel": plant_sel,
            "bottom_n": int(bottom_n),
            "inv_sel": inv_sel,
            "az_sel": az_sel,
            "cap_range": cap_range,
            "cap_set": cap_set,
            "date_range": date_range,
        }


# =========================
# 6) Main Page
# =========================
st.title("🔗 Streamlit Merge Tool + Performance Dashboard")
st.caption("Sidebar tách làm 2 phần: **MERGE** (upload/merge) và **ANALYSIS** (lọc + dashboard).")

merged = st.session_state.merged
df_a = st.session_state.df_a
df_b = st.session_state.df_b

# If user hasn't run merge yet
if merged is None:
    st.info("Hãy vào tab **MERGE** (sidebar) → upload 2 file → bấm **Run Merge**.")
    st.stop()

# --- PREVIEW area ---
st.markdown("## Preview (File A vs File B)")
if df_a is not None and df_b is not None:
    c1, c2 = st.columns(2)
    with c1:
        render_preview("File A (DATA)", df_a)
    with c2:
        render_preview("File B (CONFIG)", df_b)
else:
    st.info("Preview sẽ xuất hiện sau khi merge xong.")


# --- MERGE RESULT area ---
st.markdown("---")
st.markdown("## ✅ Merged Result")

stats = st.session_state.merge_stats
warns = st.session_state.merge_warns or []

if warns:
    for w in warns:
        st.warning(w)

if stats is not None:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows A", f"{stats.rows_a:,}")
    k2.metric("Rows B", f"{stats.rows_b:,}")
    k3.metric("Rows merged", f"{stats.rows_merged:,}")
    k4.metric("Matched", f"{stats.matched_rows:,}")
    k5.metric("Unmatched", f"{stats.unmatched_rows:,}")

    if stats.unmatched_rows > 0:
        st.info("Unmatched = label trong File A không tìm thấy trong File B. Các cột config sẽ trống.")

st.dataframe(merged.head(300), use_container_width=True, height=450)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "⬇️ Download CSV",
        data=download_csv_bytes(merged),
        file_name="merged_result.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "⬇️ Download Excel",
        data=download_xlsx_bytes(merged),
        file_name="merged_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# --- ANALYSIS area (uses settings from ANALYSIS tab) ---
st.markdown("---")
st.markdown("## 📊 Analysis")

if not analysis_settings:
    st.info("Hãy qua tab **ANALYSIS** (sidebar) để cấu hình biểu đồ và bộ lọc.")
else:
    # 1) Basic numeric chart
    st.markdown("### 1) Biểu đồ thống kê cơ bản")
    render_basic_numeric_chart(
        merged,
        metric_col=analysis_settings.get("metric_col"),
        chart_type=analysis_settings.get("chart_type", "Histogram"),
    )

    # 2) Low-performance dashboard
    st.markdown("---")
    st.markdown("### 2) Dashboard: Low-performance labels")

    label_col = analysis_settings.get("label_col")
    plant_col = analysis_settings.get("plant_col")
    perf_col = analysis_settings.get("perf_col")
    date_col = analysis_settings.get("date_col")

    if not (label_col and plant_col and perf_col):
        st.info("Thiếu cột Plant/label/Performance trong dữ liệu merge. Không thể chạy dashboard low-performance.")
    else:
        filtered = apply_analysis_filters(
            merged,
            plant_col=plant_col,
            label_col=label_col,
            perf_col=perf_col,
            date_col=date_col,
            inverter_col=analysis_settings.get("inverter_col"),
            capacity_col=analysis_settings.get("capacity_col"),
            az_col=analysis_settings.get("az_col"),
            plant_sel=analysis_settings.get("plant_sel", []),
            inv_sel=analysis_settings.get("inv_sel", []),
            az_sel=analysis_settings.get("az_sel", []),
            cap_range=analysis_settings.get("cap_range"),
            cap_set=analysis_settings.get("cap_set"),
            date_range=analysis_settings.get("date_range"),
        )

        render_low_performance_section(
            filtered,
            plant_col=plant_col,
            label_col=label_col,
            perf_col=perf_col,
            date_col=date_col,
            bottom_n=int(analysis_settings.get("bottom_n", 10)),
        )
