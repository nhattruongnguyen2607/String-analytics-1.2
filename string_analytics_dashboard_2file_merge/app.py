# app.py
# Streamlit dashboard - supports uploading TWO files:
# 1) String config CSV (configuration)
# 2) Data CSV (e.g., 202510.csv)
# The app merges them by the column "label" and then provides tabs:
# Overview, Data View, Time Analysis, Attr Analysis
#
# Run: streamlit run app.py

import io
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="String Analytics Dashboard", layout="wide")


# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(show_spinner=False)
def load_sample_data(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2024-01-01")
    dates = start + pd.to_timedelta(rng.integers(0, 180, size=rows), unit="D")
    categories = rng.choice(["A", "B", "C", "D"], size=rows, replace=True)
    regions = rng.choice(["North", "South", "East", "West"], size=rows, replace=True)
    value1 = rng.normal(100, 20, size=rows)
    value2 = rng.normal(50, 10, size=rows)
    score = (value1 * 0.4 + value2 * 0.6) + rng.normal(0, 5, size=rows)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "name": rng.choice([f"S{i:03d}" for i in range(1, 60)], size=rows),
            "label": rng.choice([f"L{i:03d}" for i in range(1, 60)], size=rows),
            "Performance": np.round(score / 100, 3),
            "Capacity": np.round(rng.normal(6.5, 0.7, size=rows), 2),
            "String Tilt": rng.choice([10, 12, 15, 20], size=rows),
            "String Azimuth": rng.choice([0, 90, 180, 270], size=rows),
            "Plant": rng.choice(["PR001", "PR002"], size=rows),
            "Roof": rng.choice(["1 Roof", "2 Roof"], size=rows),
            "Inverter": rng.choice(["I01", "I02", "I03"], size=rows),
        }
    )


def _try_read_csv(file_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1258", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(file_bytes))


def _try_read_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))


def load_uploaded_file(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None
    file_bytes = uploaded.getvalue()
    name = (uploaded.name or "").lower()
    try:
        if name.endswith(".csv"):
            return _try_read_csv(file_bytes)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return _try_read_excel(file_bytes)
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(file_bytes))
        return _try_read_csv(file_bytes)
    except Exception as e:
        st.error(f"Không đọc được file: {e}")
        return None


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    other_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    return numeric_cols, datetime_cols, other_cols


def try_parse_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s, errors="coerce")
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def kpi_card(label: str, value, help_text: Optional[str] = None):
    with st.container(border=True):
        st.caption(label)
        st.subheader(value)
        if help_text:
            st.write(help_text)


def normalize_key(series: pd.Series) -> pd.Series:
    # Force string, strip whitespace, keep original casing (labels are usually case-sensitive)
    return series.astype(str).str.strip()


def merge_by_label(data_df: pd.DataFrame, config_df: pd.DataFrame, key: str = "label") -> Tuple[pd.DataFrame, dict]:
    """
    Merge 2 dataframes on the column 'label' (left join from data -> config).
    Returns (merged_df, stats_dict)
    """
    # Some users may type 'lable' by mistake; handle it gracefully.
    key_candidates = [key, "lable", "Label", "LABEL"]

    def find_key(df: pd.DataFrame) -> Optional[str]:
        for k in key_candidates:
            if k in df.columns:
                return k
        return None

    data_key = find_key(data_df)
    cfg_key = find_key(config_df)

    if data_key is None:
        raise ValueError(f"File data không có cột '{key}'. Các cột hiện có: {list(data_df.columns)}")
    if cfg_key is None:
        raise ValueError(f"File config không có cột '{key}'. Các cột hiện có: {list(config_df.columns)}")

    d = data_df.copy()
    c = config_df.copy()

    d[data_key] = normalize_key(d[data_key])
    c[cfg_key] = normalize_key(c[cfg_key])

    merged = d.merge(c, left_on=data_key, right_on=cfg_key, how="left", suffixes=("", "_cfg"))

    # If columns differ (Label vs label), ensure output uses 'label'
    if data_key != "label":
        merged = merged.rename(columns={data_key: "label"})
    if cfg_key != "label":
        merged = merged.drop(columns=[cfg_key], errors="ignore")

    # Basic stats
    # Check match based on a config column like 'Capacity' or any other column from config
    check_col = "Capacity" if "Capacity" in merged.columns else (c.columns[0] if len(c.columns) > 0 else None)
    
    if check_col:
         matched = merged[check_col].notna().sum()
    else:
         # Fallback if config has no other columns
         matched = len(merged) 

    total = len(merged)
    unmatched = total - matched

    stats = {
        "rows_data": len(d),
        "rows_config": len(c),
        "rows_merged": total,
        "matched_rows": int(matched),
        "unmatched_rows": int(unmatched),
        "unique_labels_data": int(d["label"].nunique() if "label" in d.columns else d[data_key].nunique()),
        "unique_labels_config": int(c[cfg_key].nunique()),
    }
    return merged, stats


# ---------------------------
# Sidebar: uploads & settings
# ---------------------------

st.sidebar.title("📥 Import dữ liệu")

data_file = st.sidebar.file_uploader("1) Upload file DATA (vd: 202510.csv)", type=["csv", "xlsx", "xls", "parquet"], key="data")
config_file = st.sidebar.file_uploader("2) Upload file STRING CONFIG (CSV)", type=["csv", "xlsx", "xls", "parquet"], key="config")

use_sample = st.sidebar.toggle("Dùng dữ liệu mẫu", value=(data_file is None and config_file is None))

if use_sample:
    df = load_sample_data()
    merge_stats = None
else:
    data_df = load_uploaded_file(data_file)
    config_df = load_uploaded_file(config_file)

    if data_df is None or config_df is None:
        st.info("Hãy upload **cả 2 file** (DATA và STRING CONFIG) hoặc bật 'Dùng dữ liệu mẫu'.")
        st.stop()

    try:
        df, merge_stats = merge_by_label(data_df, config_df, key="label")
    except Exception as e:
        st.error(f"Lỗi gộp 2 file theo cột label: {e}")
        st.stop()

# Basic cleaning options
st.sidebar.markdown("---")
st.sidebar.subheader("Tiền xử lý (tuỳ chọn)")
drop_na = st.sidebar.checkbox("Bỏ dòng có NA (thiếu dữ liệu)", value=False)
dedup = st.sidebar.checkbox("Xoá dòng trùng lặp", value=False)

if drop_na:
    df = df.dropna()
if dedup:
    df = df.drop_duplicates()

# Try to parse datetime columns from object columns (optional)
st.sidebar.markdown("---")
auto_parse_dt = st.sidebar.checkbox("Tự động nhận dạng cột ngày/giờ", value=True)
if auto_parse_dt:
    for c in df.columns:
        if df[c].dtype == "object":
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() >= 0.8 and parsed.nunique(dropna=True) > 1:
                df[c] = parsed

# If common column 'date' exists but is still object, try parse it strongly
if "date" in df.columns and df["date"].dtype == "object":
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)

numeric_cols, datetime_cols, other_cols = get_column_types(df)

st.title("📊 String Analytics Dashboard")
st.caption("Import 2 file (String config + Data) và gộp theo cột **label**. File kết quả có đầy đủ cột từ cả hai file.")

tabs = st.tabs(["Overview", "Data View", "Time Analysis", "Attr Analysis"])


# ---------------------------
# Tab: Overview
# ---------------------------
with tabs[0]:
    st.subheader("Tổng quan")

    if merge_stats is not None:
        st.markdown("#### Thống kê gộp (merge)")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            kpi_card("Rows (DATA)", f"{merge_stats['rows_data']:,}")
        with m2:
            kpi_card("Rows (CONFIG)", f"{merge_stats['rows_config']:,}")
        with m3:
            kpi_card("Rows (MERGED)", f"{merge_stats['rows_merged']:,}")
        with m4:
            kpi_card("Matched rows", f"{merge_stats['matched_rows']:,}")
        with m5:
            kpi_card("Unmatched rows", f"{merge_stats['unmatched_rows']:,}")

        if merge_stats["unmatched_rows"] > 0:
            st.warning("Có một số label trong file DATA không tìm thấy trong STRING CONFIG (merge sẽ để trống các cột config ở các dòng đó).")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Số dòng", f"{len(df):,}")
    with c2:
        kpi_card("Số cột", f"{df.shape[1]:,}")
    with c3:
        kpi_card("Cột số (numeric)", f"{len(numeric_cols)}")
    with c4:
        kpi_card("Cột thời gian (datetime)", f"{len(datetime_cols)}")

    st.markdown("### Danh sách cột (kết quả sau gộp)")
    st.dataframe(
        pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}),
        use_container_width=True,
    )

    st.markdown("### Xem nhanh (20 dòng đầu)")
    st.dataframe(df.head(20), use_container_width=True)


# ---------------------------
# Tab: Data View
# ---------------------------
with tabs[1]:
    st.subheader("Xem dữ liệu & lọc")

    col_a, col_b, col_c = st.columns([2, 2, 1])

    with col_a:
        search = st.text_input("Tìm kiếm (áp dụng cho toàn bộ dữ liệu, dạng text)", value="")
    with col_b:
        show_cols = st.multiselect(
            "Chọn cột hiển thị",
            options=df.columns.tolist(),
            default=df.columns.tolist()[: min(10, df.shape[1])],
        )
    with col_c:
        nrows = st.number_input("Số dòng hiển thị", min_value=10, max_value=5000, value=200, step=10)

    view_df = df.copy()

    if search.strip():
        s = search.strip().lower()
        mask = pd.Series(False, index=view_df.index)
        for c in view_df.columns:
            mask = mask | view_df[c].astype(str).str.lower().str.contains(s, na=False)
        view_df = view_df.loc[mask]

    if show_cols:
        view_df = view_df[show_cols]

    st.dataframe(view_df.head(int(nrows)), use_container_width=True)

    st.markdown("### Tải xuống dữ liệu đã lọc")
    csv_bytes = view_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="merged_filtered_data.csv", mime="text/csv")


# ---------------------------
# Tab: Time Analysis (Đã cập nhật theo yêu cầu)
# ---------------------------
with tabs[2]:
    st.subheader("Phân tích chi tiết hiệu suất String theo thời gian")

    # 1. Kiểm tra cột Date
    if not datetime_cols:
        st.error("Không tìm thấy cột thời gian (datetime). Vui lòng kiểm tra lại file input.")
    else:
        # Chọn cột ngày (thường là 'date')
        date_col = st.selectbox("Cột thời gian", options=datetime_cols, index=0, label_visibility="collapsed")
        
        # Đảm bảo dữ liệu được sort theo ngày
        df_time = df.sort_values(by=date_col).copy()
        
        # ---------------------------
        # A. BỘ LỌC (FILTERS)
        # ---------------------------
        st.markdown("##### 1. Bộ lọc dữ liệu")
        
        # A1. Lọc theo khoảng thời gian
        # Handle cases where datetime column might have NaT (Not a Time)
        df_time = df_time.dropna(subset=[date_col])
        if df_time.empty:
             st.warning("Dữ liệu trống sau khi lọc ngày.")
        else:
            min_date = df_time[date_col].min().date()
            max_date = df_time[date_col].max().date()
            
            c_date1, c_date2 = st.columns(2)
            with c_date1:
                start_date = st.date_input("Từ ngày", value=min_date, min_value=min_date, max_value=max_date)
            with c_date2:
                end_date = st.date_input("Đến ngày", value=max_date, min_value=min_date, max_value=max_date)
                
            # A2. Các bộ lọc thuộc tính (Inverter, Capacity, Azimuth)
            # Tạo danh sách unique values để đưa vào selectbox
            # Lưu ý: Convert sang string để hiển thị cho đẹp và tránh lỗi sort
            
            all_inverters = sorted(df_time['Inverter'].dropna().astype(str).unique()) if 'Inverter' in df_time.columns else []
            all_capacities = sorted(df_time['Capacity'].dropna().unique()) if 'Capacity' in df_time.columns else []
            all_azimuths = sorted(df_time['String Azimuth'].dropna().unique()) if 'String Azimuth' in df_time.columns else []

            f1, f2, f3 = st.columns(3)
            with f1:
                sel_inv = st.multiselect("Lọc Inverter", options=all_inverters)
            with f2:
                sel_cap = st.multiselect("Lọc Capacity", options=all_capacities)
            with f3:
                sel_azi = st.multiselect("Lọc Azimuth", options=all_azimuths)

            # ---------------------------
            # B. XỬ LÝ DATA THEO BỘ LỌC
            # ---------------------------
            
            # 1. Lọc Time
            mask = (df_time[date_col].dt.date >= start_date) & (df_time[date_col].dt.date <= end_date)
            filtered_df = df_time.loc[mask]

            # 2. Lọc Attributes (Nếu user không chọn gì thì lấy tất cả)
            if sel_inv:
                filtered_df = filtered_df[filtered_df['Inverter'].astype(str).isin(sel_inv)]
            
            if sel_cap:
                filtered_df = filtered_df[filtered_df['Capacity'].isin(sel_cap)]
                
            if sel_azi:
                filtered_df = filtered_df[filtered_df['String Azimuth'].isin(sel_azi)]

            # ---------------------------
            # C. VẼ BIỂU ĐỒ (LINE CHART)
            # ---------------------------
            st.markdown("##### 2. Biểu đồ biến thiên")
            
            if filtered_df.empty:
                st.warning("Không có dữ liệu phù hợp với bộ lọc.")
            else:
                # Chọn chỉ số để vẽ (Mặc định là Performance)
                metric_col = "Performance"
                if "Performance" not in numeric_cols and len(numeric_cols) > 0:
                    metric_col = st.selectbox("Chọn chỉ số hiển thị trục Y", numeric_cols)
                
                # Cảnh báo nếu quá nhiều dây (String) được chọn -> Vẽ sẽ bị rối
                num_strings = filtered_df['label'].nunique()
                st.caption(f"Đang hiển thị **{num_strings}** string(s) trên biểu đồ.")
                
                if num_strings > 50:
                    st.warning(f"⚠️ Bạn đang chọn quá nhiều String ({num_strings}). Biểu đồ có thể bị chậm hoặc khó nhìn. Hãy lọc bớt Inverter hoặc Capacity.")

                # Vẽ biểu đồ Line
                # x: Ngày, y: Performance, color: label (tên string)
                fig = px.line(
                    filtered_df, 
                    x=date_col, 
                    y=metric_col, 
                    color='label',
                    markers=True, # Hiển thị điểm tròn trên dòng để dễ nhìn dữ liệu ngày
                    hover_data=['Inverter', 'Capacity', 'String Azimuth'], # Khi rê chuột vào sẽ hiện thêm thông tin này
                    title=f"Biến thiên {metric_col} theo ngày (Day-by-Day)"
                )
                
                # Tinh chỉnh biểu đồ cho đẹp
                fig.update_layout(
                    xaxis_title="Ngày",
                    yaxis_title=metric_col,
                    legend_title="String Label",
                    hovermode="x unified" # Hiển thị so sánh tất cả các line tại 1 điểm ngày
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Hiện bảng dữ liệu thô bên dưới nếu cần soi chi tiết
                with st.expander("Xem dữ liệu chi tiết của biểu đồ"):
                    display_cols = [date_col, 'label', metric_col]
                    extra_cols = ['Inverter', 'Capacity', 'String Azimuth']
                    for c in extra_cols:
                        if c in filtered_df.columns:
                            display_cols.append(c)
                            
                    st.dataframe(filtered_df[display_cols], use_container_width=True)


# ---------------------------
# Tab: Attr Analysis
# ---------------------------
with tabs[3]:
    st.subheader("Phân tích thuộc tính (Attr)")

    left, right = st.columns([1, 2])

    with left:
        # Favor a categorical column if present
        cat_candidates = [c for c in df.columns if df[c].dtype == "object"] + \
                         [c for c in df.columns if str(df[c].dtype).startswith("category")] + \
                         [c for c in df.columns if df[c].dtype == "bool"]
        default_cat = "Plant" if "Plant" in df.columns else (cat_candidates[0] if cat_candidates else df.columns[0])
        cat_col = st.selectbox("Cột phân loại (categorical)", options=df.columns.tolist(), index=df.columns.get_loc(default_cat) if default_cat in df.columns else 0)

        num_col = None
        if numeric_cols:
            default_num = "Performance" if "Performance" in numeric_cols else numeric_cols[0]
            num_col = st.selectbox("Cột số (numeric)
