import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Solar Data Engineering Tool",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_data(file):
    """
    Hàm đọc dữ liệu thông minh, hỗ trợ cả CSV và Excel.
    Sử dụng cache để không phải load lại file mỗi khi tương tác UI.
    """
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            return None
    except Exception as e:
        st.error(f"Lỗi khi đọc file {file.name}: {e}")
        return None

def convert_df_to_csv(df):
    """
    Chuyển DataFrame thành CSV để download.
    """
    return df.to_csv(index=False).encode('utf-8')

# ==========================================
# SIDEBAR: UPLOAD & SETTINGS
# ==========================================
st.sidebar.header("📂 Data Input")
st.sidebar.info("Upload dữ liệu để bắt đầu quy trình ETL.")

# Bước 1: Upload Files
uploaded_file_a = st.sidebar.file_uploader("1. Upload File Data (Performance)", type=['csv', 'xlsx'])
uploaded_file_b = st.sidebar.file_uploader("2. Upload File Config (Plant Info)", type=['csv', 'xlsx'])

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("📊 Solar Data Merge & Analytics Platform")
st.markdown("---")

if uploaded_file_a and uploaded_file_b:
    # --- Load Data ---
    df_a = load_data(uploaded_file_a)
    df_b = load_data(uploaded_file_b)

    if df_a is not None and df_b is not None:
        
        # --- Bước 2: Preview Data (Song song) ---
        st.subheader("1. Data Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"File A: {uploaded_file_a.name}")
            st.write(f"Shape: {df_a.shape}")
            st.dataframe(df_a.head(), use_container_width=True)
            
        with col2:
            st.info(f"File B: {uploaded_file_b.name}")
            st.write(f"Shape: {df_b.shape}")
            st.dataframe(df_b.head(), use_container_width=True)

        st.markdown("---")

        # --- Bước 3: Merge Settings ---
        st.sidebar.header("⚙️ Merge Configuration")
        
        # Tự động phát hiện cột chung, ưu tiên 'label' theo yêu cầu
        common_cols = list(set(df_a.columns) & set(df_b.columns))
        default_idx_a = df_a.columns.get_loc('label') if 'label' in df_a.columns else 0
        default_idx_b = df_b.columns.get_loc('label') if 'label' in df_b.columns else 0

        key_col_a = st.sidebar.selectbox("Key Column (File A)", df_a.columns, index=default_idx_a)
        key_col_b = st.sidebar.selectbox("Key Column (File B)", df_b.columns, index=default_idx_b)
        
        merge_mode = st.sidebar.radio("Kiểu gộp (Merge Type)", ["inner", "left", "right", "outer"], index=0)

        # --- Bước 4: Process Merge ---
        if st.sidebar.button("🚀 Thực hiện Gộp Dữ liệu (Merge)", type="primary"):
            try:
                # Merge Data
                merged_df = pd.merge(
                    df_a, 
                    df_b, 
                    left_on=key_col_a, 
                    right_on=key_col_b, 
                    how=merge_mode
                )
                
                # Lưu vào session state để dùng cho các bước sau mà không cần merge lại
                st.session_state['merged_df'] = merged_df
                st.success("Gộp dữ liệu thành công!")
                
            except Exception as e:
                st.error(f"Lỗi khi gộp dữ liệu: {e}")

        # --- Hiển thị kết quả sau khi Merge (Nếu có trong session state) ---
        if 'merged_df' in st.session_state:
            merged_df = st.session_state['merged_df']

            st.subheader("2. Kết quả Gộp (Merged Data)")
            
            # --- Bước 5: Download & Stats ---
            col_res1, col_res2 = st.columns([3, 1])
            with col_res1:
                st.dataframe(merged_df, use_container_width=True)
            
            with col_res2:
                st.write("**Thống kê nhanh:**")
                st.write(f"Tổng số dòng: `{len(merged_df)}`")
                st.write(f"Tổng số cột: `{len(merged_df.columns)}`")
                
                csv_data = convert_df_to_csv(merged_df)
                st.download_button(
                    label="⬇️ Tải file kết quả (CSV)",
                    data=csv_data,
                    file_name="merged_solar_data.csv",
                    mime="text/csv",
                )

            st.markdown("---")
            
            # ==========================================
            # DASHBOARD: LOW PERFORMANCE ANALYSIS
            # ==========================================
            st.header("📉 Dashboard: Low Performance Analysis")
            
            # Kiểm tra xem các cột cần thiết có tồn tại không
            required_cols = ['Performance', 'Plant'] # Dựa trên file user cung cấp
            if all(col in merged_df.columns for col in required_cols):
                
                # --- Controls cho Dashboard ---
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    threshold = st.slider("Ngưỡng Performance thấp (Threshold)", 0.0, 1.0, 0.80, step=0.01)
                
                # Lọc dữ liệu Low Performance
                low_perf_df = merged_df[merged_df['Performance'] < threshold]
                
                with col_d2:
                    st.metric("Số lượng Label lỗi (Low Perf)", len(low_perf_df))
                with col_d3:
                    avg_perf = low_perf_df['Performance'].mean() if not low_perf_df.empty else 0
                    st.metric("Performance trung bình nhóm lỗi", f"{avg_perf:.2f}")

                # --- Visualization ---
                if not low_perf_df.empty:
                    col_chart1, col_chart2 = st.columns(2)
                    
                    # Chart 1: Số lượng lỗi theo Plant
                    with col_chart1:
                        st.subheader("Phân bố lỗi theo Nhà máy (Plant)")
                        error_by_plant = low_perf_df['Plant'].value_counts().reset_index()
                        error_by_plant.columns = ['Plant', 'Count']
                        
                        fig1 = px.bar(
                            error_by_plant, 
                            x='Plant', 
                            y='Count',
                            color='Count',
                            text='Count',
                            title=f"Số lượng Label < {threshold} theo Plant",
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig1, use_container_width=True)

                    # Chart 2: Boxplot Performance theo Plant (để xem phân tán)
                    with col_chart2:
                        st.subheader("Phân tán Performance nhóm lỗi")
                        fig2 = px.box(
                            low_perf_df, 
                            x='Plant', 
                            y='Performance', 
                            color='Plant',
                            title=f"Boxplot Performance (Dưới ngưỡng {threshold})",
                            points="all" # Hiển thị cả các điểm dữ liệu
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    # --- Detail Table ---
                    st.subheader("Chi tiết các Label có Performance thấp")
                    
                    # Cho phép lọc theo Plant cụ thể
                    plant_list = ["All"] + list(low_perf_df['Plant'].unique())
                    selected_plant = st.selectbox("Lọc theo Plant:", plant_list)
                    
                    if selected_plant != "All":
                        display_df = low_perf_df[low_perf_df['Plant'] == selected_plant]
                    else:
                        display_df = low_perf_df
                    
                    # Hiển thị các cột quan trọng trước
                    cols_order = ['date', 'Plant', 'label', 'Performance', 'Capacity', 'Inverter']
                    # Chỉ lấy các cột tồn tại
                    cols_to_show = [c for c in cols_order if c in display_df.columns]
                    
                    st.dataframe(
                        display_df[cols_to_show].style.format({"Performance": "{:.2f}"}).background_gradient(subset=['Performance'], cmap='Reds_r'),
                        use_container_width=True
                    )
                else:
                    st.success(f"Tuyệt vời! Không có label nào có Performance dưới {threshold}.")
            else:
                st.warning("Dữ liệu sau khi gộp thiếu cột 'Performance' hoặc 'Plant'. Vui lòng kiểm tra lại file đầu vào.")

else:
    # Màn hình chờ khi chưa upload
    st.info("👈 Vui lòng upload File A và File B từ thanh bên (Sidebar) để bắt đầu.")
