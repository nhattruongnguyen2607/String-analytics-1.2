with tabs[2]:
    st.subheader("Phân tích theo thời gian (day-by-day theo từng label)")

    required_cols = ["date", "label", "Performance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Thiếu cột bắt buộc: {missing}. Cần có: date, label, Performance")
    else:
        # parse date mạnh hơn nếu cần
        if "date" in df.columns and df["date"].dtype == "object":
            df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)

        if df["date"].isna().all():
            st.error("Cột 'date' không parse được (toàn NA). Hãy kiểm tra format ngày.")
        else:
            left, right = st.columns([1, 2])

            tmp0 = df.dropna(subset=["date"]).copy()
            tmp0["label"] = tmp0["label"].astype(str)

            min_date = tmp0["date"].min().date()
            max_date = tmp0["date"].max().date()

            with left:
                st.markdown("#### Bộ lọc")

                # 1) lọc ngày
                date_range = st.date_input(
                    "Chọn khoảng ngày phân tích",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date, end_date = min_date, max_date

                # 2) lọc inverter
                inverter_vals = sorted(tmp0["Inverter"].dropna().astype(str).unique().tolist()) if "Inverter" in tmp0.columns else []
                inv_sel = st.multiselect("Inverter", options=inverter_vals, default=inverter_vals[:]) if inverter_vals else []

                # 3) lọc string azimuth
                az_vals = sorted(tmp0["String Azimuth"].dropna().astype(str).unique().tolist()) if "String Azimuth" in tmp0.columns else []
                az_sel = st.multiselect("String Azimuth", options=az_vals, default=az_vals[:]) if az_vals else []

                # 4) lọc capacity (range nếu là numeric, nếu không thì multiselect)
                cap_mode, cap_min, cap_max, cap_sel = None, None, None, None
                if "Capacity" in tmp0.columns:
                    if pd.api.types.is_numeric_dtype(tmp0["Capacity"]):
                        cap_mode = "range"
                        cap_min0 = float(np.nanmin(tmp0["Capacity"].values))
                        cap_max0 = float(np.nanmax(tmp0["Capacity"].values))
                        cap_min, cap_max = st.slider(
                            "Capacity (range)",
                            min_value=cap_min0,
                            max_value=cap_max0,
                            value=(cap_min0, cap_max0),
                        )
                    else:
                        cap_mode = "set"
                        cap_vals = sorted(tmp0["Capacity"].dropna().astype(str).unique().tolist())
                        cap_sel = st.multiselect("Capacity", options=cap_vals, default=cap_vals[:])

                # 5) chọn label để vẽ (đỡ quá nhiều line)
                labels = sorted(tmp0["label"].unique().tolist())
                default_labels = labels[: min(10, len(labels))]
                label_sel = st.multiselect("Label", options=labels, default=default_labels)

                agg = st.selectbox("Phép tổng hợp theo ngày", options=["mean", "median", "min", "max"], index=0)
                show_table = st.checkbox("Hiện bảng sau filter + group", value=False)

            # --- apply filters ---
            tmp = tmp0[(tmp0["date"].dt.date >= start_date) & (tmp0["date"].dt.date <= end_date)].copy()

            if inverter_vals and inv_sel:
                tmp = tmp[tmp["Inverter"].astype(str).isin(inv_sel)]
            if az_vals and az_sel:
                tmp = tmp[tmp["String Azimuth"].astype(str).isin(az_sel)]

            if "Capacity" in tmp.columns:
                if cap_mode == "range" and cap_min is not None and cap_max is not None:
                    tmp = tmp[(tmp["Capacity"] >= cap_min) & (tmp["Capacity"] <= cap_max)]
                elif cap_mode == "set" and cap_sel is not None:
                    tmp = tmp[tmp["Capacity"].astype(str).isin(cap_sel)]

            if label_sel:
                tmp = tmp[tmp["label"].astype(str).isin(label_sel)]

            if tmp.empty:
                st.warning("Không có dữ liệu sau khi lọc.")
            else:
                # day-by-day theo label
                tmp["_day"] = tmp["date"].dt.to_period("D").dt.to_timestamp()
                ts = tmp.groupby(["_day", "label"])["Performance"].agg(agg).reset_index()

                with right:
                    fig = px.line(
                        ts,
                        x="_day",
                        y="Performance",
                        color="label",
                        markers=True,
                        title=f"Performance ({agg}) theo ngày - mỗi label là 1 đường",
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="label")
                    st.plotly_chart(fig, use_container_width=True)

                if show_table:
                    st.dataframe(ts.sort_values(["_day", "label"]), use_container_width=True)
