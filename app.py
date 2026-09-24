import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ortools.sat.python import cp_model


# =========================================================
# CP-SAT 自动优化
# =========================================================
def solve_three_stage_flow_shop(
    wait_times, process_times, rough_times, time_limit=30, capacity=5
):
    n = len(wait_times)
    assert n == len(process_times) == len(rough_times)

    horizon = sum(rough_times) + sum(wait_times) + sum(process_times)
    model = cp_model.CpModel()

    rs = [model.NewIntVar(0, horizon, f"rs_{i}") for i in range(n)]
    re = [model.NewIntVar(0, horizon, f"re_{i}") for i in range(n)]
    ws = [model.NewIntVar(0, horizon, f"ws_{i}") for i in range(n)]
    we = [model.NewIntVar(0, horizon, f"we_{i}") for i in range(n)]
    ps = [model.NewIntVar(0, horizon, f"ps_{i}") for i in range(n)]
    pe = [model.NewIntVar(0, horizon, f"pe_{i}") for i in range(n)]

    rough_interval = [
        model.NewIntervalVar(rs[i], rough_times[i], re[i], f"rough_{i}")
        for i in range(n)
    ]
    wait_interval = [
        model.NewIntervalVar(ws[i], wait_times[i], we[i], f"wait_{i}")
        for i in range(n)
    ]
    proc_interval = [
        model.NewIntervalVar(ps[i], process_times[i], pe[i], f"proc_{i}")
        for i in range(n)
    ]

    model.AddNoOverlap(rough_interval)
    model.AddCumulative(wait_interval, [1] * n, capacity)
    model.AddNoOverlap(proc_interval)

    for i in range(n):
        model.Add(re[i] == ws[i])                 # 粗轧结束 = 待温开始
        model.Add(we[i] == ws[i] + wait_times[i])  # 待温时间精确
        model.Add(we[i] == ps[i])                  # 无等待

    rank = [model.NewIntVar(0, n - 1, f"rank_{i}") for i in range(n)]
    model.AddAllDifferent(rank)

    for i in range(n):
        for j in range(i + 1, n):
            b = model.NewBoolVar(f"before_{i}_{j}")
            model.Add(rank[i] < rank[j]).OnlyEnforceIf(b)
            model.Add(rank[j] < rank[i]).OnlyEnforceIf(b.Not())
            model.Add(re[i] <= rs[j]).OnlyEnforceIf(b)
            model.Add(re[j] <= rs[i]).OnlyEnforceIf(b.Not())
            model.Add(ws[i] <= ws[j]).OnlyEnforceIf(b)
            model.Add(ws[j] <= ws[i]).OnlyEnforceIf(b.Not())
            model.Add(pe[i] <= ps[j]).OnlyEnforceIf(b)
            model.Add(pe[j] <= ps[i]).OnlyEnforceIf(b.Not())

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, pe)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    order = sorted(range(n), key=lambda i: solver.Value(rank[i]))
    rows = []
    for pos, i in enumerate(order, start=1):
        rows.append({
            "顺序": pos,
            "板坯": f"板坯 {i + 1}",
            "粗轧时间(s)": rough_times[i],
            "待温时间(s)": wait_times[i],
            "精轧时间(s)": process_times[i],
            "粗轧开始(s)": solver.Value(rs[i]),
            "粗轧结束(s)": solver.Value(re[i]),
            "待温开始(s)": solver.Value(ws[i]),
            "待温结束(s)": solver.Value(we[i]),
            "精轧开始(s)": solver.Value(ps[i]),
            "精轧结束(s)": solver.Value(pe[i]),
        })

    df = pd.DataFrame(rows)
    return {
        "status": solver.StatusName(status),
        "makespan": df["精轧结束(s)"].max(),
        "table": df,
    }


# =========================================================
# 固定顺序
# =========================================================
def solve_fixed_order(wait_times, process_times, rough_times, order, capacity=5):
    n = len(wait_times)
    assert len(order) == n
    assert sorted(order) == list(range(n))

    rows = []
    prev_re = 0       # 上一块粗轧结束
    prev_pe = 0       # 上一块精轧结束
    wait_intervals = []

    for pos, i in enumerate(order, start=1):
        r = rough_times[i]
        w = wait_times[i]
        p = process_times[i]

        if pos == 1:
            rs = 0
        else:
            # 粗轧开始必须同时满足：
            # 1) 上一块粗轧结束：rs >= prev_re
            # 2) 待温结束 >= 上一块精轧结束：rs + r + w >= prev_pe
            rs = max(prev_re, prev_pe - w - r)

        re = rs + r
        ws = re
        we = ws + w

        # 容量检查：待温区最多 capacity 块
        while True:
            cnt = 1
            for (s, e) in wait_intervals:
                if s < we and e > ws:
                    cnt += 1
            if cnt <= capacity:
                break
            # 推迟粗轧开始，让待温区间整体后移
            candidates = [
                e for (s, e) in wait_intervals if s < we and e > ws
            ]
            if not candidates:
                break
            shift = min(candidates) - ws
            rs += shift
            re = rs + r
            ws = re
            we = ws + w

        ps = we
        pe = ps + p

        rows.append({
            "顺序": pos,
            "板坯": f"板坯 {i + 1}",
            "粗轧时间(s)": r,
            "待温时间(s)": w,
            "精轧时间(s)": p,
            "粗轧开始(s)": rs,
            "粗轧结束(s)": re,
            "待温开始(s)": ws,
            "待温结束(s)": we,
            "精轧开始(s)": ps,
            "精轧结束(s)": pe,
        })

        prev_re = re
        prev_pe = pe
        wait_intervals.append((ws, we))

    df = pd.DataFrame(rows)
    return {
        "status": "FIXED_ORDER_3STAGE",
        "makespan": df["精轧结束(s)"].max(),
        "table": df,
    }


# =========================================================
# 页面
# =========================================================
st.set_page_config(page_title="粗轧-待温-精轧协同排产", layout="wide")
st.title("粗轧-待温-精轧协同排产")
st.caption("三段流水线：粗轧 → 待温 → 精轧。粗轧结束即待温开始，待温结束即精轧开始。")

st.sidebar.header("参数设置")
n = st.sidebar.number_input("板坯数量", min_value=1, max_value=50, value=9, step=1)
capacity = st.sidebar.number_input("待温区容量", min_value=1, max_value=20, value=5, step=1)
time_limit = st.sidebar.number_input(
    "求解时间上限（秒）", min_value=1, max_value=300, value=30, step=1
)

st.sidebar.markdown("---")
st.sidebar.header("排产模式")
mode = st.sidebar.radio("选择模式", ["自动优化顺序", "固定顺序"], index=0)

fixed_order_str = None
if mode == "固定顺序":
    st.sidebar.markdown("**输入固定顺序**")
    st.sidebar.caption("用逗号分隔，例如：1,2,3,4,5,6,7,8,9")
    default_order_str = ",".join(str(i + 1) for i in range(n))
    fixed_order_str = st.sidebar.text_input("板坯顺序", value=default_order_str)

st.subheader("1. 输入板坯数据")

default_rough = [150] * 9
default_wait = [50, 50, 50, 200, 200, 200, 500, 500, 1000]
default_proc = [120] * 9


def get_default(lst, i, fallback):
    return lst[i] if i < len(lst) else fallback


input_data = []
cols = st.columns([1, 2, 2, 2])
cols[0].markdown("**板坯**")
cols[1].markdown("**粗轧时间 (s)**")
cols[2].markdown("**待温时间 (s)**")
cols[3].markdown("**精轧时间 (s)**")

for i in range(n):
    c0, c1, c2, c3 = st.columns([1, 2, 2, 2])
    c0.write(f"板坯 {i + 1}")
    r = c1.number_input(
        f"粗轧时间_{i}", min_value=1,
        value=int(get_default(default_rough, i, 150)), step=10, key=f"r_{i}"
    )
    w = c2.number_input(
        f"待温时间_{i}", min_value=0,
        value=int(get_default(default_wait, i, 100)), step=10, key=f"w_{i}"
    )
    p = c3.number_input(
        f"精轧时间_{i}", min_value=1,
        value=int(get_default(default_proc, i, 120)), step=10, key=f"p_{i}"
    )
    input_data.append((r, w, p))

st.subheader("2. 求解结果")

if st.button("开始排产"):
    rough_times = [x[0] for x in input_data]
    wait_times = [x[1] for x in input_data]
    process_times = [x[2] for x in input_data]

    if mode == "固定顺序":
        try:
            order_1based = [
                int(x.strip()) for x in fixed_order_str.split(",") if x.strip()
            ]
        except ValueError:
            st.error("顺序格式不对，请用逗号分隔的数字")
            st.stop()

        if len(order_1based) != n:
            st.error(f"顺序长度应为 {n}，当前为 {len(order_1based)}")
            st.stop()

        if sorted(order_1based) != list(range(1, n + 1)):
            st.error("顺序必须是 1 到 n 的一个排列，不能重复或遗漏")
            st.stop()

        order_0based = [x - 1 for x in order_1based]

        with st.spinner("正在按固定顺序计算..."):
            result = solve_fixed_order(
                wait_times, process_times, rough_times, order_0based, capacity=capacity
            )
    else:
        with st.spinner("正在求解最优排产顺序..."):
            result = solve_three_stage_flow_shop(
                wait_times, process_times, rough_times,
                time_limit=time_limit, capacity=capacity
            )

    if result is None:
        st.error("未找到可行解，请检查输入数据或放宽容量限制。")
    else:
        st.success(f"求解状态：{result['status']}")
        col1, col2 = st.columns(2)
        col1.metric("最小总完工时间 makespan", f"{result['makespan']} s")
        col2.metric("板坯数量", n)

        st.markdown("### 排产顺序")
        st.dataframe(result["table"])

        st.markdown("### 粗轧 + 待温 + 精轧 时间线")
        df = result["table"].copy()
        fig = go.Figure()

        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["粗轧时间(s)"]],
                y=[row["板坯"]],
                base=[row["粗轧开始(s)"]],
                orientation="h",
                name="粗轧",
                marker_color="lightgreen",
                text=f"粗轧 {row['粗轧开始(s)']}→{row['粗轧结束(s)']}s",
                textposition="inside",
                hoverinfo="text",
                showlegend=False,
            ))
            fig.add_trace(go.Bar(
                x=[row["待温时间(s)"]],
                y=[row["板坯"]],
                base=[row["待温开始(s)"]],
                orientation="h",
                name="待温",
                marker_color="lightblue",
                text=f"待温 {row['待温开始(s)']}→{row['待温结束(s)']}s",
                textposition="inside",
                hoverinfo="text",
                showlegend=False,
            ))
            fig.add_trace(go.Bar(
                x=[row["精轧时间(s)"]],
                y=[row["板坯"]],
                base=[row["精轧开始(s)"]],
                orientation="h",
                name="精轧",
                marker_color="salmon",
                text=f"精轧 {row['精轧开始(s)']}→{row['精轧结束(s)']}s",
                textposition="inside",
                hoverinfo="text",
                showlegend=False,
            ))

        fig.update_layout(
            barmode="overlay",
            xaxis_title="时间 (s)",
            yaxis_title="板坯",
            height=450,
            margin=dict(l=80, r=40, t=40, b=40),
        )
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=list(df["板坯"])[::-1],
        )

        st.plotly_chart(fig, use_container_width=True)

        csv = result["table"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载排产结果 CSV",
            data=csv,
            file_name="排产结果.csv",
            mime="text/csv",
        )