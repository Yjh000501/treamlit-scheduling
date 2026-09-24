import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ortools.sat.python import cp_model


# =========================================================
# 求解函数 1：自动优化顺序（两阶段流水车间）
# =========================================================
def solve_flow_shop(wait_times, process_times, time_limit=30):
    n = len(wait_times)
    assert n == len(process_times)

    horizon = sum(wait_times) + sum(process_times)
    model = cp_model.CpModel()

    ws = [model.NewIntVar(0, horizon, f"ws_{i}") for i in range(n)]
    we = [model.NewIntVar(0, horizon, f"we_{i}") for i in range(n)]
    ps = [model.NewIntVar(0, horizon, f"ps_{i}") for i in range(n)]
    pe = [model.NewIntVar(0, horizon, f"pe_{i}") for i in range(n)]

    wait_interval = [
        model.NewIntervalVar(ws[i], wait_times[i], we[i], f"wait_{i}")
        for i in range(n)
    ]
    proc_interval = [
        model.NewIntervalVar(ps[i], process_times[i], pe[i], f"proc_{i}")
        for i in range(n)
    ]

    model.AddNoOverlap(wait_interval)
    model.AddNoOverlap(proc_interval)

    for i in range(n):
        model.Add(we[i] <= ps[i])

    rank = [model.NewIntVar(0, n - 1, f"rank_{i}") for i in range(n)]
    model.AddAllDifferent(rank)

    for i in range(n):
        for j in range(i + 1, n):
            b = model.NewBoolVar(f"before_{i}_{j}")
            model.Add(rank[i] < rank[j]).OnlyEnforceIf(b)
            model.Add(rank[j] < rank[i]).OnlyEnforceIf(b.Not())
            model.Add(we[i] <= ws[j]).OnlyEnforceIf(b)
            model.Add(we[j] <= ws[i]).OnlyEnforceIf(b.Not())
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
            "待温时间(s)": wait_times[i],
            "精轧时间(s)": process_times[i],
            "待温开始(s)": solver.Value(ws[i]),
            "待温结束(s)": solver.Value(we[i]),
            "精轧开始(s)": solver.Value(ps[i]),
            "精轧结束(s)": solver.Value(pe[i]),
        })

    return {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "table": pd.DataFrame(rows),
    }


# =========================================================
# 求解函数 2：固定顺序（线性递推）
# =========================================================
def solve_fixed_order_flow_shop(wait_times, process_times, order):
    n = len(wait_times)
    assert len(order) == n
    assert sorted(order) == list(range(n)), "顺序必须包含所有板坯且不重复"

    rows = []
    prev_we = 0
    prev_pe = 0

    for pos, i in enumerate(order, start=1):
        w = wait_times[i]
        p = process_times[i]

        ws = prev_we
        we = ws + w
        ps = max(we, prev_pe)
        pe = ps + p

        rows.append({
            "顺序": pos,
            "板坯": f"板坯 {i + 1}",
            "待温时间(s)": w,
            "精轧时间(s)": p,
            "待温开始(s)": ws,
            "待温结束(s)": we,
            "精轧开始(s)": ps,
            "精轧结束(s)": pe,
        })

        prev_we = we
        prev_pe = pe

    return {
        "status": "FIXED_ORDER",
        "makespan": prev_pe,
        "table": pd.DataFrame(rows),
    }


# =========================================================
# 页面
# =========================================================
st.set_page_config(page_title="板坯待温-精轧协同排产", layout="wide")
st.title("板坯待温-精轧协同排产")
st.caption("待温顺序 = 精轧顺序，两阶段流水车间，最小化总完工时间。")

st.sidebar.header("参数设置")
n = st.sidebar.number_input("板坯数量", min_value=1, max_value=50, value=10, step=1)
time_limit = st.sidebar.number_input(
    "求解时间上限（秒）", min_value=1, max_value=300, value=30, step=1
)

st.sidebar.markdown("---")
st.sidebar.header("排产模式")
mode = st.sidebar.radio("选择模式", ["自动优化顺序", "固定顺序"], index=0)

fixed_order_str = None
if mode == "固定顺序":
    st.sidebar.markdown("**输入固定顺序**")
    st.sidebar.caption("用逗号分隔，例如：3,1,5,2,4,6,7,8,9,10")
    default_order_str = ",".join(str(i + 1) for i in range(n))
    fixed_order_str = st.sidebar.text_input("板坯顺序", value=default_order_str)

st.subheader("1. 输入板坯数据")

default_wait = [70, 150, 230, 310, 390, 470, 550, 630, 710, 770]
default_proc = [120, 90, 150, 110, 130, 100, 140, 95, 160, 105]


def get_default(lst, i, fallback):
    return lst[i] if i < len(lst) else fallback


input_data = []
cols = st.columns([1, 2, 2])
cols[0].markdown("**板坯**")
cols[1].markdown("**待温时间 (s)**")
cols[2].markdown("**精轧时间 (s)**")

for i in range(n):
    c0, c1, c2 = st.columns([1, 2, 2])
    c0.write(f"板坯 {i + 1}")
    w = c1.number_input(
        f"待温时间_{i}", min_value=0,
        value=int(get_default(default_wait, i, 100)), step=10, key=f"w_{i}"
    )
    p = c2.number_input(
        f"精轧时间_{i}", min_value=1,
        value=int(get_default(default_proc, i, 120)), step=10, key=f"p_{i}"
    )
    input_data.append((w, p))

st.subheader("2. 求解结果")

if st.button("开始排产"):
    wait_times = [x[0] for x in input_data]
    process_times = [x[1] for x in input_data]

    if mode == "固定顺序":
        try:
            order_1based = [
                int(x.strip()) for x in fixed_order_str.split(",") if x.strip()
            ]
        except ValueError:
            st.error("顺序格式不对，请用逗号分隔的数字，例如：3,1,5,2,4")
            st.stop()

        if len(order_1based) != n:
            st.error(f"顺序长度应为 {n}，当前为 {len(order_1based)}")
            st.stop()

        if sorted(order_1based) != list(range(1, n + 1)):
            st.error("顺序必须是 1 到 n 的一个排列，不能重复或遗漏")
            st.stop()

        order_0based = [x - 1 for x in order_1based]

        with st.spinner("正在按固定顺序计算..."):
            result = solve_fixed_order_flow_shop(
                wait_times, process_times, order_0based
            )
    else:
        with st.spinner("正在求解最优排产顺序..."):
            result = solve_flow_shop(
                wait_times, process_times, time_limit=time_limit
            )

    if result is None:
        st.error("未找到可行解，请检查输入数据。")
    else:
        st.success(f"求解状态：{result['status']}")
        col1, col2 = st.columns(2)
        col1.metric("最小总完工时间 makespan", f"{result['makespan']} s")
        col2.metric("板坯数量", n)

        st.markdown("### 排产顺序")
        st.dataframe(result["table"])

        st.markdown("### 待温 + 精轧 时间线")

        df = result["table"].copy()
        fig = go.Figure()

        for _, row in df.iterrows():
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
            height=400,
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