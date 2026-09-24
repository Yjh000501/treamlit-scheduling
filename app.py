import streamlit as st
import pandas as pd
import plotly.express as px
from ortools.sat.python import cp_model
import plotly.graph_objects as go


# =========================================================
# 求解函数：单机调度 1 | r_i | Cmax
# =========================================================
def solve_single_machine_schedule(wait_times, process_times, time_limit=30):
    n = len(wait_times)
    assert n == len(process_times)

    horizon = max(wait_times) + sum(process_times)

    model = cp_model.CpModel()

    start = [model.NewIntVar(wait_times[i], horizon, f"start_{i}") for i in range(n)]
    end = [model.NewIntVar(0, horizon, f"end_{i}") for i in range(n)]
    interval = [
        model.NewIntervalVar(start[i], process_times[i], end[i], f"interval_{i}")
        for i in range(n)
    ]

    rank = [model.NewIntVar(0, n - 1, f"rank_{i}") for i in range(n)]
    model.AddAllDifferent(rank)

    model.AddNoOverlap(interval)

    before = {}
    for i in range(n):
        for j in range(i + 1, n):
            b = model.NewBoolVar(f"before_{i}_{j}")
            before[(i, j)] = b
            model.Add(rank[i] < rank[j]).OnlyEnforceIf(b)
            model.Add(rank[j] < rank[i]).OnlyEnforceIf(b.Not())
            model.Add(end[i] <= start[j]).OnlyEnforceIf(b)
            model.Add(end[j] <= start[i]).OnlyEnforceIf(b.Not())

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, end)
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
            "开始精轧(s)": solver.Value(start[i]),
            "结束精轧(s)": solver.Value(end[i]),
        })

    return {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "table": pd.DataFrame(rows),
    }


# =========================================================
# 页面配置
# =========================================================
st.set_page_config(page_title="板坯待温-精轧协同排产", layout="wide")
st.title("板坯待温-精轧协同排产")
st.caption("输入每块板坯的待温时间和精轧时间，系统自动求解最小 makespan 的最优排产顺序。")

# =========================================================
# 输入区
# =========================================================
st.sidebar.header("参数设置")

n = st.sidebar.number_input("板坯数量", min_value=1, max_value=50, value=10, step=1)
time_limit = st.sidebar.number_input("求解时间上限（秒）", min_value=1, max_value=300, value=30, step=1)

st.subheader("1. 输入板坯数据")

default_wait = [70, 150, 230, 310, 390, 470, 550, 630, 710, 770]
default_proc = [120, 90, 150, 110, 130, 100, 140, 95, 160, 105]

# 保证默认数据长度足够
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
        f"待温时间_{i}", min_value=0, value=int(get_default(default_wait, i, 100)),
        step=10, key=f"w_{i}"
    )
    p = c2.number_input(
        f"精轧时间_{i}", min_value=1, value=int(get_default(default_proc, i, 120)),
        step=10, key=f"p_{i}"
    )
    input_data.append((w, p))

# =========================================================
# 求解
# =========================================================
st.subheader("2. 求解结果")

if st.button("开始排产"):
    wait_times = [x[0] for x in input_data]
    process_times = [x[1] for x in input_data]

    with st.spinner("正在求解最优排产顺序..."):
        result = solve_single_machine_schedule(wait_times, process_times, time_limit=time_limit)

    if result is None:
        st.error("未找到可行解，请检查输入数据。")
    else:
        st.success(f"求解状态：{result['status']}")
        col1, col2 = st.columns(2)
        col1.metric("最小总完工时间 makespan", f"{result['makespan']} s")
        col2.metric("板坯数量", n)

        st.markdown("### 最优排产顺序")
        st.dataframe(result["table"])

        import plotly.graph_objects as go

        st.markdown("### 精轧机时间线（甘特图）")

        df = result["table"].copy()
        df["持续"] = df["结束精轧(s)"] - df["开始精轧(s)"]

        fig = go.Figure()

        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["持续"]],
                y=[row["板坯"]],
                base=[row["开始精轧(s)"]],
                orientation="h",
                name=row["板坯"],
                text=f"顺序 {row['顺序']}｜{row['开始精轧(s)']}→{row['结束精轧(s)']}s",
                textposition="inside",
                hoverinfo="text",
            ))

        fig.update_layout(
            barmode="stack",
            xaxis_title="时间 (s)",
            yaxis_title="板坯",
            showlegend=False,
            height=400,
            margin=dict(l=80, r=40, t=40, b=40),
        )

        # 让 y 轴按排产顺序从上到下显示
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=list(df["板坯"])[::-1],
        )

        st.plotly_chart(fig, use_container_width=True)

        # 下载结果
        csv = result["table"].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载排产结果 CSV",
            data=csv,
            file_name="排产结果.csv",
            mime="text/csv",
        )