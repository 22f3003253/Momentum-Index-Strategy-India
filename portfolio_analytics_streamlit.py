import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="India Momentum Dashboard",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================
code_path = Path(r"C:\Users\0310a\Desktop\momentum_output")
RETURNS_PATH = code_path / "daily_returns.csv"
STATS_PATH = code_path / "performance_stats.csv"
WEIGHTS_PATH = code_path / "portfolio_weights_history.csv"
REBAL_PATH = code_path/ "rebalancing_log.csv"

returns = pd.read_csv(
    RETURNS_PATH,
    index_col=0,
    parse_dates=True
)

stats = pd.read_csv(
    STATS_PATH,
    index_col=0
)

weights = pd.read_csv(
    WEIGHTS_PATH,
    parse_dates=["Date"]
)

rebal = pd.read_csv(
    REBAL_PATH,
    parse_dates=["Date"]
)

# ============================================================
# PREP
# ============================================================

cum = (1 + returns).cumprod() * 100

momentum = returns["Momentum"]
n500 = returns["Nifty500"]

# ============================================================
# FUNCTIONS
# ============================================================

def drawdown(series):

    wealth = (1 + series).cumprod()

    peak = wealth.cummax()

    dd = (wealth - peak) / peak

    return dd * 100


def annualized_return(r):

    years = len(r) / 252

    total = (1 + r).prod()

    return total ** (1 / years) - 1


def annualized_vol(r):

    return r.std() * np.sqrt(252)


def tracking_error(port, bench):

    diff = port - bench

    return diff.std() * np.sqrt(252)


def information_ratio(port, bench):

    diff = port - bench

    te = diff.std() * np.sqrt(252)

    if te == 0:
        return np.nan

    return diff.mean() * 252 / te


# ============================================================
# HEADER
# ============================================================

st.title("MSCI-Style India Momentum Dashboard")

st.caption(
    f"Latest Data : {returns.index.max().date()}"
)

# ============================================================
# KPI ROW
# ============================================================

mom_stats = stats["India Momentum"]

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "CAGR",
    f"{mom_stats['CAGR (%)']:.2f}%"
)

col2.metric(
    "Sharpe",
    f"{mom_stats['Sharpe Ratio']:.2f}"
)

col3.metric(
    "Volatility",
    f"{mom_stats['Ann. Volatility (%)']:.2f}%"
)

col4.metric(
    "Max DD",
    f"{mom_stats['Max Drawdown (%)']:.2f}%"
)

col5.metric(
    "Win Rate",
    f"{mom_stats['Monthly Win Rate']*100:.1f}%"
)

col6.metric(
    "Total Return",
    f"{mom_stats['Total Return (%)']:.1f}%"
)

st.divider()

# ============================================================
# GROWTH OF 100
# ============================================================

st.subheader("Growth of ₹100")

fig = px.line(
    cum,
    labels={
        "value": "Index Value",
        "index": "Date"
    }
)

fig.update_layout(
    height=500,
    legend_title=""
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# ROLLING ALPHA & DRAWDOWN
# ============================================================

left, right = st.columns(2)

with left:

    st.subheader("Rolling 12M Excess Return vs Nifty500")

    rolling_returns = pd.DataFrame({
        "Momentum": (
            (1 + momentum)
            .rolling(252)
            .apply(np.prod, raw=True) - 1
        ) * 100,

        "Nifty500": (
            (1 + n500)
            .rolling(252)
            .apply(np.prod, raw=True) - 1
        ) * 100
    })
    fig_alpha = px.line(
    rolling_returns.dropna()
)

    fig_alpha.update_layout(
        height=350,
        yaxis_title="Rolling 12M Return (%)"
    )

    st.plotly_chart(
        fig_alpha,
        width="stretch"
    )

with right:

    st.subheader("Drawdown")

    dd = pd.DataFrame({
        "Momentum": drawdown(momentum),
        "Nifty500": drawdown(n500)
    })

    fig_dd = px.line(dd)

    fig_dd.update_layout(height=350)

    st.plotly_chart(
        fig_dd,
        use_container_width=True
    )

# ============================================================
# RISK ANALYTICS
# ============================================================
st.subheader("Risk Analytics")

risk_df = pd.DataFrame({

    "Metric": [
        "Annual Return (%)",
        "Annual Volatility (%)",
        "Sharpe Ratio",
        "Max Drawdown (%)"
    ],

    "Momentum": [
        round(annualized_return(momentum) * 100, 2),
        round(annualized_vol(momentum) * 100, 2),
        round(stats.loc["Sharpe Ratio", "India Momentum"], 2),
        round(drawdown(momentum).min(), 2)
    ],

    "Nifty500": [
        round(annualized_return(n500) * 100, 2),
        round(annualized_vol(n500) * 100, 2),
        round(stats.loc["Sharpe Ratio", "Nifty 500"], 2),
        round(drawdown(n500).min(), 2)
    ]
})

risk_df["Difference"] = (
    risk_df["Momentum"] - risk_df["Nifty500"]
).round(2)

styled_risk = (
    risk_df.style
    .background_gradient(
        subset=["Difference"],
        cmap="RdYlGn"
    )
    .format({
        "Momentum": "{:.2f}",
        "Nifty500": "{:.2f}",
        "Difference": "{:+.2f}"
    })
)

st.dataframe(
    styled_risk,
    use_container_width=True,
    hide_index=True
)

st.markdown("### Active Risk Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Tracking Error",
        f"{tracking_error(momentum, n500)*100:.2f}%"
    )

with col2:
    st.metric(
        "Information Ratio",
        f"{information_ratio(momentum, n500):.2f}"
    )

# st.subheader("Momentum vs Nifty500")

# metrics = st.columns(4)

# metrics[0].metric(
#     "Annual Return",
#     f"{annualized_return(momentum)*100:.2f}%",
#     f"{(annualized_return(momentum)-annualized_return(n500))*100:.2f}%"
# )

# metrics[1].metric(
#     "Volatility",
#     f"{annualized_vol(momentum)*100:.2f}%",
#     f"{(annualized_vol(momentum)-annualized_vol(n500))*100:.2f}%"
# )

# metrics[2].metric(
#     "Sharpe",
#     f"{stats.loc['Sharpe Ratio','India Momentum']:.2f}",
#     f"{stats.loc['Sharpe Ratio','India Momentum'] - stats.loc['Sharpe Ratio','Nifty 500']:.2f}"
# )

# metrics[3].metric(
#     "Max Drawdown",
#     f"{drawdown(momentum).min():.2f}%",
#     f"{drawdown(momentum).min() - drawdown(n500).min():.2f}%"
# )
# ============================================================
# YEARLY RETURNS
# ============================================================

st.subheader("Annual Returns")

annual = returns.resample("Y").apply(
    lambda x: (1 + x).prod() - 1
)

annual = annual * 100

fig_ann = go.Figure()
fig_ann.add_bar(
    x=annual.index.year,
    y=annual["Momentum"],
    name="Momentum",
    text=annual["Momentum"].round(1),
    textposition="outside"
)

fig_ann.add_bar(
    x=annual.index.year,
    y=annual["Nifty500"],
    name="Nifty500",
    text=annual["Nifty500"].round(1),
    textposition="outside"
)

fig_ann.update_layout(
    barmode="group",
    height=450,
    uniformtext_minsize=8,
    uniformtext_mode="hide"
)

fig_ann.update_traces(
    texttemplate="%{text:.1f}%"
)

st.plotly_chart(
    fig_ann,
    use_container_width=True
)

# ============================================================
# TOP HOLDINGS
# ============================================================

st.subheader("Current Top Holdings")

latest_date = weights["Date"].max()


top10 = (
    weights[weights["Date"] == latest_date]
    .sort_values("Weight (%)", ascending=False)
    .head(10)
    [["Ticker", "Weight (%)"]]
)

left, right = st.columns([2,1])

with left:

    st.subheader("Top 10 Holdings")

    st.dataframe(
        top10.style.format({
            "Weight (%)": "{:.2f}%"
        }),
        width="stretch"
    )

with right:

    st.metric(
        "Top 10 Concentration",
        f"{top10['Weight (%)'].sum():.2f}%"
    )

    st.metric(
        "Largest Position",
        f"{top10['Weight (%)'].max():.2f}%"
    )

# ============================================================
# FULL PERFORMANCE TABLE
# ============================================================

st.subheader("Performance Statistics")

st.dataframe(
    stats,
    use_container_width=True
)