import streamlit as st
import pandas as pd
import stock_analyzer as sa

st.set_page_config(page_title="Stock Health Analyzer", page_icon="📈", layout="wide")
st.title("📈 Stock Health Analyzer")
st.caption("Technical + Fundamental + Machine-Learning check. Data: Yahoo Finance. Educational use only - not investment advice.")

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Ticker symbol", "RELIANCE.NS").strip().upper()
st.sidebar.caption("NSE: TCS.NS | BSE: TCS.BO | US: AAPL")
start = st.sidebar.date_input("History from", pd.Timestamp("2015-01-01"))
horizon = st.sidebar.selectbox("Predict price direction after (trading days)", [5, 10, 20], index=0)
run = st.sidebar.button("Analyze", type="primary")


@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(ticker, start, horizon):
    return sa.analyze(ticker, str(start), horizon)


if run:
    try:
        with st.spinner("Downloading data and training models..."):
            R = run_analysis(ticker, start, horizon)
    except Exception as e:
        st.error(f"Could not analyze '{ticker}': {e}")
        st.stop()

    name = R["info"].get("longName", ticker)
    st.subheader(f"{name} ({ticker})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall score", f"{R['overall']}/100")
    c2.metric("Technical", f"{R['tech_score']}/100")
    c3.metric("Fundamental", "NA" if R["fund_score"] is None else f"{R['fund_score']}/100")
    c4.metric(f"P(price up in {horizon}d)", f"{R['ml']['prob_up']:.0%}")
    st.success(f"Verdict: **{R['label']}**")

    t1, t2, t3, t4, t5 = st.tabs(["Chart", "Technical", "Fundamental", "ML model", "Download data"])
    with t1:
        st.pyplot(sa.plot_dashboard(R["data"], ticker))
    with t2:
        st.dataframe(R["tech_table"], use_container_width=True)
    with t3:
        st.dataframe(sa.company_snapshot(R["info"]).astype(str), use_container_width=True)
        if R["fund_table"].empty:
            st.warning("Yahoo Finance returned no fundamental ratios for this ticker.")
        else:
            st.dataframe(R["fund_table"], use_container_width=True)
    with t4:
        st.write(f"Best model (chosen by time-series cross-validation): **{R['ml']['best_name']}**")
        st.dataframe(R["ml"]["results"].round(3), use_container_width=True)
        st.info(f"A model that always guesses the majority class scores about {R['ml']['baseline']:.0%} on the test period. "
                "Accuracy only a little above that is normal - markets are very hard to predict.")
    with t5:
        xls = sa.build_excel_bytes(R["prices"], R["data"], R["info"], R["statements"],
                                   R["tech_table"], R["fund_table"], R["ml"]["results"])
        st.download_button("⬇️ Download Excel report", xls, file_name=f"{ticker}_analysis.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Enter a ticker in the sidebar and click **Analyze**.")
