"""Interactive live-demo dashboard.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metacost.data import (ALL_FEATURES, FEATURE_SETS, GROUP_OF, NOISE_FLOOR_MSE, load_all)
from metacost.calibration import estimate_offset_prior, shrunk_offset
from metacost.evaluation import metrics
from metacost.models import make_model

st.set_page_config(page_title="MetaCost - HILO metabolic cost predictor", page_icon="🦿", layout="wide")
BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
SUBJ_C = {"S1": BLUE, "S2": ORANGE}
RES = ROOT / "results"


@st.cache_data
def get_data():
    return load_all()


@st.cache_data
def table(name):
    p = RES / "tables" / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def saved_model(subject, fs, name):
    slug = f"{subject}__{fs.replace(' ', '').replace('+', '_')}__{name.split(' ')[0].rstrip('.')}.joblib"
    p = RES / "models" / slug
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def fit_lodo(subject, fs, model_name, day):
    d = get_data()[subject]
    X, y = d.features(FEATURE_SETS[fs]), d.y
    tr, te = d.day != day, d.day == day
    m = make_model(model_name).fit(X[tr], y[tr])
    if model_name == "Gaussian Process":
        mu, sd = m.predict(X[te], return_std=True)
    else:
        mu, sd = m.predict(X[te]), None
    return np.ravel(mu), sd


@st.cache_resource
def offset_prior(subject, fs, model_name, day):
    d = get_data()[subject]
    tr = d.day != day
    return estimate_offset_prior(make_model(model_name), d.features(FEATURE_SETS[fs])[tr], d.y[tr], d.day[tr])


data = get_data()

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🦿 MetaCost")
st.sidebar.caption("Predicting metabolic cost during human-in-the-loop exoskeleton optimisation")
subject = st.sidebar.radio("Subject", ["S1", "S2"], format_func=lambda s: f"Subject {s[-1]}", horizontal=True)
fs = st.sidebar.selectbox("Feature set", list(FEATURE_SETS), index=0)
model_name = st.sidebar.selectbox("Model", ["Gaussian Process", "LASSO", "SVR (RBF)", "Bayesian-reg. NN",
                                            "Extra Trees", "PLS"], index=0)
d = data[subject]
st.sidebar.markdown("---")
st.sidebar.metric("Samples", len(d.y))
st.sidebar.metric("Target variance", f"{d.y.var():.4f}")
st.sidebar.metric("Noise-floor MSE", f"{NOISE_FLOOR_MSE}")

st.title("Predicting Metabolic Cost During Human-in-the-Loop Optimisation")
st.caption("Replace a slow, mask-based metabolic measurement with wearable EMG + gait sensors.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Data", "🔮 Predict an unseen day", "🎛️ What-if simulator", "🏆 Results", "📂 Batch predict"])

# ---------------------------------------------------------------- tab 1: data
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiment days", int(d.day.max() + 1))
    c2.metric("Conditions / day", 36)
    c3.metric("Features", 29, "9 step · 16 EMG · 4 control", delta_color="off")
    c4.metric("Mean cost (normalised)", f"{d.y.mean():.3f}")
    t = d.day * 36 + d.condition
    fig = go.Figure(go.Scatter(x=t, y=d.y, mode="lines+markers", line=dict(color=SUBJ_C[subject], width=2),
                               marker=dict(size=5), name="Metabolic cost",
                               hovertemplate="Day %{customdata[0]}, condition %{customdata[1]}<br>cost %{y:.3f}",
                               customdata=np.c_[d.day + 1, d.condition + 1]))
    for k in range(1, 5):
        fig.add_vline(x=k * 36 - .5, line_dash="dash", line_color="#8a8984")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title="Metabolic cost across 5 days",
                      xaxis_title="Condition", yaxis_title="Normalised metabolic cost")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Correlation of every feature with metabolic cost**")
    r = pd.DataFrame({"feature": ALL_FEATURES, "group": [GROUP_OF[f] for f in ALL_FEATURES],
                      "r": [np.corrcoef(d.X[f], d.y)[0, 1] for f in ALL_FEATURES]})
    fig = go.Figure(go.Bar(x=r.feature, y=r.r, marker_color=[{"EMG": BLUE, "Step": AQUA, "Control": "#eda100"}[g]
                                                            for g in r.group]))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Pearson r")
    st.plotly_chart(fig, width="stretch")
    with st.expander("Raw data"):
        st.dataframe(d.X.assign(day=d.day + 1, metabolic_cost=d.y), width="stretch")

# ---------------------------------------------------------------- tab 2: unseen day
with tab2:
    st.markdown("Train on **4 days**, predict the **5th day the model has never seen** - "
                "the realistic deployment scenario.")
    c1, c2 = st.columns([3, 2])
    day = c1.select_slider("Held-out day", options=list(range(5)), format_func=lambda k: f"Day {k+1}", value=4)
    k_cal = c2.slider("Mask spot-checks on this day (calibration)", 0, 8, 0,
                      help="Measure k random conditions with the mask, then shift all other predictions by an "
                           "empirical-Bayes shrunken offset (see metacost/calibration.py).")
    with st.spinner("Training on the other four days..."):
        mu, sd = fit_lodo(subject, fs, model_name, day)
        sigma2, tau2 = offset_prior(subject, fs, model_name, day)
    y = d.y[d.day == day]
    cond = d.condition[d.day == day] + 1
    cal = np.sort(np.random.RandomState(day).choice(len(y), k_cal, replace=False))
    rest = np.setdiff1d(np.arange(len(y)), cal)
    off, w = shrunk_offset(y[cal] - mu[cal], sigma2, tau2)
    mu_c = mu + off
    m0, m = metrics(y[rest], mu[rest]), metrics(y[rest], mu_c[rest])
    rho = spearmanr(y[rest], mu_c[rest])[0]
    base = np.mean((y[rest] - d.y[d.day != day].mean()) ** 2)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", f"{m['MSE']:.4f}",
              f"{(m['MSE'] / m0['MSE'] - 1) * 100:+.0f}% from calibration" if k_cal else
              (f"{(1 - m['MSE'] / base) * 100:.0f}% better than predicting the mean" if m["MSE"] < base else
               f"{(m['MSE'] / base - 1) * 100:.0f}% worse than predicting the mean - try calibration"),
              delta_color="inverse" if k_cal else "normal")
    c2.metric("Error (% of walking cost)", f"{m['pct_error']:.1f}%")
    c3.metric("Day offset applied", f"{off:+.3f}", f"trust weight w = {w:.2f}", delta_color="off")
    c4.metric("Rank correlation (Spearman)", f"{rho:.2f}", help="What CMA-ES actually needs: correct ordering")
    fig = go.Figure()
    if sd is not None:
        fig.add_trace(go.Scatter(x=np.r_[cond, cond[::-1]], y=np.r_[mu_c + 1.96 * sd, (mu_c - 1.96 * sd)[::-1]],
                                 fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                                 name="95 % interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=cond, y=y, mode="lines+markers", name="Measured", line=dict(color="#0b0b0b")))
    if k_cal:
        fig.add_trace(go.Scatter(x=cond, y=mu, mode="lines", name="Before calibration",
                                 line=dict(color="#8a8984", dash="dot")))
    fig.add_trace(go.Scatter(x=cond, y=mu_c, mode="lines+markers", name=f"Predicted ({model_name})",
                             line=dict(color=SUBJ_C[subject], dash="dash")))
    if k_cal:
        fig.add_trace(go.Scatter(x=cond[cal], y=y[cal], mode="markers", name="Mask spot-checks",
                                 marker=dict(size=14, color="#eda100", symbol="diamond",
                                             line=dict(color="white", width=2))))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Condition (2-min control law)",
                      yaxis_title="Normalised metabolic cost", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    best_true, best_pred = int(cond[rest][np.argmin(y[rest])]), int(cond[rest][np.argmin(mu_c[rest])])
    rank_of_pred = int((y[rest] < y[rest][np.argmin(mu_c[rest])]).sum()) + 1
    st.info(f"**Optimiser view:** the true cheapest condition was **#{best_true}**; the model would pick "
            f"**#{best_pred}**, which is truly ranked **{rank_of_pred} / {len(rest)}**.")

# ---------------------------------------------------------------- tab 3: what-if
with tab3:
    bundle = saved_model(subject, fs if fs in ("All features", "Step + EMG", "EMG only") else "All features",
                         "Gaussian Process")
    if bundle is None:
        st.warning("Run `python scripts/run_experiments.py` first to train the saved models.")
    else:
        feats = bundle["features"]
        st.markdown(f"Start from a real condition and nudge the sensor readings. "
                    f"Model: **Gaussian Process** on **{bundle['feature_set']}** (trained on all of Subject {subject[-1]}'s data).")
        idx = st.slider("Start from sample #", 0, len(d.y) - 1, 0)
        base = d.X.iloc[idx][feats].copy()
        mean_r = {f: abs(np.corrcoef(d.X[f], d.y)[0, 1]) for f in feats}
        top = sorted(feats, key=mean_r.get, reverse=True)[:8]
        cols = st.columns(4)
        x = base.copy()
        for i, f in enumerate(top):
            lo, hi = float(d.X[f].min()), float(d.X[f].max())
            x[f] = cols[i % 4].slider(f, lo, hi, float(base[f]), (hi - lo) / 100)
        mu, sd = bundle["model"].predict(x.to_numpy()[None, :], return_std=True)
        mu0 = bundle["model"].predict(base.to_numpy()[None, :])
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted metabolic cost", f"{mu[0]:.3f}", f"{mu[0] - mu0[0]:+.3f} vs. start")
        c2.metric("95 % interval", f"{mu[0] - 1.96 * sd[0]:.3f} – {mu[0] + 1.96 * sd[0]:.3f}")
        c3.metric("Measured at start", f"{d.y[idx]:.3f}")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=float(mu[0]),
                                     gauge=dict(axis=dict(range=[0.3, 1.6]), bar=dict(color=SUBJ_C[subject]),
                                                steps=[dict(range=[0.3, 1.0], color="#e7f3ec"),
                                                       dict(range=[1.0, 1.6], color="#fbe9e2")]),
                                     title=dict(text="Cost relative to normal walking (1.0)")))
        fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tab 4: results
with tab4:
    e3 = table("E3_model_zoo.csv")
    if e3 is None:
        st.warning("No results yet - run `python scripts/run_experiments.py`.")
    else:
        proto = st.radio("Validation protocol", ["random", "day"], horizontal=True,
                         format_func=lambda p: "Random 5-fold ×5 (paper)" if p == "random" else "Leave-one-day-out")
        sub = e3[(e3.subject == subject) & (e3.protocol == proto)]
        piv = sub.pivot(index="model", columns="feature_set", values="MSE").sort_values("All features")
        st.dataframe(piv.style.format("{:.4f}").background_gradient(cmap="Blues_r"), width="stretch")
        st.caption("Paper (Krimsky & Ng 2018) best MSE: Subject 1 = 0.0089, Subject 2 = 0.0168.")
        figs = sorted((RES / "figures").glob("*.png"))
        pick = st.selectbox("Figure", [f.stem for f in figs])
        if pick:
            st.image(str(RES / "figures" / f"{pick}.png"), width="stretch")

# ---------------------------------------------------------------- tab 5: batch
with tab5:
    st.markdown("Upload a CSV with the feature columns below to get predictions "
                "(e.g. `data/hilo_metabolic_dataset.csv`).")
    bundle = saved_model(subject, "All features", "Gaussian Process")
    st.code(", ".join(ALL_FEATURES), language=None)
    up = st.file_uploader("CSV file", type="csv")
    if up is not None and bundle is not None:
        df = pd.read_csv(up)
        missing = [f for f in bundle["features"] if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            mu, sd = bundle["model"].predict(df[bundle["features"]].to_numpy(), return_std=True)
            df["predicted_cost"], df["pred_std"] = mu, sd
            st.dataframe(df, width="stretch")
            st.download_button("Download predictions", df.to_csv(index=False), "predictions.csv")
