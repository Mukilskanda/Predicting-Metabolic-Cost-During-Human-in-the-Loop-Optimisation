"""Build docs/MetaCost_Writeup.pdf (2 pages) from the results tables.   python docs/build_writeup.py"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, NextPageTemplate, KeepTogether)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
OUT = ROOT / "docs" / "MetaCost_Writeup.pdf"

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d2")
TINT = colors.HexColor("#eef4fc")

body = ParagraphStyle("b", fontName="Times-Roman", fontSize=9.2, leading=11.3, alignment=TA_JUSTIFY,
                      textColor=INK, spaceAfter=3)
h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9.6, leading=12, textColor=BLUE,
                   spaceBefore=4, spaceAfter=2)
cap = ParagraphStyle("c", fontName="Helvetica", fontSize=7.2, leading=8.6, textColor=INK2, spaceAfter=4)
title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=15.5, leading=19, alignment=TA_CENTER)
sub = ParagraphStyle("s", fontName="Helvetica", fontSize=8.6, leading=11, alignment=TA_CENTER, textColor=INK2)
bullet = ParagraphStyle("bl", parent=body, leftIndent=8, bulletIndent=0, spaceAfter=1.5)
cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.3, leading=8.6)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")

W, H = A4
M = 14 * mm
GAP = 6 * mm
COLW = (W - 2 * M - GAP) / 2
HEAD_H = 27 * mm


def frames(first):
    top = H - M - (HEAD_H if first else 0)
    fr = []
    if first:
        fr.append(Frame(M, H - M - HEAD_H, W - 2 * M, HEAD_H, id="head", showBoundary=0,
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0))
    for i in range(2):
        fr.append(Frame(M + i * (COLW + GAP), M, COLW, top - M, id=f"c{i}", showBoundary=0,
                        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0))
    return fr


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7)
    c.setFillColor(INK2)
    c.drawString(M, 8 * mm, "UE24CS352A Machine Learning - Mini-Project #11")
    c.drawRightString(W - M, 8 * mm, f"Page {d.page} / 2")
    c.restoreState()


P = lambda t, s=body: Paragraph(t, s)
B = lambda t: Paragraph(t, bullet, bulletText="•")


def fig(name, width, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(FIG / name)).getSize()
    return [KeepTogether([Image(str(FIG / name), width=width, height=width * ih / iw), P(caption, cap)])]


# ---- numbers from the result tables --------------------------------------------------
e3 = pd.read_csv(TAB / "E3_model_zoo.csv")
e1 = pd.read_csv(TAB / "E1_reproduction.csv")


def mse(s, m, proto="random", fs="All features"):
    return e3[(e3.subject == s) & (e3.model == m) & (e3.protocol == proto) & (e3.feature_set == fs)].MSE.item()


hdr = ParagraphStyle("hdr", parent=cellb, textColor=colors.white)
rows = [[Paragraph(t, hdr) for t in ["Model", "S1<br/>random", "S1<br/>unseen day", "S2<br/>random", "S2<br/>unseen day"]]]
show = ["Constant (mean)", "LASSO", "Bayesian-reg. NN", "PLS", "SVR (RBF)", "Extra Trees",
        "Gaussian Process", "Ensemble (avg)"]
best = {(s, p): e3[(e3.subject == s) & (e3.protocol == p) & (e3.feature_set == "All features")
                   & (e3.model.isin(show[1:]))].MSE.min() for s in ["S1", "S2"] for p in ["random", "day"]}
for m in show:
    r = [Paragraph(m, cell)]
    for s in ["S1", "S2"]:
        for p in ["random", "day"]:
            v = mse(s, m, p)
            r.append(Paragraph(f"{v:.4f}", cellb if abs(v - best[(s, p)]) < 1e-12 else cell))
    rows.append(r)
rows.append([Paragraph("<i>Paper (Krimsky &amp; Ng)</i>", cell), Paragraph("0.0089", cell), Paragraph("-", cell),
             Paragraph("0.0168", cell), Paragraph("-", cell)])
tbl = Table(rows, colWidths=[COLW * 0.32] + [COLW * 0.17] * 4)
tbl.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.3), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
    ("LINEBELOW", (0, -2), (-1, -2), 0.6, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ("LEFTPADDING", (0, 0), (-1, -1), 3),
]))

abl = [["Feature set", "S1 LASSO", "S1 GP", "S2 LASSO", "S2 GP"]]
for fs in ["All features", "Step + EMG", "EMG only", "Step only"]:
    g = lambda s, m: e1[(e1.subject == s) & (e1.model == m) & (e1.feature_set == fs)].MSE.item()
    abl.append([fs] + [f"{g(s, m):.4f}" for s in ["S1", "S2"] for m in ["LASSO", "Gaussian Process"]])
tbl2 = Table(abl, colWidths=[COLW * 0.32] + [COLW * 0.17] * 4)
tbl2.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Helvetica", 7.3), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.3),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
    ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
]))

story = [
    P("Predicting Metabolic Cost During Human-in-the-Loop Exoskeleton Optimisation", title),
    Spacer(1, 3),
    P("Team: Mukil Skanda A (PES1UG24CS281), Likhith M (PES1UG24CS282) &nbsp;|&nbsp; UE24CS352A Machine Learning, "
      "Mini-Project - Problem #11 &nbsp;|&nbsp; October 2026", sub),
    P("Code: github.com/Mukilskanda/Predicting-Metabolic-Cost-During-Human-in-the-Loop-Optimisation", sub),
    NextPageTemplate("later"), FrameBreak(),

    P("1. Problem statement", h),
    P("Ankle exoskeletons are personalised by <b>human-in-the-loop optimisation (HILO)</b>: a CMA-ES optimiser "
      "applies a new torque profile every 2 minutes and keeps the profiles that minimise the wearer's "
      "<b>metabolic cost</b>. Metabolic cost, however, is measured by indirect calorimetry through a face "
      "mask: it needs ~1 min to settle, is very noisy, and only works in the lab. <b>Goal:</b> learn a regression "
      "model that predicts metabolic cost from signals a wearable could record (surface EMG, gait "
      "features, controller settings), so that HILO could run faster and outside the laboratory. We reproduce "
      "and extend Krimsky &amp; Ng (Stanford CS229, 2018)."),

    P("2. Dataset", h),
    P("Stanford Biomechatronics Lab HILO study with bilateral ankle exoskeletons and naive users. "
      "<b>2 subjects × 5 days × 36 control conditions</b>; one row covers the last 30 s of a 2-min "
      "condition. One physically impossible (negative) target is removed, leaving <b>180 + 179 samples</b> "
      "with <b>29 features</b>:"),
    B("<b>9 step/gait</b>: L/R peak vertical ground-reaction force (÷ body weight), L/R peak dorsi- and "
      "plantar-flexion angles, L/R step time and step width, each the per-step median."),
    B("<b>16 EMG</b> (8 muscles per leg): band-pass, rectify, low-pass and RMS, normalised by the "
      "normal-walking peak of the same day."),
    B("<b>4 control</b>: peak torque, peak time, rise time and fall time of the exoskeleton torque profile."),
    B("<b>Target</b>: metabolic rate minus the day's standing baseline, ÷ the day's normal-walking rate "
      "(1.0 = unassisted walking). Variance: 0.0665 (S1) and 0.0358 (S2). Measurement noise sets an "
      "MSE floor of about 0.0025."),
    P("The data are tiny (p ≈ n/6), noisy and <b>non-stationary</b>: cost falls by ~40 % over the five "
      "days as subjects learn to walk with the device (Fig. 1)."),
    *fig("F1_dataset_overview.png", COLW, "<b>Fig. 1.</b> Measured metabolic cost; black bars mark daily "
         "means. Both subjects adapt over the days."),

    P("3. Approach", h),
    B("<b>Leakage-free, nested evaluation.</b> Every model is a scikit-learn pipeline, so scaling and all "
      "hyper-parameter searches (LASSO λ, SVR C/ε, PLS components, number of NN neurons) run on training "
      "folds only. The paper selected λ on the same folds it reported."),
    B("<b>Two protocols.</b> (a) Random 5-fold CV repeated 5× (the paper's protocol, for comparison). "
      "(b) <b>Leave-one-day-out</b>: predict an entire unseen session. This is the realistic case, which the paper "
      "named as the ideal test but did not run."),
    B("<b>13 models + ensemble.</b> OLS, LASSO, Ridge, Elastic Net, PLS, kNN, SVR-RBF, Random Forest, "
      "Extra Trees, Gradient Boosting, a <b>Gaussian Process</b> (linear + RBF + white-noise kernel, giving "
      "calibrated uncertainty) and a <b>Bayesian-regularised NN</b>. The NN is our NumPy re-implementation of "
      "MATLAB <i>trainbr</i>: Levenberg-Marquardt with MacKay evidence updates of α, β and γ."),
    B("<b>Analyses.</b> Feature-group ablation, neuron sweep, permutation and LASSO-stability importance, "
      "PCA vs. PLS vs. forward selection, learning curves, GP calibration, cross-subject transfer, and "
      "<i>within-day rank correlation</i> (what the optimiser actually needs)."),

    P("4. Implementation", h),
    P("Python 3 · NumPy · scikit-learn · Matplotlib. Two layers: <font face='Courier'>paper_reproduction/</font>, "
      "a 1:1 port of the paper's MATLAB scripts that reproduces its baseline (S2 Lin. Reg. 0.0172 vs 0.0168); and "
      "<font face='Courier'>metacost/</font> + <font face='Courier'>scripts/</font> (12 experiments, 16 figures, "
      "10 unit tests incl. a Jacobian check), and a command-line predictor used for the live demo, which trains "
      "on 4 days and predicts the unseen 5th."),

    P("5. Results", h),
    P("<b>Table 1.</b> Cross-validated MSE using all 29 features (bold = best per column).", cap),
    tbl, Spacer(1, 4),
    P("<b>Beats the paper.</b> Under the paper's protocol, our best MSE is <b>0.0079</b> for S1 "
      f"(ensemble; R² = 0.88, ≈ 8.9 % error) versus the paper's 0.0089, and <b>0.0109</b> for S2 (SVR) versus 0.0168, "
      "i.e. 11 % and 35 % lower. The Gaussian Process is the most consistent single model."),
    P("<b>Unseen days are harder.</b> Leave-one-day-out roughly doubles the error. S1 still generalises "
      "(OLS/PLS/Ridge MSE ≈ 0.014, R² = 0.79; GP r = 0.88 in Fig. 2) with a within-day Spearman ρ ≈ 0.6. S2 drops to "
      "R² = 0.27: its day-level drop in cost on days 3-5 is not visible in the sensors. Simple, heavily regularised linear "
      "models generalise best across days, while flexible trees and kNN overfit within-day structure."),
    *fig("F5_unseen_day_timeline.png", COLW * 0.94, "<b>Fig. 2.</b> Each day predicted by a GP trained only on the "
         "other four days, shaded by its 95 % interval."),
    P("<b>Which signals matter?</b> Table 2 reproduces the paper's ablation: <b>EMG beats step data</b> for S1 and for LASSO on S2 "
      "(for the GP on S2 the two groups are similar), and dropping the control parameters costs little (S1: 0.0081 → 0.0090). Grouped permutation importance "
      "ranks the groups EMG &gt; Step &gt; Control for S1 and EMG &gt; Control &gt; Step for S2. Right-leg muscles R4 and R7 "
      "(S1) and R2 (S2) are selected in 93-100 % of 200 LASSO bootstraps, so a reduced sensor set is "
      "feasible."),
    P("<b>Table 2.</b> Feature-set ablation, MSE (random CV).", cap), tbl2, Spacer(1, 5),
    *fig("F8_feature_importance.png", COLW, "<b>Fig. 3.</b> GP permutation importance (held-out folds): top-12 "
         "features per subject and grouped importance."),
    P("<b>Dimensionality reduction.</b> PCA ignores the target, which is why the paper found it worked poorly. Supervised "
      "<b>PLS with a single component</b> (0.0138) matches the best PCA model, which needs 8 components, on S2. "
      "<b>Neural networks:</b> the evidence framework selects only 1-2 hidden neurons, and the NN never beats the GP. "
      "<b>Across people:</b> a model trained on S1 and tested on S2 does worse than predicting the mean "
      "(MSE 0.152 vs 0.036). Per-subject z-scoring cuts this to 0.046, which is still not useful. "
      "<b>Uncertainty:</b> the GP's 95 % intervals cover 94 % of unseen-day points for S1 but only 77 % for S2, "
      "which correctly flags the harder subject. Subject 1's learning curve (0.0155 → 0.0088) is still falling."),
    *fig("F9_dimensionality_reduction.png", COLW, "<b>Fig. 4.</b> CV MSE vs. number of components/features. "
         "Supervised PLS needs far fewer dimensions than PCA."),

    *fig("F4_model_zoo.png", COLW, "<b>Fig. 5.</b> All models, random CV (dark) vs. leave-one-day-out (light). "
         "Red dashes: predict-the-mean baseline; dotted line: noise floor."),
    P("6. Improvements without new data", h),
    P("<b>Session calibration.</b> If k conditions of a new day are also measured with the mask, the day's "
      "predictions can be shifted by the mean residual. A raw mean of k noisy residuals is itself noisy, so we use an "
      "<b>empirical-Bayes shrunken offset</b>, w·mean(residual) with w = k/(k + σ²/τ²). Here σ² (within-day noise) and "
      "τ² (true day-offset variance) are estimated from the training days only. With <b>3 random spot-checks out of 36</b> "
      "(92 % fewer mask readings), unseen-day MSE for S2 drops <b>0.0460 → 0.0223 (GP, −51 %)</b> and 0.0362 → 0.0182 "
      "(Ridge). S1 drops 0.0175 → 0.0142 (Fig. 6). Calibrating on the <i>first</i> conditions of a session is worse: "
      "cost runs high early in each session, so it hurt S1 at k ≥ 5 (Ridge 0.0120 → 0.0143, or 0.0184 without shrinkage)."),
    *fig("F14_day_calibration.png", COLW, "<b>Fig. 6.</b> Unseen-day MSE vs. number of mask measurements on the "
         "new day. Solid: random spot-checks; dashed: start of session (scored on conditions 9-36)."),
    P("<b>Fewer sensors.</b> Keeping only the k EMG channels that enter the LASSO path first (chosen inside each "
      "training fold) shows that <b>4 electrodes beat all 16 for S2</b>: unseen-day Ridge 0.0362 → 0.0242, and random-CV GP "
      "0.0132 → 0.0114. S1 needs about 8 electrodes to stay within 3 % (Fig. 7). <b>Transfer:</b> recalibrating a "
      "model from the other person does not help. A model trained on just 5-10 of the new person's own conditions wins."),
    *fig("F15_sensor_reduction.png", COLW, "<b>Fig. 7.</b> CV MSE vs. number of EMG electrodes kept (all step and "
         "control features always kept). Solid: random CV; dashed: unseen day."),
    P("7. Conclusions", h),
    B("EMG and gait signals predict normalised metabolic cost to within about <b>9-10 %</b> within a session "
      "(mean baseline: 19-26 %), improving on the original study for both subjects (MSE 0.0079 and 0.0109)."),
    B("Honest evaluation exposed the real weakness: on a completely unseen day the error rises to 12 % (S1) "
      "and 16 % (S2), because random splits leak day-level information."),
    B("<b>We substantially reduced this unseen-day weakness without collecting new data.</b> With "
      "<b>empirical-Bayes shrinkage</b>, just <b>3 mask spot-checks</b> per 36-condition session (92 % fewer mask "
      "readings) cut Subject 2's unseen-day error by <b>51 %</b> (GP MSE 0.046 → 0.022) without harming Subject 1. "
      "Keeping only <b>4 of the 16 EMG electrodes</b> also lowered Subject 2's unseen-day Ridge MSE from 0.036 to "
      "<b>0.024</b>, pointing to cheaper wearable hardware."),
    B("Models must still be <b>subject-specific</b>: cross-person transfer fails even after calibration. "
      "<b>Future work</b> (needs new data): more subjects, anthropometrics and IMU kinematics, and the "
      "calibrated surrogate as a prior inside CMA-ES or Bayesian optimisation."),
    P("<font size=6.8 color='#52514e'><b>References.</b> [1] E. Krimsky, E. Ng, CS229 Stanford, 2018. "
      "[2] J. Zhang et al., <i>Science</i> 356:1280, 2017. [3] D. MacKay, <i>Neural Comp.</i> 4:415, 1992. "
      "[4] F. Foresee, M. Hagan, ICNN 1997. [5] C. Rasmussen, C. Williams, <i>GPML</i>, MIT Press 2006.</font>",
      body),
]

doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                      title="MetaCost - Predicting Metabolic Cost During HILO", author="Team #11")
doc.addPageTemplates([PageTemplate("first", frames(True), onPage=footer),
                      PageTemplate("later", frames(False), onPage=footer)])
doc.build(story)
print("wrote", OUT)
