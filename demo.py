# patient_centered_ai_demo_final.py
# --- UI-Enhanced Version (same content, improved layout) ---

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap, base64, time, random
from datetime import datetime

# ----------------------------
# PAGE CONFIG & THEME
# ----------------------------
st.set_page_config(
    page_title="Patient-Centered AI — Final Demo (Lung Cancer)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject modern dark theme CSS
st.markdown("""
<style>
body {
    background-color: #0f1117;
    color: #f0f2f6;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #f3f4f6;
    font-weight: 700;
}
hr, .stTabs [data-baseweb="tab-list"] {
    border-color: #222 !important;
}

/* Container width */
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab"] {
    background: #1b1e27;
    color: #f0f2f6;
    border-radius: 0.6rem;
    padding: 0.5rem 1.25rem;
    margin-right: 0.4rem;
    transition: all 0.2s ease-in-out;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #25304a;
    color: #ffffff;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #0066ff, #00bcd4);
    color: #fff;
    box-shadow: 0 0 6px rgba(0, 136, 255, 0.5);
}

/* Card / box style */
.card {
    background: #1b1f2a;
    color: #e6e8eb;
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    box-shadow: 0 0 8px rgba(0,0,0,0.3);
    margin-top: 1rem;
}

/* Muted text */
.small-muted {
    color: #b3b8c2;
    font-size: 0.92rem;
    margin-bottom: 1.2rem;
}

/* Dataframe tweaks */
.css-1l269bu, .stDataFrame {
    background: #12151d !important;
    color: #eaeaea !important;
}

/* Metric progress / success highlights */
.stSuccess, .stMetric {
    background: #142a16 !important;
}

/* Slight breathing space between sections */
section {
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def simulate_alignment(qol, fatigue, steps, sleep, pf, ps, pm, treatment_bias=0.0):
    base = (
        (qol / 100) * 0.28 +
        (1 - fatigue / 10) * 0.26 +
        (steps / 10000) * 0.14 +
        (sleep / 9) * 0.08 +
        (pf * 0.12 + ps * 0.08 + pm * 0.04)
    ) * 100
    noise = np.random.normal(0, 2.5)
    return float(np.clip(base + treatment_bias + noise, 0, 100))

def build_dummy_patients(n=10):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append({
            "patient_id": f"P{i+1}",
            "age": int(rng.integers(40, 85)),
            "QoL": int(rng.integers(40, 95)),
            "Fatigue": int(rng.integers(1, 9)),
            "Steps": int(rng.integers(500, 9000)),
            "Sleep": round(rng.uniform(4.0, 8.5), 1),
            "Pref_Fatigue": round(rng.uniform(0.2,0.9), 2),
            "Pref_Survival": round(rng.uniform(0.1,0.7), 2),
            "Pref_Mobility": round(rng.uniform(0.1,0.6), 2),
            "site": random.choice(["Hospital A","Hospital B","Hospital C"])
        })
    return pd.DataFrame(rows)

def novelty_score_demo(n=12):
    return np.round(np.clip(np.random.beta(1,5,size=n), 0, 1),3)

def df_to_csv_bytes(df): return df.to_csv(index=False).encode('utf-8')
def make_download_link_csv(df, filename="data.csv", text="Download CSV"):
    b64 = base64.b64encode(df_to_csv_bytes(df)).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color:#00bcd4">{text}</a>'

def make_download_text(text, filename="report.txt", label="Download report"):
    b64 = base64.b64encode(text.encode('utf-8')).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}" style="color:#00bcd4">{label}</a>'

def styled_box_html(text, bg=None):
    return f"<div class='card'>{text}</div>"

# ----------------------------
# HEADER
# ----------------------------
st.markdown("<h1>🫀 Patient-Centered AI — Final Demo (Lung Cancer)</h1>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>A concept prototype demonstrating how PPI + COA + DHTM integration, explainability, federated learning, adaptive surveys and novelty detection can support shared decision-making.</div>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# MAIN TABS
# ----------------------------
tabs = st.tabs([
    "Overview", "Data Input", "AI Model", "Explainability", "Adaptive Timeline",
    "Federated Learning", "Treatment Comparison", "SNN Novelty Detection",
    "NLP Chat", "Summary & Export"
])

# Preserve state
if 'demo_seed' not in st.session_state: st.session_state['demo_seed'] = 42

# ----------------------------
# INCLUDE YOUR EXISTING TAB CODE HERE (unchanged content)
# ----------------------------
# ⬆️ Paste all your existing TAB sections (Overview → Summary & Export)
# Exactly as in your provided code — no changes required.


# ----------------------------
# TAB: Overview
# ----------------------------
with tabs[0]:
    st.subheader("Overview & Architecture")
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("""
        **Vision:** Create a trusted, privacy-preserving, explainable system that integrates patient preferences (PPI),
        clinical outcome assessments (COA), and digital health technology measures (DHTM) to support shared decision-making.
        """)
        st.write("- **PPI:** Stated preferences via adaptive questionnaires or DCEs.")
        st.write("- **COA:** QoL scores, symptom scales, clinical markers.")
        st.write("- **DHTM:** Wearables, ePROMs, telemetry.")
        st.write("- **Our demo:** illustrates the full pipeline (dummy data) and shows how each component contributes to patient-centered decisions.")
    with col2:
        # Simple architecture diagram using Graphviz via graphviz_chart
        st.graphviz_chart("""
            digraph {
                rankdir=LR;
                node [shape=box, style=filled, color="#E6F0FF", fontname="Helvetica"];
                Inputs [label="Inputs:\\nPPI (surveys)\\nCOA (QoL, symptoms)\\nDHT (wearables)"];
                Pre [label="Preprocessing:\\nharmonize, impute, NLP"];
                Integr [label="Integration:\\nMultimodal encoders / embeddings"];
                Model [label="Adaptive Model:\\nFederation + personalization"];
                Explain [label="Explainability & MCID module"];
                Dashboard [label="Decision Dashboard\\nClinician + Patient"];
                Inputs -> Pre -> Integr -> Model -> Explain -> Dashboard;
                subgraph cluster_f { label = "Federated Orchestration"; style=dashed; Fed[label="Aggregator (weights)"]; Model -> Fed; Fed -> Model; }
            }
        """)
    st.markdown(styled_box_html("<b>Presentation tip:</b> Use the 'Treatment Comparison' and 'Explainability' tabs to show clinicians how decisions are justified and actionable."), unsafe_allow_html=True)

# ----------------------------
# TAB: Data Input
# ----------------------------
with tabs[1]:
    st.subheader("Data Input — PPI, COA & DHT (Simulated)")
    left, right = st.columns([1,1])
    with left:
        st.markdown("### Patient Preferences (PPI) — Adaptive elicitation")
        pf = st.slider("Importance: Reduce Fatigue", 0.0, 1.0, 0.6, 0.05, key="pf_main")
        ps = st.slider("Importance: Survival Gain", 0.0, 1.0, 0.3, 0.05, key="ps_main")
        pm = st.slider("Importance: Physical Mobility", 0.0, 1.0, 0.5, 0.05, key="pm_main")
        st.caption("These weights are typical outputs of SP/DCE or adaptive questionnaires.")
    with right:
        st.markdown("### Clinical & DHT signals (COA & Wearables)")
        qol = st.slider("Quality of Life (0-100)", 0, 100, 70, key="qol_main")
        fatigue = st.slider("Fatigue (0-10)", 0, 10, 4, key="fat_main")
        steps = st.slider("Avg Daily Steps", 500, 10000, 4800, step=100, key="steps_main")
        sleep = st.slider("Average Sleep (hrs/day)", 3.0, 9.0, 6.5, 0.1, key="sleep_main")
        st.caption("Wearables and patient diaries provide continuous and episodic data (DHTM).")
    st.markdown("---")

    st.markdown("### Dummy cohort (multi-site) — example dataset")
    df_demo = build_dummy_patients(12)
    # Show a small interactive table and allow downloading
    st.dataframe(df_demo.style.format({"QoL":"{:.0f}","Fatigue":"{:.0f}","Steps":"{:.0f}"}), height=240)
    st.markdown(make_download_link_csv(df_demo, filename="demo_patients.csv", text="Download dummy cohort CSV"), unsafe_allow_html=True)

    st.write("")
    st.markdown(styled_box_html("<b>How data would be collected in real project:</b> standardized ePROMs/ePREMs, wearable APIs, EHR (FHIR), and adaptive SP instruments. All data harmonized and timestamped for multimodal integration."), unsafe_allow_html=True)

# ----------------------------
# TAB: AI Model
# ----------------------------
with tabs[2]:
    st.subheader("AI Model — Integration & Simulated Outputs")
    st.markdown("This tab demonstrates how the integration core produces preference-aligned scores for candidate treatments.")
    # two treatments meta
    treatments = {
        "Treatment A — Chemo + Immuno": 2.5,  # bias upwards (pretend higher survival)
        "Treatment B — Targeted Therapy": -1.5  # bias towards QoL
    }
    sim_scores = {name: simulate_alignment(qol, fatigue, steps, sleep, pf, ps, pm, bias) for name, bias in treatments.items()}
    sim_df = pd.DataFrame({"Treatment": list(sim_scores.keys()), "Alignment": list(sim_scores.values())})

    colA, colB = st.columns([2,1])
    with colA:
        fig = px.bar(sim_df, x="Treatment", y="Alignment", text_auto=".1f", color="Alignment", color_continuous_scale="Tealgrn")
        fig.update_layout(title="Preference-Aligned Scores (simulated)", yaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)
        chosen = sim_df.loc[sim_df.Alignment.idxmax(), "Treatment"]
        st.success(f"Preliminary recommendation: **{chosen}** (alignment {sim_df.Alignment.max():.1f}/100) — demo output.")
    with colB:
        st.markdown("### Model Confidence & Efficiency (demo gauges)")
        # Simple gauge: alignment confidence and model efficiency
        conf = round(np.clip(np.random.normal(0.75, 0.08), 0.45, 0.98),2)
        efficiency = round(np.clip(np.random.normal(0.68, 0.1), 0.35, 0.95),2)
        fig_g = go.Figure()
        fig_g.add_trace(go.Indicator(mode="gauge+number", value=conf*100, title={"text":"Alignment Confidence (%)"}, gauge={'axis':{'range':[0,100]}}))
        fig_g.add_trace(go.Indicator(mode="gauge+number", value=efficiency*100, title={"text":"Model Efficiency (%)"}, gauge={'axis':{'range':[0,100]}}, domain={'x':[0.5,1.0],'y':[0,1]}))
        fig_g.update_layout(height=290)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown(styled_box_html("Note: In production, the 'integration core' would be a modular pipeline (multimodal encoders, fine-tuned Foundation Models, or hybrid models). Federated training and SNN novelty detectors support robustness and privacy."), unsafe_allow_html=True)

# ----------------------------
# TAB: Explainability
# ----------------------------
with tabs[3]:
    st.subheader("Explainability & MCID checks")
    st.markdown("Explainability turns the model score into actionable reasons clinicians and patients can discuss.")
    # Simulated SHAP-like importance
    imp = {
        "Fatigue (symptom)": 0.28 * pf,
        "QoL (patient report)": 0.22 * (qol/100),
        "Steps (wearable)": 0.16 * (steps/10000),
        "Sleep (DHT)": 0.08 * (sleep/9),
        "Preference: Survival": 0.14 * ps,
        "Preference: Mobility": 0.12 * pm
    }
    imp_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Raw"])
    imp_df["Importance"] = imp_df["Raw"] / imp_df["Raw"].sum()
    fig_imp = px.bar(imp_df.sort_values("Importance", ascending=True), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
    st.plotly_chart(fig_imp, use_container_width=True)

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### Clinician-facing explanation (demo)")
        explanation = f"""
        The model identifies **Fatigue** and **QoL** as the dominant drivers for this patient's treatment alignment.
        For example, fatigue contributes {imp_df.loc[imp_df.Feature=='Fatigue (symptom)','Importance'].values[0]*100:.1f}% to the model's decision,
        while QoL contributes {imp_df.loc[imp_df.Feature=='QoL (patient report)','Importance'].values[0]*100:.1f}%.
        This suggests prioritizing therapies with lower fatigue burden to align with the patient's stated values.
        """
        st.markdown(styled_box_html(explanation), unsafe_allow_html=True)
        st.markdown("### MCID check (demo):")
        # Simulate MCID detection
        mcid_threshold = 10  # percent QoL improvement as demo threshold
        predicted_qol_gain = round(np.random.uniform(6,15),1)
        mcid_pass = predicted_qol_gain >= mcid_threshold
        st.write(f"- Predicted QoL gain: **{predicted_qol_gain}%**")
        st.write(f"- MCID threshold (demo): **{mcid_threshold}%**")
        if mcid_pass:
            st.success("This predicted gain exceeds the MCID — clinically meaningful (demo).")
        else:
            st.warning("Predicted gain below MCID — may not be clinically meaningful (demo).")
    with col2:
        st.markdown("### Counterfactual Explorer (toy)")
        st.markdown("Adjust fatigue level to see a simulated change in alignment (illustrative).")
        fatigue_test = st.slider("Test fatigue (0-10)", 0, 10, int(fatigue), key="fat_test")
        new_align_A = simulate_alignment(qol, fatigue_test, steps, sleep, pf, ps, pm, 2.5)
        new_align_B = simulate_alignment(qol, fatigue_test, steps, sleep, pf, ps, pm, -1.5)
        cf_df = pd.DataFrame({"Treatment":["A","B"], "Alignment":[new_align_A, new_align_B]})
        fig_cf = px.bar(cf_df, x="Treatment", y="Alignment", title="Counterfactual Alignment by Treatment", text_auto=".1f")
        st.plotly_chart(fig_cf, use_container_width=True)

# ----------------------------
# TAB: Adaptive Timeline
# ----------------------------
with tabs[4]:
    st.subheader("Adaptive Timeline & Survey Engine")
    st.markdown("Two linked demos: 1) adaptive model learning from repeated data; 2) adaptive survey personalization to reduce burden and improve signal quality.")
    weeks = st.slider("Simulate weeks (visits)", 4, 24, 12, key="weeks")
    rng = np.random.default_rng(101)
    qol_series = np.clip(qol + np.cumsum(rng.normal(0.4,1.0,weeks)), 30, 100)
    fat_series = np.clip(fatigue + np.cumsum(rng.normal(-0.1,0.5,weeks)), 0, 10)
    alignment_series = [simulate_alignment(int(q), float(f), steps + int(rng.normal(0,300)), sleep, pf, ps, pm, -1.2) for q,f in zip(qol_series, fat_series)]
    timeline_df = pd.DataFrame({"Week": list(range(1, weeks+1)), "QoL": qol_series, "Fatigue": fat_series, "Alignment": alignment_series})
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=timeline_df.Week, y=timeline_df.QoL, name="QoL", mode="lines+markers"))
    fig_tl.add_trace(go.Scatter(x=timeline_df.Week, y=timeline_df.Alignment, name="Alignment", mode="lines+markers"))
    fig_tl.update_layout(title="Patient Progress and Model Alignment (simulated)", xaxis_title="Week")
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("### Adaptive Survey Engine (demo)")
    # Simulate engagement scores and adaptation logic
    engagement = np.round(np.clip(np.random.normal(0.7, 0.12), 0.3, 1.0), 2)
    ignored_attr = random.choice(["Mobility", "Symptoms/side-effects", "Sleep quality"])
    adaptation_note = f"Detected lower engagement for questions related to **{ignored_attr}**. Next survey will reduce frequency for those items and focus on high-salience attributes like fatigue."
    colA, colB = st.columns([1,2])
    with colA:
        st.metric("Engagement (simulated)", f"{engagement*100:.0f}%")
        st.progress(int(engagement*100))
    with colB:
        st.markdown(styled_box_html(adaptation_note), unsafe_allow_html=True)
    st.caption("Adaptive surveys reduce burden and increase data quality — a core part of patient-centered design.")

# ----------------------------
# TAB: Federated Learning (with dataset view)
# ----------------------------
with tabs[5]:
    st.subheader("Federated Learning — Multi-site Demo & Dataset")
    st.markdown("This tab demonstrates the concept of local models + central aggregation and provides the multi-site dummy dataset.")

    # show the dummy dataset grouped by site
    df_sites = df_demo.copy()
    site_counts = df_sites['site'].value_counts().to_dict()
    st.markdown(f"Dummy cohort distribution across sites: {site_counts}")
    st.dataframe(df_sites, height=240)

    st.markdown("### Local training metrics (simulated) vs Global aggregated")
    sites = ["Hospital A", "Hospital B", "Hospital C"]
    local_pre = np.round(np.random.uniform(55,70,size=len(sites)),1)
    local_post = np.round(local_pre + np.random.uniform(6.0,10.0,size=len(sites)),1)
    fed_metrics = pd.DataFrame({"Site": sites, "Local Pre-agg (%)": local_pre, "Global Post-agg (%)": local_post})
    st.dataframe(fed_metrics.style.format({"Local Pre-agg (%)":"{:.1f}", "Global Post-agg (%)":"{:.1f}"}), height=200)
    fig_f = go.Figure()
    fig_f.add_trace(go.Bar(x=sites, y=local_pre, name="Local Pre-agg"))
    fig_f.add_trace(go.Bar(x=sites, y=local_post, name="Global Post-agg"))
    fig_f.update_layout(barmode='group', title="Federated effect: local vs aggregated (simulated)")
    st.plotly_chart(fig_f, use_container_width=True)

    st.markdown(styled_box_html("Privacy note: In a real system, secure aggregation protocols, differential privacy, and audit logging would be used. Here we only simulate the accuracy improvements of a federated approach."), unsafe_allow_html=True)
    st.markdown(make_download_link_csv(df_sites, filename="federated_demo_dataset.csv", text="Download federated demo dataset (CSV)"), unsafe_allow_html=True)

# ----------------------------
# TAB: Treatment Comparison
# ----------------------------
with tabs[6]:
    st.subheader("Treatment Comparison — Multi-metric")
    st.markdown("Compare candidate treatments in clinical, toxicity, duration and patient-preference-alignment metrics.")
    # treatments meta (recreated for clarity)
    treatments_meta = [
        {"name":"Treatment A — Chemo + Immuno","effectiveness":85,"toxicity":70,"duration_weeks":24},
        {"name":"Treatment B — Targeted Therapy","effectiveness":78,"toxicity":25,"duration_weeks":12}
    ]
    tr_df = pd.DataFrame(treatments_meta)
    tr_df["Alignment"] = tr_df["name"].apply(lambda x: simulate_alignment(qol, fatigue, steps, sleep, pf, ps, pm, (3 if "Chemo" in x else -3)))
    st.dataframe(tr_df.style.format({"effectiveness":"{:.0f}","toxicity":"{:.0f}","duration_weeks":"{:.0f}","Alignment":"{:.1f}"}), height=220)

    # radar chart / spider for visual comparison
    categories = ["Effectiveness", "Toxicity (inverted)", "Duration (inverted)", "Alignment"]
    traces = []
    for idx, row in tr_df.iterrows():
        eff = row["effectiveness"]
        tox_inv = 100 - row["toxicity"]
        dur_inv = 100 - (row["duration_weeks"] / 24 * 100)  # normalize duration to 24w
        align = row["Alignment"]
        values = [eff, tox_inv, dur_inv, align]
        traces.append(go.Scatterpolar(r=values, theta=categories, fill='toself', name=row["name"]))
    fig_radar = go.Figure(data=traces)
    fig_radar.update_layout(title="Treatment Radar Comparison", polar=dict(radialaxis=dict(visible=True, range=[0,100])))
    st.plotly_chart(fig_radar, use_container_width=True)
    st.success(f"Preliminary recommendation (demo): {tr_df.loc[tr_df.Alignment.idxmax(),'name']} — highest alignment with patient preferences.")

    st.markdown(styled_box_html("Use this view in stakeholder meetings to show tradeoffs clearly: clinical gains vs toxicity vs patient preference alignment."), unsafe_allow_html=True)

# ----------------------------
# TAB: SNN Novelty Detection
# ----------------------------
with tabs[7]:
    st.subheader("SNN-based Novelty Detection (Simulated)")
    st.markdown("This module demonstrates how an SNN-style novelty detector can flag unusual patterns during training and inference.")
    colA, colB = st.columns([2,1])
    with colA:
        st.markdown("### Training-phase novelty timeline (simulated)")
        n = 18
        nov_scores = novelty_score_demo(n)
        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(x=list(range(1,n+1)), y=nov_scores, mode='lines+markers', name='Novelty score'))
        fig_n.update_layout(title="Novelty scores across training batches (0=normal,1=highly novel)", yaxis=dict(range=[0,1]))
        st.plotly_chart(fig_n, use_container_width=True)
        st.markdown("When novelty is high for a batch, our logic pauses updates or downweights the batch to avoid catastrophic learning on anomalies.")
    with colB:
        st.markdown("### Inference-phase anomaly table")
        inf_samples = [f"Case {i+1}" for i in range(6)]
        inf_scores = novelty_score_demo(6)
        df_inf = pd.DataFrame({"Sample": inf_samples,"Novelty": inf_scores,
                  "Action": ["Review" if s>0.35 else "Auto" for s in inf_scores]})
        st.dataframe(df_inf, height=230)
        st.markdown(styled_box_html("SNN detectors are energy-efficient and well-suited to time-series novelty detection on-device (edge). Here we simulate alarms that route unusual cases to clinician review."), unsafe_allow_html=True)
    # small heatmap showing where novelty concentrated (simulated)
    st.markdown("### Novelty heatmap (simulated batches x features)")
    heat = np.round(np.clip(np.random.normal(0.05, 0.12, (8,6)), 0, 1), 3)
    fig_heat = px.imshow(heat, labels=dict(x="Feature Index", y="Batch"), x=list(range(1,7)), y=list(range(1,9)), color_continuous_scale="Reds")
    st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------
# TAB: NLP Chat (LLM-style)
# ----------------------------
with tabs[8]:
    st.subheader("NLP Chat — LLM-style explanation (canned)")
    st.markdown("Use simple prompts to produce clinician/patient-friendly explanations. These are canned templates for demo; real LLM integration would need privacy controls and prompt safety.")
    prompt = st.selectbox("Prompt", [
        "Explain the recommendation in plain language",
        "Why is Treatment A less preferred?",
        "Summarize patient progress (past weeks)",
        "Explain novelty detection role",
        "What is MCID and how was it applied?"
    ])
    if st.button("Generate explanation"):
        # simple canned responses customized with variables
        if prompt == "Explain the recommendation in plain language":
            out = f"Recommendation (plain): Based on your stated priorities (fatigue: {pf:.2f}, survival: {ps:.2f}, mobility: {pm:.2f}) and the wearable and QoL data, the system suggests the treatment that best preserves quality of life and reduces fatigue while maintaining effectiveness. It provides an explainable rationale behind this suggestion."
        elif prompt == "Why is Treatment A less preferred?":
            out = "Treatment A has higher toxicity and longer duration. For this patient, whose priority is reducing fatigue and maintaining mobility, the trade-off favors a therapy with lower toxicity even if survival benefit is slightly lower."
        elif prompt == "Summarize patient progress (past weeks)":
            out = f"Patient progress summary: QoL baseline ~{qol}, recent trend shows fluctuations; fatigue remains around {fatigue}/10. The model's alignment score has been updating as new wearable and survey data arrive."
        elif prompt == "Explain novelty detection role":
            out = "Novelty detectors flag unusual data during training or monitoring; flagged cases are routed for manual review to avoid corrupting model updates and to protect patient safety."
        else:
            out = "MCID (Minimal Clinically Important Difference) is the smallest change in an outcome (e.g., QoL) that patients perceive as beneficial. In this demo we use a placeholder MCID threshold to illustrate how we would report clinically meaningful changes."
        # "typing" simulation
        with st.spinner("AI is generating explanation..."):
            time.sleep(0.8)
        st.markdown(styled_box_html(out), unsafe_allow_html=True)

# ----------------------------
# TAB: Summary & Export
# ----------------------------
with tabs[9]:
    st.subheader("Summary & Export (Stakeholder-ready)")
    st.markdown("A brief stakeholder summary and options to export demo artifacts (CSV / TXT).")
    chosen_treatment = sim_df.loc[sim_df.Alignment.idxmax(), "Treatment"]
    summary_text = textwrap.dedent(f"""
    Patient-centered AI — demo summary
    Date: {datetime.now().strftime('%Y-%m-%d')}
    Recommended treatment (demo): {chosen_treatment}
    Alignment score: {sim_df.Alignment.max():.1f}/100
    Key drivers: Fatigue, QoL, Mobility (simulated)
    MCID demo: predicted QoL gain approx. {round(np.random.uniform(6,15),1)}% (demo)
    Notes: This is a simplified prototype demonstrating architecture and core components: PPI collection, multimodal integration,
    explainability, federated learning, SNN novelty detection and adaptive surveys. In production we will replace simulated parts with
    validated models and secure federated orchestration.
    """)
    st.text_area("Stakeholder summary (editable)", summary_text, height=260)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(make_download_text(summary_text, filename="demo_summary.txt", label="Download summary (.txt)"), unsafe_allow_html=True)
    with col2:
        st.markdown(make_download_link_csv(df_demo, filename="demo_cohort.csv", text="Download cohort CSV"), unsafe_allow_html=True)
    with col3:
        if st.button("Generate PDF (placeholder)"):
            # create a simple bytes buffer pretend pdf (real implementation would render actual PDF)
            b = f"PDF report (placeholder)\n\n{summary_text}".encode('utf-8')
            b64 = base64.b64encode(b).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="demo_report.pdf">Download demo_report.pdf</a>'
            st.markdown(href, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(styled_box_html("Presentation tip: Use the 'Treatment Comparison' and 'Explainability' tabs during panel Q&A to show how clinical tradeoffs are reconciled with patient preferences and how MCID is used as a clinical filter."), unsafe_allow_html=True)

# ----------------------------
# End of app
# ----------------------------
