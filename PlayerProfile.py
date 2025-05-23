import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from openai import OpenAI

@st.cache_data

def load_data():
    data = pd.read_excel("PremCC - Attempt 2.xlsx", sheet_name="Data")
    data = data[data['Attempt'] != 0]

    for col in data.columns:
        if "Speed" in col or "Time" in col or "Distance" in col or "Acuity" in col or "Digit Span" in col or col == "Accuracy":
            data[col] = pd.to_numeric(data[col], errors='coerce')

    static_acuity_col = ['Visual Acuity']
    dynamic_cols = ['Dynamic 1', 'Dynamic 2', 'Dynamic 3']
    contrast_col = ['Contrast']
    stereopsis_cols = ['Stereopsis 1', 'Stereopsis 2', 'Stereopsis 3']
    color_col = ['Color']

    reaction_cols = ['Reaction Speed 1', 'Reaction Speed 2', 'Reaction Speed 3', 'Reaction Speed 4', 'Reaction Speed 5']
    decision_speed_cols = ['Decision Making Speed 1', 'Decision Making Speed 2', 'Decision Making Speed 3', 'Decision Making Speed 4', 'Decision Making Speed 5']
    cognitive_cols = ['Digit Span', 'Visual Memory']
    hand_eye_cols = [c for c in data.columns if 'Hand-eye-coordination time' in c]
    anticipation_linear_cols = [c for c in data.columns if 'Anticipation Time-Linear' in c]
    anticipation_sin_cols = [c for c in data.columns if 'Anticipation Time-Sin' in c]
    anticipation_cols = anticipation_linear_cols + anticipation_sin_cols

    def normalize(series):
        norm = (series - series.min()) / (series.max() - series.min())
        return norm.clip(0, 1)

    def scale_to_range(series, min_val=2, max_val=5):
        return series * (max_val - min_val) + min_val

    def compute_score(cols, invert=False, outlier_max_cap=False):
        raw = data[cols].mean(axis=1, skipna=True)
        raw[raw.isna()] = 0
        if outlier_max_cap:
            cap_value = pd.concat([data[c] for c in cols], axis=0).quantile(0.95)
            raw = raw.clip(upper=cap_value)
        if invert:
            raw = 1 - normalize(raw.fillna(raw.max()))
        else:
            raw = normalize(raw.fillna(0))
        return scale_to_range(raw)

    data['Static Acuity Avg'] = data[static_acuity_col].mean(axis=1, skipna=True)
    data['Dynamic Acuity Avg'] = data[dynamic_cols].mean(axis=1, skipna=True)
    data['Contrast Sensitivity'] = data[contrast_col].mean(axis=1, skipna=True)
    data['Stereopsis Avg'] = data[stereopsis_cols].mean(axis=1, skipna=True)

    static_score = 1 - normalize(data['Static Acuity Avg'].fillna(data['Static Acuity Avg'].max()))
    dynamic_score = 1 - normalize(data['Dynamic Acuity Avg'].fillna(data['Dynamic Acuity Avg'].max()))
    stereo_score = 1 - normalize(data['Stereopsis Avg'].fillna(data['Stereopsis Avg'].max()))
    color_score = data['Color'].apply(lambda x: 1 if str(x).strip().lower() in ['normal', 'none', 'no deficiency'] else 0).fillna(0)

    raw_vision = (
        0.50 * static_score +
        0.50 * dynamic_score +
        0.25 * stereo_score +
        0.20 * normalize(data['Contrast Sensitivity'].fillna(0)) +
        0.05 * color_score
    )
    data['Vision Score'] = scale_to_range(normalize(raw_vision))

    reaction_cap = pd.concat([data[c] for c in reaction_cols], axis=0).quantile(0.95)
    decision_cap = pd.concat([data[c] for c in decision_speed_cols], axis=0).quantile(0.95)

    data['Reaction Time Avg'] = data[reaction_cols].mean(axis=1, skipna=True).clip(upper=reaction_cap)
    data['Decision Making Speed Avg'] = data[decision_speed_cols].mean(axis=1, skipna=True).clip(upper=decision_cap)
    data['Decision Latency'] = data['Decision Making Speed Avg'] - data['Reaction Time Avg']

    data['Reaction Score'] = compute_score(reaction_cols, invert=True, outlier_max_cap=True)
    data['Decision Speed Subscore'] = compute_score(decision_speed_cols, invert=True, outlier_max_cap=True)
    data['Decision Accuracy Subscore'] = scale_to_range(normalize(data['Accuracy'].fillna(0)))
    data['Decision Latency Subscore'] = scale_to_range(1 - normalize(data['Decision Latency'].fillna(data['Decision Latency'].max())))

    data['Decision Making Score'] = (
        data['Decision Speed Subscore'] + data['Decision Accuracy Subscore'] + data['Decision Latency Subscore']
    ) / 3

    data['Cognitive Avg'] = data[cognitive_cols].mean(axis=1, skipna=True)
    data['Hand-Eye Avg'] = data[hand_eye_cols].mean(axis=1, skipna=True)

    hand_eye_score = compute_score(hand_eye_cols, invert=True, outlier_max_cap=True)
    hand_eye_score[data[hand_eye_cols].isna().all(axis=1)] = 2
    data['Hand-Eye Score'] = hand_eye_score

    data['Cognitive Score'] = compute_score(cognitive_cols)

    anticipation_linear_score = compute_score(anticipation_linear_cols, invert=True, outlier_max_cap=True)
    anticipation_sin_score = compute_score(anticipation_sin_cols, invert=True, outlier_max_cap=True)
    data['Anticipation Score'] = 0.55 * anticipation_linear_score + 0.45 * anticipation_sin_score

    progression_data = data.sort_values(by=["Name", "Attempt"])
    latest_attempts = progression_data.groupby("Name").tail(1)

    return latest_attempts, progression_data

def radar_chart(player_row, name, color=None):
    labels = ["Vision", "Reaction", "Decision Making", "Cognitive", "Hand-Eye", "Anticipation"]
    values = [
        player_row['Vision Score'],
        player_row['Reaction Score'],
        player_row['Decision Making Score'],
        player_row['Cognitive Score'],
        player_row['Hand-Eye Score'],
        player_row['Anticipation Score']
    ]
    return go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name=name,
        line=dict(color=color) if color else None
    )

def extract_strengths_weaknesses(pdata):
    insights = []
    strengths = []
    weaknesses = []

    def check_motion_loss(static, dynamic):
        if static and dynamic and static > 0:
            loss_percent = ((dynamic - static) / static) * 100
            if loss_percent > 25:
                weaknesses.append(f"Dynamic vision drops significantly (~{int(loss_percent)}% worse than static acuity)")
            elif 0 < loss_percent <= 10:
                insights.append("Objects in motion have little to no effect on how far he can see")
            elif loss_percent < 0:
                strengths.append("Performs better visually when objects are moving")
        return None

    def check_decision_conflict(speed, latency, accuracy):
        insight = []
        if speed and latency:
            reaction_speed = speed - latency
            if reaction_speed > 0:
                delay_pct = (reaction_speed / speed) * 100
                if delay_pct > 50:
                    insight.append(f"Significant delay after decision (~{int(delay_pct)}% of speed duration spent in latency)")
        if accuracy is not None and accuracy < 0.65:
            accuracy_drop = (1 - accuracy / 0.9) * 100
            insight.append(f"Decision accuracy is below expectation (~{int(accuracy_drop)}% shortfall from 90%)")
        return ", ".join(insight) if insight else None

    def check_anticipation_diff(score, sin=None, linear=None):
        if sin is not None and linear is not None:
            diff = abs(sin - linear)
            if min(sin, linear) > 0:
                percent_diff = (diff / min(sin, linear)) * 100
                if percent_diff > 25:
                    weaknesses.append(f"Anticipation differs significantly (~{int(percent_diff)}% gap between Sin and Linear) – may struggle with pattern consistency")
                elif percent_diff <= 10:
                    strengths.append("Consistent anticipation across sinusoidal and linear patterns")
        return None

    # Compare static vs dynamic acuity
    motion_loss = check_motion_loss(pdata.get('Static Acuity Avg'), pdata.get('Dynamic Acuity Avg'))
    if motion_loss:
        insights.append(motion_loss)

    # Compare decision metrics
    decision_insight = check_decision_conflict(
        pdata.get('Decision Making Speed Avg'),
        pdata.get('Decision Latency'),
        pdata.get('Accuracy')
    )
    if decision_insight:
        insights.append(decision_insight)

    # Compare sin vs linear anticipation if available
    sin_keys = [k for k in pdata.index if 'Anticipation Time-Sin' in k]
    linear_keys = [k for k in pdata.index if 'Anticipation Time-Linear' in k]
    if sin_keys and linear_keys:
        sin_avg = pdata[sin_keys].mean()
        linear_avg = pdata[linear_keys].mean()
        ant_diff = check_anticipation_diff(pdata.get('Anticipation Score'), sin_avg, linear_avg)
        if ant_diff:
            insights.append(ant_diff)

    raw_thresholds = {
        'Static Acuity Avg': ('Vision', 'low'),
        'Dynamic Acuity Avg': ('Vision', 'low'),
        'Contrast Sensitivity': ('Vision', 'high'),
        'Stereopsis Avg': ('Vision', 'low'),
        'Reaction Time Avg': ('Reaction', 'low'),
        'Decision Making Speed Avg': ('Decision Making', 'low'),
        'Decision Latency': ('Decision Making', 'low'),
        'Accuracy': ('Decision Making', 'high'),
        'Digit Span': ('Cognitive', 'high'),
        'Visual Memory': ('Cognitive', 'high'),
        'Hand-Eye Avg': ('Hand-Eye', 'low'),
        'Anticipation Score': ('Anticipation', 'high')
    }
    category_scores = {}

    for metric, (category, preference) in raw_thresholds.items():
        value = pdata.get(metric, None)
        if pd.isnull(value):
            continue
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append((metric, value, preference))

    for category, items in category_scores.items():
        low_metrics = []
        high_metrics = []
        for metric, val, pref in items:
            if metric == 'Stereopsis Avg':
                if val < 100:
                    strengths.append(category)
                elif val > 800:
                    weaknesses.append(category)
                continue
            if pref == 'low':
                low_metrics.append(val)
            elif pref == 'high':
                high_metrics.append(val)

        if low_metrics:
            low_avg = sum(low_metrics) / len(low_metrics)
            if low_avg < 4:
                strengths.append(category)
            elif low_avg > 6:
                weaknesses.append(category)

        if high_metrics:
            high_avg = sum(high_metrics) / len(high_metrics)
            if high_avg > 4:
                strengths.append(category)
            elif high_avg < 3:
                weaknesses.append(category)

    weaknesses = list(set(weaknesses) - set(strengths))
    return strengths, weaknesses, insights

def generate_summary(name, strengths, weaknesses):
    summary = f"**{name} Summary:**\n"
    if strengths:
        summary += f"Shows strong performance in {', '.join(strengths)}. "
    if not strengths:
        summary += "Has scored low across attributes, needs improvement"
    if weaknesses:
        summary += f"Needs improvement in {', '.join(weaknesses)}."
    if not weaknesses:
        summary += "Has a balanced profile across all attributes."
    
    return summary

def setupAI():
    client = OpenAI(
        api_key=""
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {"role": "user", "content": "write a haiku about ai"}
        ]
        )
    pass

st.set_page_config(page_title="Player Dashboard", layout="wide")
# Add this right after st.set_page_config(...)
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("SportOptimaLogoWhite.png", width=100)

with col_title:
    st.markdown("""
    <div style='display: flex; align-items: center; height: 100%;'>
        <h2 style='margin: 0; font-size : 45px'>Prem CC Performance Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    #st.markdown("<h2 style='padding-top: 20px; text-align: left;'>Sport Optima Performance Dashboard</h2>", unsafe_allow_html=True)

latest_data, full_data = load_data()

setupAI()

# Convert to tabs instead of radio
tab1, tab2 = st.tabs(["🧠 Player Deepdive", "🆚 Compare Players"])

with tab1:
    st.header("Current Player Status")
    col_player, _ = st.columns([1, 5])
    with col_player:
        selected_player = st.selectbox("Select a Player", latest_data['Name'].unique(), key="current")
    player_data = latest_data[latest_data['Name'] == selected_player].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radar Profile")
        fig = go.Figure(data=[radar_chart(player_data, selected_player)])
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)
        st.plotly_chart(fig, key=f"radar_chart_{selected_player}")
    with col2:
        st.subheader("Key Metrics")
        key_metrics = [
            'Static Acuity Avg', 'Dynamic Acuity Avg', 'Contrast Sensitivity', 'Stereopsis Avg',
            'Reaction Time Avg', 'Decision Making Speed Avg', 'Decision Latency', 'Accuracy',
            'Digit Span', 'Visual Memory', 'Hand-Eye Avg', 'Anticipation Score']
        latest_all = latest_data.set_index('Name')
        metric_descriptions = {
            'Static Acuity Avg': "Ability to identify fine detail. Critical for sighting the ball early.",
            'Dynamic Acuity Avg': "Clarity while objects are moving. Important for judging fast deliveries.",
            'Contrast Sensitivity': "Ability to distinguish shades. Useful in poor lighting or against similar backgrounds.",
            'Stereopsis Avg': "Depth perception. Crucial for catching and gauging ball trajectory.",
            'Reaction Time Avg': "How fast you respond to stimuli. Impacts fielding and batting reflexes.",
            'Decision Making Speed Avg': "Speed of choosing between options. Affects shot selection and gameplay strategy.",
            'Decision Latency': "Delay between decision and action. Impacts real-time gameplay decisions.",
            'Accuracy': "Precision of decision outcomes. Affects consistency in performance.",
            'Digit Span': "Short-term memory span. Useful for recalling field placements or opponent patterns.",
            'Visual Memory': "Ability to remember visual cues. Helps in anticipation and strategic decisions.",
            'Hand-Eye Avg': "Coordination between vision and hand movement. Core skill for batting, catching, throwing.",
            'Anticipation Score': "Ability to predict what's next. Crucial for reacting before the event occurs."
        }
        metrics_df = pd.DataFrame({'Metric': key_metrics})
        
        metrics_df[selected_player] = metrics_df["Metric"].map(player_data.to_dict())
        metrics_df["Team Avg"] = metrics_df["Metric"].map(latest_all[key_metrics].mean().to_dict())
        metrics_df["Percentile"] = metrics_df["Metric"].map(
            lambda m: latest_all[m].rank(pct=True).get(selected_player, None) * 100 if m in latest_all else None
        ).round(1)
        metrics_df["What it means"] = metrics_df["Metric"].map(metric_descriptions)
        st.dataframe(metrics_df)

        s, w, i = extract_strengths_weaknesses(player_data)
        st.subheader("Detailed Insights")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Strengths")
            if s:
                for category in s:
                    st.markdown(f"- {category}")
            else:
                st.markdown("None")
        with col_b:
            st.markdown("### Weaknesses")
            if w:
                for category in w:
                    st.markdown(f"- {category}")
            else:
                st.markdown("None")

        st.markdown("---")
        st.markdown("### Summary")
        summary_bullets = []
        if s:
            summary_bullets.append(f"**Strength Areas:** {', '.join(s)}")
        if w:
            summary_bullets.append(f"**Needs Improvement In:** {', '.join(w)}")
        if not s and not w:
            summary_bullets.append("**Balanced performer across all metrics.**")
        for item in summary_bullets:
            st.markdown(f"- {item}")

        if i:
            st.markdown("### Diagnostic Insights")
            for insight in i:
                st.markdown(f"- {insight}")

with tab2:
    st.header("Compare Two Players")
    players = st.multiselect("Select Two Players", latest_data['Name'].unique(), default=list(latest_data['Name'].unique())[:2])
    if len(players) == 2:
        pdata1 = latest_data[latest_data['Name'] == players[0]].iloc[0]
        pdata2 = latest_data[latest_data['Name'] == players[1]].iloc[0]

        def extract_strengths_weaknesses(pdata):
            strengths = []
            weaknesses = []
            for key in ['Vision Score', 'Reaction Score', 'Decision Making Score', 'Cognitive Score', 'Hand-Eye Score', 'Anticipation Score']:
                if pdata[key] > 4:
                    strengths.append(key.replace(' Score', ''))
                elif pdata[key] < 3:
                    weaknesses.append(key.replace(' Score', ''))
            return strengths, weaknesses

        s1, w1 = extract_strengths_weaknesses(pdata1)
        s2, w2 = extract_strengths_weaknesses(pdata2)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.markdown(f"### {players[0]}")
            st.markdown(generate_summary(players[0],s1,w1))
            # st.markdown("**Strengths:** " + (", ".join(s1) if s1 else "None"))
            # st.markdown("**Weaknesses:** " + (", ".join(w1) if w1 else "None"))

        with col2:
            fig = go.Figure(data=[
                radar_chart(pdata1, players[0], color='blue'),
                radar_chart(pdata2, players[1], color='red')
            ])
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True)
            st.plotly_chart(fig)

        with col3:
            st.markdown(f"### {players[1]}")
            st.markdown(generate_summary(players[1],s2,w2))
            # st.markdown("**Strengths:** " + (", ".join(s2) if s2 else "None"))
            # st.markdown("**Weaknesses:** " + (", ".join(w2) if w2 else "None"))        