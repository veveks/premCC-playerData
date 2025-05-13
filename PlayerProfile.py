import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

st.set_page_config(page_title="Player Dashboard", layout="wide")
st.title("Prem CC Player Dashboard")

latest_data, full_data = load_data()

# Convert to tabs instead of radio
tab1, tab2, tab3 = st.tabs(["Player Deepdive", "Compare Players", "Player Progression"])

with tab1:
    st.header("Current Player Status")
    selected_player = st.selectbox("Select a Player", latest_data['Name'].unique(), key="current")
    player_data = latest_data[latest_data['Name'] == selected_player].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radar Profile")
        fig = go.Figure(data=[radar_chart(player_data, selected_player)])
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=False)
        st.plotly_chart(fig, key=f"radar_chart_{selected_player}")
    with col2:
        st.subheader("Category-wise Scores")
        df = player_data[[
            'Vision Score', 'Reaction Time Avg', 'Reaction Score',
            'Decision Speed Subscore', 'Decision Accuracy Subscore', 'Decision Latency Subscore', 'Decision Making Score',
            'Cognitive Score', 'Hand-Eye Score', 'Anticipation Score']].T
        df.columns = [selected_player]
        st.dataframe(df)

    st.subheader("Summary Table")
    summary = pd.DataFrame({
        'Category': ['Vision', 'Reaction', 'Decision Making', 'Cognitive', 'Hand-Eye', 'Anticipation'],
        'Score (/5)': [
            player_data['Vision Score'],
            player_data['Reaction Score'],
            player_data['Decision Making Score'],
            player_data['Cognitive Score'],
            player_data['Hand-Eye Score'],
            player_data['Anticipation Score']
        ]
    })
    st.table(summary.set_index('Category'))

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
        

with tab3:
    st.header("Player Progression Over Time")
    selected_player = st.selectbox("Choose Player", full_data['Name'].unique(), key="progression")
    history = full_data[full_data['Name'] == selected_player]
    st.line_chart(history.set_index('Attempt')[[
        'Vision Score', 'Reaction Score', 'Decision Making Score',
        'Cognitive Score', 'Hand-Eye Score', 'Anticipation Score'
    ]])
