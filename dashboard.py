import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ast  # Required to safely parse stringified Python dicts
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load data
file_path = 'PremCC - Selection.xlsx'
sheet_name = 'Eye-Anticipation-HandEye-Memory'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Clean column names
df.columns = df.columns.str.strip()

# Pre-computed metrics
df["Dynamic Average"] = df[["Dynamic 1", "Dynamic 2", "Dynamic 3"]].mean(axis=1).round(1)
df["Visual Acuity"] = df["Visual Acuity"].round(1)

# Set up the app
st.title("Prem CC - Player Dashboard")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["👁️ Visual Attributes", "⚡ Reactions", "🎯 Hand-Eye Coordination","🧠 Anticipation","📊 Player Profiling"])

 
# ---------------------- TAB 1: Visual Attributes ---------------------- #
with tab1:
    st.header("Visual Attribute Metrics")
    sort_order = st.radio("View", ["Top Performers", "Bottom Performers"], horizontal=True)

    def rank_df(metric_col, ascending=True, round_decimals=None):
        temp_df = df[['Name', metric_col]].copy()
        if round_decimals is not None:
            temp_df[metric_col] = temp_df[metric_col].round(round_decimals)
        temp_df = temp_df.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True)
        return temp_df

    # Visual Acuity (lower is better)
    st.subheader("Visual Acuity (lower is better)")
    sorted_va = rank_df("Visual Acuity", ascending=True)
    sorted_va["Visual Acuity"] = sorted_va["Visual Acuity"].map(lambda x: f"{x:.1f}")
    st.table(sorted_va.head(5) if sort_order == "Top Performers" else sorted_va.tail(5))

    # Dynamic Vision (lower is better)
    st.subheader("Dynamic Vision (average of 3 tests, lower is better)")
    sorted_dv = rank_df("Dynamic Average", ascending=True)
    sorted_dv["Dynamic Average"] = sorted_dv["Dynamic Average"].map(lambda x: f"{x:.1f}")
    st.table(sorted_dv.head(5) if sort_order == "Top Performers" else sorted_dv.tail(5))

    # Contrast Sensitivity (higher is better)
    st.subheader("Contrast Sensitivity (higher is better)")
    sorted_cs = rank_df("Contrast", ascending=False)
    sorted_cs["Contrast"] = sorted_cs["Contrast"].map(lambda x: f"{x:.2f}")
    st.table(sorted_cs.head(5) if sort_order == "Top Performers" else sorted_cs.tail(5))

    # Color Vision Deficiency
    st.subheader("Color Vision Deficiency")
    color_deficient = df[df['Color'] == 'Deficient']
    percent = round(len(color_deficient) / len(df) * 100, 1)
    st.metric("Proportion of Color Deficient Players", f"{percent}%")

# ---------------------- TAB 2: Reactions ---------------------- #

with tab2:
    st.header("Reaction & Decision Metrics")

    # Reaction Speed columns
    reaction_cols = [f'Reaction Speed {i}' for i in range(1, 6)]

    # Replace outliers > 1s with player-specific max ≤ 1s
    def replace_outliers_with_row_max(row):
        valid_vals = [x for x in row if x <= 1.0]
        row_max = max(valid_vals) if valid_vals else 1.0
        return [x if x <= 1.0 else row_max for x in row]

    df[reaction_cols] = df[reaction_cols].apply(replace_outliers_with_row_max, axis=1, result_type='expand')
    df['Reaction Speed Avg'] = df[reaction_cols].mean(axis=1)

    # Decision Speed columns
    decision_speed_cols = [f'Decision Making Speed {i}' for i in range(1, 6)]
    df['Decision Speed Avg'] = df[decision_speed_cols].mean(axis=1)

    # Parse accuracy (wrong/total) format
    def parse_wrong_accuracy(val):
        try:
            wrong, total = map(int, str(val).split('/'))
            correct = total - wrong
            return correct, total
        except:
            return 0, 0

    corrects, totals = [], []
    for col in [f'Decision Making Accuracy {i}' for i in range(1, 6)]:
        parsed = df[col].apply(parse_wrong_accuracy)
        df[f'{col} Correct'] = parsed.apply(lambda x: x[0])
        df[f'{col} Total'] = parsed.apply(lambda x: x[1])
        corrects.append(f'{col} Correct')
        totals.append(f'{col} Total')

    df['Correct Decisions'] = df[corrects].sum(axis=1)
    df['Total Decisions'] = df[totals].sum(axis=1)
    df['Wrong Decisions'] = df['Total Decisions'] - df['Correct Decisions']

    # Decision Making Latency
    df['Decision Making Latency (ms)'] = (df['Decision Speed Avg'] - df['Reaction Speed Avg']) * 1000

    # Final table (convert times to ms)
    reactions_df = df[['Name']].copy()
    reactions_df['Reaction Speed Avg (ms)'] = (df['Reaction Speed Avg'] * 1000).round(1)
    reactions_df['Decision Speed Avg (ms)'] = (df['Decision Speed Avg'] * 1000).round(1)
    reactions_df['Decision Making Latency (ms)'] = df['Decision Making Latency (ms)'].round(0)
    reactions_df['Wrong Decisions'] = df['Wrong Decisions']

    st.subheader("Summary Table")
    st.dataframe(reactions_df.sort_values(by='Wrong Decisions'))

# ---------------------- TAB 3: Hand-eye ---------------------- # 
with tab3:
    st.header("Trajectory Paths & Completion Times")

    # Player selection filter
    selected_players = st.multiselect(
        "Select players to compare:",
        options=df["Name"].dropna().unique(),
        default=df["Name"].dropna().unique()[:3]  # default to first 3 players
    )

    if not selected_players:
        st.warning("Please select at least one player to visualize.")
    else:
        time_cols = {
            "Square": "Hand-eye-coordination time-Square",
            "Line": "Hand-eye-coordination time-Line",
            "Circle": "Hand-eye-coordination time-Circle",
            "Curved": "Hand-eye-coordination time-Curved"
        }

        coord_cols = {
            "Square": "Hand-eye-coordinates-Square",
            "Line": "Hand-eye-coordinates-Line",
            "Circle": "Hand-eye-coordinates-Circle",
            "Curved": "Hand-eye-coordinates-Curved"
        }

        for level in ["Square", "Line", "Circle", "Curved"]:
            with st.expander(f"🧩 {level} Obstacle"):
                fig, ax = plt.subplots(figsize=(10, 5))
                for _, row in df[df["Name"].isin(selected_players)].iterrows():
                    coords_raw = row.get(coord_cols[level], None)
                    if pd.notna(coords_raw):
                        try:
                            coords_list = ast.literal_eval(coords_raw)
                            path_coords = [tuple(item["py/tuple"]) for item in coords_list]
                            x_coords, y_coords = zip(*path_coords)
                            ax.plot(x_coords, y_coords, marker='o', linestyle='-', label=row["Name"])
                        except Exception:
                            continue

                ax.set_title(f"{level} - Path")
                ax.set_xlabel("X Coordinates")
                ax.set_ylabel("Y Coordinates")
                ax.grid(True)
                ax.axis('equal')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)

                # Completion time table
                time_col = time_cols[level]
                if time_col in df.columns:
                    times_df = df[df["Name"].isin(selected_players)][["Name", time_col]].copy()
                    times_df.columns = ["Name", "Completion Time (s)"]
                    st.dataframe(times_df.sort_values(by="Completion Time (s)"))

# ---------------------- TAB 4: Anticipation ---------------------- #
with tab4:
    st.header("Anticipation Visualization")

    # Standard base trajectories (4 linear + 2 sine) with time scaled from 0 to 12 seconds
    base_trajectories = {
        "Linear 1": (np.linspace(0, 12, 100), np.linspace(0, 1, 100)),
        "Linear 2": (np.linspace(0, 12, 100), np.linspace(1, 0, 100)),
        "Linear 3": (np.linspace(0, 12, 100), np.linspace(0.2, 0.8, 100)),
        "Linear 4": (np.linspace(0, 12, 100), np.linspace(0.8, 0.2, 100)),
        "Sin 1": (np.linspace(0, 12, 100), np.sin(4 * np.linspace(0, np.pi, 100))),
        "Sin 2": (np.linspace(0, 12, 100), 2 * np.sin(6 * np.linspace(0, np.pi, 100)))
    }

    # Mapping
    anticipation_tests = {
        "Linear 1": ("Anticipation Distance-Linear 1", "Anticipation Time-Linear 1"),
        "Linear 2": ("Anticipation Distance-Linear 2", "Anticipation Time-Linear 2"),
        "Linear 3": ("Anticipation Distance-Linear 3", "Anticipation Time-Linear 3"),
        "Linear 4": ("Anticipation Distance-Linear 4", "Anticipation Time-Linear 4"),
        "Sin 1": ("Anticipation Distance-Sin 1", "Anticipation Time-Sin 1"),
        "Sin 2": ("Anticipation Distance-Sin 2", "Anticipation Time-Sin 2")
    }

    # Player filter
    selected_players = st.multiselect("Select players to compare", df['Name'].unique())
    filtered_df = df[df['Name'].isin(selected_players)] if selected_players else df

    color_map = plt.colormaps['tab10']
    player_colors = {name: color_map(i % 10) for i, name in enumerate(filtered_df['Name'].unique())}

    for traj_label, (dist_col, time_col) in anticipation_tests.items():
        with st.expander(f"\U0001F4C8 {traj_label} Trajectory"):
            base_x, base_y = base_trajectories[traj_label]
            fig, ax = plt.subplots(figsize=(8, 4))

            # Plot dashed line for occlusion zone
            occlusion_start_index = int(0.5 * len(base_x))  # 50% of 12s = 6s
            ax.plot(base_x[:occlusion_start_index], base_y[:occlusion_start_index], 'k-', linewidth=1, label='_nolegend_')
            ax.plot(base_x[occlusion_start_index:], base_y[occlusion_start_index:], 'k--', linewidth=1, label='_nolegend_')

            any_points = False
            for _, row in filtered_df.iterrows():
                try:
                    dist_val = row[dist_col]
                    time_val = row[time_col]
                    if pd.notna(dist_val) and pd.notna(time_val):
                        dist_err = float(dist_val)  # in cm
                        time_err = float(time_val)  # in seconds

                        predicted_time = 12 + time_err  # calculate from the target point
                        predicted_index = int(np.clip(predicted_time / 12 * len(base_x), 0, len(base_x) - 1))
                        predicted_x = base_x[predicted_index]
                        predicted_y = base_y[predicted_index] + dist_err / 100  # convert cm to meters

                        ax.plot(predicted_x, predicted_y, 'o', markersize=8, label=row['Name'],
                                color=player_colors.get(row['Name'], 'gray'))
                        any_points = True
                except Exception as e:
                    st.warning(f"Error plotting for {row['Name']}: {e}")
                    continue

            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            ax.set_title(f"{traj_label} Anticipation")
            if any_points:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

# ---------------------- TAB 5: Player Profiling ---------------------- #
with tab5:
    st.header("Player Profiling Summary (Normalized)")

    # --- Score Functions --- #
    def compute_visual_score(row):
        acuity = 1 / row["Visual Acuity"] if row["Visual Acuity"] > 0 else 0
        dyn_vis = 1 / row["Dynamic Average"] if row["Dynamic Average"] > 0 else 0
        contrast = row["Contrast"]
        color_score = 1 if row["Color"] == "Normal" else 0
        return 0.45 * acuity + 0.40 * dyn_vis + 0.10 * contrast + 0.05 * color_score

    def compute_reaction_score(row):
        reaction = 1 / row["Reaction Speed Avg"] if row["Reaction Speed Avg"] > 0 else 0
        decision = 1 / row["Decision Speed Avg"] if row["Decision Speed Avg"] > 0 else 0
        mistakes = 1 / (1 + row["Wrong Decisions"])
        return 0.40 * reaction + 0.40 * decision + 0.20 * mistakes

    def compute_hec_score(row):
        total = 0
        for col in [
            "Hand-eye-coordination time-Square",
            "Hand-eye-coordination time-Line",
            "Hand-eye-coordination time-Circle",
            "Hand-eye-coordination time-Curved"
        ]:
            score = 1 / row[col] if row[col] > 0 else 0
            total += 0.25 * score
        return total

    def compute_anticipation_score(row):
        safe = lambda x: 1 / max(abs(x), 0.1)  # prevent over-rewarding tiny errors
        return (
            0.15 * safe(row["Anticipation Time-Linear 1"]) +
            0.15 * safe(row["Anticipation Time-Linear 2"]) +
            0.15 * safe(row["Anticipation Time-Linear 3"]) +
            0.15 * safe(row["Anticipation Time-Linear 4"]) +
            0.20 * safe(row["Anticipation Time-Sin 1"]) +
            0.20 * safe(row["Anticipation Time-Sin 2"])
        )



    def compute_overall_score(row):
        return (
            0.20 * row["Visual Score"] +
            0.20 * row["Reaction Score"] +
            0.20 * row["HEC Score"] +
            0.20 * row["Anticipation Score"] +
            0.20 * row["Memory Score"]
        )

    # --- Compute Scores --- #
    df["Visual Score"] = df.apply(compute_visual_score, axis=1)
    df["Reaction Score"] = df.apply(compute_reaction_score, axis=1)
    df["HEC Score"] = df.apply(compute_hec_score, axis=1)
    df["Anticipation Score"] = df.apply(compute_anticipation_score, axis=1)
    df["Memory Score"] = df["Short Term memory"]
    df["Overall Score"] = df.apply(compute_overall_score, axis=1)

    # --- Player Comparison Dropdowns --- #
    st.subheader("Compare Players")
    col1, col2 = st.columns(2)
    with col1:
        player_1 = st.selectbox("Select Player 1", df["Name"].unique(), key="player1")
    with col2:
        player_2 = st.selectbox("Select Player 2", df["Name"].unique(), key="player2")

    player1_data = df[df["Name"] == player_1].iloc[0]
    player2_data = df[df["Name"] == player_2].iloc[0]

    # --- Radar Chart --- #
    categories = ['Visual Score', 'Reaction Score', 'HEC Score', 'Anticipation Score', 'Memory Score']
    values_1 = [player1_data[cat] for cat in categories] + [player1_data[categories[0]]]
    values_2 = [player2_data[cat] for cat in categories] + [player2_data[categories[0]]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values_1, label=player_1, linewidth=2)
    ax.fill(angles, values_1, alpha=0.25)

    if player_1 != player_2:
        ax.plot(angles, values_2, label=player_2, linewidth=2, linestyle='dashed')
        ax.fill(angles, values_2, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)

    # --- Detailed Metric Table for Player 1 --- #
    st.subheader(f"Detailed Metrics: {player_1}")
    metrics_dict = {
        "Visual Acuity": player1_data["Visual Acuity"],
        "Dynamic Visual Acuity": player1_data["Dynamic Average"],
        "Contrast Sensitivity": player1_data["Contrast"],
        "Color Perception": player1_data["Color"],
        "Reaction Time (s)": player1_data["Reaction Speed Avg"],
        "Decision Time (s)": player1_data["Decision Speed Avg"],
        "Wrong Decisions": player1_data["Wrong Decisions"],
        "HEC Time - Square": player1_data["Hand-eye-coordination time-Square"],
        "HEC Time - Line": player1_data["Hand-eye-coordination time-Line"],
        "HEC Time - Circle": player1_data["Hand-eye-coordination time-Circle"],
        "HEC Time - Curved": player1_data["Hand-eye-coordination time-Curved"],
        "Anticipation Time - Linear 1": player1_data["Anticipation Time-Linear 1"],
        "Anticipation Time - Linear 2": player1_data["Anticipation Time-Linear 2"],
        "Anticipation Time - Linear 3": player1_data["Anticipation Time-Linear 3"],
        "Anticipation Time - Linear 4": player1_data["Anticipation Time-Linear 4"],
        "Anticipation Time - Sin 1": player1_data["Anticipation Time-Sin 1"],
        "Anticipation Time - Sin 2": player1_data["Anticipation Time-Sin 2"],
        "Short-term Memory (score)": player1_data["Short Term memory"]
    }
    metrics_df = pd.DataFrame(metrics_dict.items(), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df["Value"].astype(str)  # force all to string
    st.dataframe(metrics_df, use_container_width=True)


    # --- Player Rankings with Medals --- #
    st.subheader("Player Rankings 🏅")

    rank_df = df[["Name", "Overall Score"]].sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
    medals = ["🥇", "🥈", "🥉"]
    rank_df.insert(0, "🏆", [medals[i] if i < 3 else "" for i in range(len(rank_df))])
    st.dataframe(rank_df)
