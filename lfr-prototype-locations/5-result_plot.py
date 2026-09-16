import pandas as pd
import plotly.graph_objects as go
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_CSV = "lfr-prototype-locations/2-test_cases.csv"

OUTPUT_HTML = (
    "lfr-prototype-locations/"
    "loss_comparison_interactive-2.html"
)

TEST_COL = "Test"

LOSS_COLUMNS = [
    "Lz",
    "Lx",
    "Ly",
    "L"
]


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULT_CSV)

# Keep only the columns needed for visualization
plot_df = df[
    [TEST_COL] + LOSS_COLUMNS
].copy()


# ============================================================
# CLEAN EXPERIMENT NAMES
# ============================================================

plot_df[TEST_COL] = (
    plot_df[TEST_COL]
    .astype(str)
    .str.strip()
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig = go.Figure()


# ============================================================
# COLORS
#
# One color for each loss.
# You can change these later if desired.
# ============================================================

loss_colors = {
    "Lz": "#3BA3C3",
    "Lx": "#E7B5B5",
    "Ly": "#BFC5CE",
    "L": "#8C78B8"
}


# ============================================================
# FULL-OPACITY COLORS
# ============================================================

for loss_name in LOSS_COLUMNS:

    values = plot_df[
        loss_name
    ].tolist()

    experiments = plot_df[
        TEST_COL
    ].tolist()


    fig.add_trace(

        go.Bar(

            name=loss_name,

            x=experiments,

            y=values,

            marker_color=loss_colors[
                loss_name
            ],

            # -----------------------------------------------
            # EXACT VALUES ABOVE BARS
            # -----------------------------------------------

            text=[
                f"{value:.4f}"
                for value in values
            ],

            textposition="outside",

            textfont=dict(
                size=10
            ),

            # -----------------------------------------------
            # HOVER INFORMATION
            # -----------------------------------------------

            customdata=experiments,

            hovertemplate=(
                "<b>%{x}</b><br>"
                + loss_name
                + ": %{y:.6f}"
                + "<extra></extra>"
            )
        )
    )


# ============================================================
# FIGURE LAYOUT
# ============================================================

fig.update_layout(

    title=dict(
        text=(
            "LFR Prototype Experiment "
            "Loss Comparison"
        ),
        x=0.5
    ),

    # --------------------------------------------------------
    # GROUPED COLUMN CHART
    # --------------------------------------------------------

    barmode="group",

    bargap=0.20,

    bargroupgap=0.05,


    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    xaxis=dict(

        title="Experiment",

        tickangle=-35,

        automargin=True
    ),

    yaxis=dict(

        title="Loss Value",

        rangemode="tozero",

        gridcolor="rgba(0,0,0,0.12)",

        zeroline=True
    ),


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend=dict(

        title="Loss",

        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="center",

        x=0.5
    ),


    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    width=1600,

    height=750,


    # --------------------------------------------------------
    # MARGINS
    # --------------------------------------------------------

    margin=dict(
        l=80,
        r=50,
        t=120,
        b=180
    )
)


# ============================================================
# CREATE HTML
# ============================================================
#
# Plotly's normal Python click callbacks only work while
# running a Dash/Jupyter application.
#
# Because we want a standalone HTML file, we add a small
# JavaScript click handler.
#
# ============================================================

html = fig.to_html(

    full_html=True,

    include_plotlyjs=True,

    div_id="loss-chart"
)


# ============================================================
# INTERACTIVE CLUSTER SELECTION JAVASCRIPT
# ============================================================
#
# Behavior:
#
# 1. Click any bar in an experiment:
#       -> select that entire experiment
#
# 2. Click another experiment:
#       -> both remain selected
#
# 3. Unselected experiments fade
#
# 4. Click selected experiment again:
#       -> deselect it
#
# 5. Reset button:
#       -> restore all experiments
#
# ============================================================

experiments_json = json.dumps(
    plot_df[TEST_COL].tolist()
)

losses_json = json.dumps(
    LOSS_COLUMNS
)

colors_json = json.dumps(
    loss_colors
)


javascript = f"""

<script>

const chart = document.getElementById(
    "loss-chart"
);

const experiments = {experiments_json};

const losses = {losses_json};

const baseColors = {colors_json};


// ============================================================
// SELECTED EXPERIMENTS
// ============================================================

let selectedExperiments = new Set();


// ============================================================
// OPACITY SETTINGS
// ============================================================

const SELECTED_OPACITY = 1.0;

const FADED_OPACITY = 0.18;


// ============================================================
// CREATE RESET BUTTON
// ============================================================

const resetButton = document.createElement(
    "button"
);

resetButton.innerText = "Reset selection";

resetButton.style.position = "fixed";
resetButton.style.top = "15px";
resetButton.style.right = "20px";

resetButton.style.padding = "9px 16px";

resetButton.style.fontSize = "14px";

resetButton.style.border = (
    "1px solid #888"
);

resetButton.style.borderRadius = "6px";

resetButton.style.backgroundColor = "white";

resetButton.style.cursor = "pointer";

resetButton.style.zIndex = "1000";

document.body.appendChild(
    resetButton
);


// ============================================================
// STATUS TEXT
// ============================================================

const statusText = document.createElement(
    "div"
);

statusText.style.position = "fixed";

statusText.style.top = "20px";

statusText.style.left = "20px";

statusText.style.fontFamily = (
    "Arial, sans-serif"
);

statusText.style.fontSize = "14px";

statusText.style.backgroundColor = (
    "rgba(255,255,255,0.9)"
);

statusText.style.padding = "6px 10px";

statusText.style.borderRadius = "5px";

statusText.style.zIndex = "1000";

statusText.innerText = (
    "Click an experiment cluster to highlight it"
);

document.body.appendChild(
    statusText
);


// ============================================================
// UPDATE CHART
// ============================================================

function updateChart() {{

    const hasSelection =
        selectedExperiments.size > 0;


    // --------------------------------------------------------
    // Loop through four loss traces
    // --------------------------------------------------------

    for (
        let traceIndex = 0;
        traceIndex < losses.length;
        traceIndex++
    ) {{

        const lossName =
            losses[traceIndex];

        const baseColor =
            baseColors[lossName];


        // ----------------------------------------------------
        // Build opacity for every experiment/bar
        // ----------------------------------------------------

        const opacityArray =
            experiments.map(
                experiment => {{

                    if (!hasSelection) {{
                        return 1.0;
                    }}

                    if (
                        selectedExperiments.has(
                            experiment
                        )
                    ) {{
                        return SELECTED_OPACITY;
                    }}

                    return FADED_OPACITY;
                }}
            );


        // ----------------------------------------------------
        // Build colors
        // ----------------------------------------------------

        const colorArray =
            experiments.map(
                () => baseColor
            );


        // ----------------------------------------------------
        // Apply changes
        // ----------------------------------------------------

        Plotly.restyle(

            chart,

            {{
                "marker.color": [
                    colorArray
                ],

                "marker.opacity": [
                    opacityArray
                ]
            }},

            [traceIndex]
        );
    }}


    // --------------------------------------------------------
    // UPDATE STATUS
    // --------------------------------------------------------

    if (!hasSelection) {{

        statusText.innerText =
            "Click an experiment cluster to highlight it";

    }}

    else {{

        statusText.innerText =
            "Selected: "
            + Array.from(
                selectedExperiments
            ).join(", ");

    }}
}}


// ============================================================
// CLICK EVENT
// ============================================================

chart.on(
    "plotly_click",

    function(eventData) {{

        if (
            !eventData.points
            || eventData.points.length === 0
        ) {{
            return;
        }}


        // ----------------------------------------------------
        // Get experiment name from clicked bar
        // ----------------------------------------------------

        const experiment =
            eventData.points[0].x;


        // ----------------------------------------------------
        // Toggle experiment
        // ----------------------------------------------------

        if (
            selectedExperiments.has(
                experiment
            )
        ) {{

            selectedExperiments.delete(
                experiment
            );

        }}

        else {{

            selectedExperiments.add(
                experiment
            );

        }}


        // ----------------------------------------------------
        // REDRAW
        // ----------------------------------------------------

        updateChart();
    }}
);


// ============================================================
// RESET
// ============================================================

resetButton.addEventListener(
    "click",

    function() {{

        selectedExperiments.clear();

        updateChart();
    }}
);

</script>

"""


# ============================================================
# INSERT JAVASCRIPT INTO HTML
# ============================================================

html = html.replace(
    "</body>",
    javascript + "</body>"
)


# ============================================================
# SAVE HTML
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_HTML
    ),
    exist_ok=True
)

with open(
    OUTPUT_HTML,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        html
    )


# ============================================================
# FINISHED
# ============================================================

print(
    f"Interactive plot saved to:"
)

print(
    OUTPUT_HTML
)

print()

print(
    f"Experiments plotted: "
    f"{len(plot_df)}"
)

print(
    "Click experiment bars to select/deselect clusters."
)
