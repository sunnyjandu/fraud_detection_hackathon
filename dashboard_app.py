#import libraries
import streamlit as st        # For building the web app
import pandas as pd           # For data manipulation
import numpy as np            # For numerical operations
import plotly.express as px   # For simple charts
import plotly.graph_objects as go  # For advanced charts
from plotly.subplots import make_subplots  # For multi-panel plots
from scipy import stats       # For statistical tests


# Set up the Streamlit page layout and appearance
st.set_page_config(
    page_title="NovaPay · Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",  # Use full width of screen
    initial_sidebar_state="expanded",  # Show sidebar by default
)

# Custom CSS to make the dashboard look professional

st.markdown("""
<style>
/* Use standard system fonts for performance and reliability */
html, body, [class*="css"]       { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
h1, h2, h3, h4                   { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important; }
.stApp                           { background-color: #080d18; color: #e2e8f0; }
section[data-testid="stSidebar"] { background-color: #0b1120; border-right: 1px solid #1a2d47; }

/* KPI Card - individual metric boxes */
.kpi-card {
    background: linear-gradient(135deg, #0f1e38 0%, #091526 100%);
    border: 1px solid #1a3050; border-radius: 14px;
    padding: 20px 22px 16px; position: relative; overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 120px; /* fixed height to make all cards equal */
}
.kpi-card::before {
    content:''; position:absolute; top:0; left:0;
    width:4px; height:100%; background:#00e5ff; border-radius:4px 0 0 4px;
}
/* Color variants for different KPI types */
.kpi-card.red::before    { background:#ff4560; }
.kpi-card.amber::before  { background:#ffb300; }
.kpi-card.green::before  { background:#00e676; }
.kpi-card.purple::before { background:#a78bfa; }

/* KPI text styling */
.kpi-label { font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#4a6280; margin-bottom:6px; text-align:center; }
.kpi-value { font-family:system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size:24px; font-weight:800; color:#f0f6ff; line-height:1.2; text-align:center; word-break:break-word; }
.kpi-sub   { font-size:10px; color:#3d5570; margin-top:5px; text-align:center; }


/* Section headers */
.sec-head {
    font-size:10px; letter-spacing:3px; text-transform:uppercase;
    color:#00e5ff; border-bottom:1px solid #1a3050;
    padding-bottom:7px; margin: 28px 0 16px;
}

/* Badge/chip styling for status indicators */
.chip        { display:inline-block; border-radius:5px; padding:2px 9px; font-size:10px; letter-spacing:1px; }
.chip-blue   { background:#00e5ff18; color:#00e5ff; border:1px solid #00e5ff33; }
.chip-red    { background:#ff456018; color:#ff4560; border:1px solid #ff456033; }
.chip-green  { background:#00e67618; color:#00e676; border:1px solid #00e67633; }
.chip-amber  { background:#ffb30018; color:#ffb300; border:1px solid #ffb30033; }

/* Hypothesis test card styling */
.hyp-card {
    background:#0b1929; border:1px solid #1a3050;
    border-radius:12px; padding:18px 20px; margin-bottom:12px;
}
.hyp-title  { font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:#e2e8f0; margin:8px 0 4px; }
.hyp-body   { font-size:11px; color:#4a6280; line-height:1.65; }
.hyp-finding{ font-size:11px; color:#94a3b8; margin-top:10px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background:#0b1120;
    border-bottom:1px solid #1a2d47;
    gap:6px;
    padding:2px 0;
    flex-wrap:nowrap;
    width:100%;
}
.stTabs [data-baseweb="tab"] {
    font-family:'DM Mono',monospace;
    font-size:11px;
    letter-spacing:1px;
    color:#64748b;
    background:#0b1120;
    border:1px solid #1a2d47;
    padding:6px 8px;
    border-radius:8px;
    flex:1 1 0;
    justify-content:center;
}
.stTabs [data-baseweb="tab"]:hover { color:#94a3b8; border-color:#2b4669; }
.stTabs [aria-selected="true"] {
    color:#00e5ff !important;
    background:#080d18 !important;
    border:1px solid #1a3050 !important;
    box-shadow: inset 0 -2px 0 #00e5ff;
}

/* Chart container styling */
div[data-testid="stPlotlyChart"]  {
    border:1px solid #1a3050;
    border-radius:12px;
    overflow:hidden;
    margin-top:8px;
    margin-bottom:18px;
}

/* Give metric rows and info blocks breathing room */
div[data-testid="stMetric"] { margin-bottom:10px; }
div[data-testid="stAlert"]  { margin:8px 0 14px; }
</style>
""", unsafe_allow_html=True)


# Path to the fraud dataset (must be relative to where you run the app)
DATA_PATH = r".\data\processed\dashboard.csv"

# This function uses @st.cache_data to load the data only once
# It remembers the data so reloads are instant (faster dashboard)
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the fraud dataset and prepare it for analysis."""
    df = pd.read_csv(path)
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Convert fraud columns to integer format (0 or 1)
    for col in ["is_fraud", "predicted_fraud"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    return df

# Try to load the data, show error message if file not found
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"⚠️  Dataset not found at **{DATA_PATH}**.\n\n"
        "Make sure `dashboard.csv` lives at `data/processed/` "
        "relative to where you run the app."
    )
    st.stop()  # Stop the app if data is missing



# These colors are used throughout all charts for consistency
COLORS = {
    "fraud": "#ff4560",   # Red for fraud
    "legit": "#00e5ff",   # Cyan for legitimate
    "amber": "#ffb300",   # Orange/amber for warnings
    "green": "#00e676",   # Green for positive metrics
    "purple": "#a78bfa"   # Purple for other metrics
}

# Chart styling configuration
BG = "rgba(0,0,0,0)"              # Transparent background
GRID = "#112035"                  # Grid color
FONT = "DM Mono, monospace"       # Font family for charts

def base_layout(**kwargs):
    """
    Create a standard Plotly layout with consistent styling.
    This ensures all charts look the same throughout the dashboard.
    """
    d = dict(
        paper_bgcolor=BG,  # Transparent background
        plot_bgcolor=BG,   # Transparent plot area
        font=dict(family=FONT, color="#64748b", size=11),  # Standard font
        margin=dict(l=12, r=12, t=40, b=12),  # Spacing around chart
        legend=dict(bgcolor=BG, bordercolor="#1a3050", borderwidth=1),  # Legend styling
        xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),  # X-axis styling
        yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),  # Y-axis styling
    )
    d.update(kwargs)  # Apply any custom overrides
    return d


# The sidebar (left panel) contains all user controls
with st.sidebar:
    # Header
    st.markdown('<div class="sec-head">🛡 NOVAPAY</div>', unsafe_allow_html=True)
    st.markdown("**Fraud Intelligence Platform**")
    st.markdown('<p style="color:#2d4a6a;font-size:10px;margin-bottom:24px;">Credit Card Fraud Detection</p>',
                unsafe_allow_html=True)

    # Filter section - User can select which data to view
    st.markdown('<div class="sec-head">FILTERS</div>', unsafe_allow_html=True)

    # Filter by merchant category using a dropdown (single select plus All)
    all_cats = sorted(df["category"].unique().tolist())
    cat_options = ["<ALL>"] + all_cats
    choice = st.selectbox("Merchant Category", cat_options)
    if choice == "<ALL>":
        sel_cats = all_cats
    else:
        sel_cats = [choice]

    # Filter by gender (if column exists)
    if "gender" in df.columns:
        all_genders = sorted(df["gender"].unique().tolist())
        sel_genders = st.multiselect("Gender", all_genders, default=all_genders)
    else:
        sel_genders = None

    # Filter by time of day (0-23 hours)
    hour_range = st.slider("Transaction Hour", 0, 23, (0, 23))
    
    # Filter by age range
    age_range = st.slider(
        "Cardholder Age",
        int(df["age"].min()), 
        int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max()))
    )

    # Model threshold - adjust sensitivity of fraud detection
    st.markdown('<div class="sec-head">MODEL</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Fraud Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Lower threshold = more fraud alerts (more false positives)"
    )

    # Data source information
    st.markdown('<div class="sec-head">SOURCE</div>', unsafe_allow_html=True)
    st.caption(DATA_PATH)
    st.caption(f"Total rows: {len(df):,}")

    # Allow user to upload additional transactions CSV
    st.markdown('<div class="sec-head">UPLOAD</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload more transactions (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            df = pd.concat([df, new_df], ignore_index=True)
            st.success(f"Added {len(new_df)} rows, new total {len(df)}")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    # Provide CSV template for correct structure
    st.markdown('<div class="sec-head">TEMPLATE</div>', unsafe_allow_html=True)
    template_df = df.head(0)  # empty frame with correct columns
    csv_string = template_df.to_csv(index=False)
    st.download_button(
        label="Download CSV template",
        data=csv_string,
        file_name="dashboard_template.csv",
        mime="text/csv",
    )


# Create a boolean mask based on all selected filters
mask = (
    df["category"].isin(sel_cats) &  # Category filter
    df["trans_hour"].between(*hour_range) &  # Hour filter
    df["age"].between(*age_range)  # Age filter
)
# Also apply gender filter if it exists
if sel_genders:
    mask &= df["gender"].isin(sel_genders)

# Create filtered dataset
dff = df[mask].copy()

# Apply fraud threshold: mark transactions with high fraud probability as fraud
dff["predicted_fraud"] = (dff["fraud_probability"] >= threshold).astype(int)


# Basic transaction statistics
n_total = len(dff)  # Total transactions in filtered data
n_fraud = int(dff["is_fraud"].sum())  # Number of confirmed fraudulent transactions
fraud_rate = n_fraud / n_total * 100 if n_total else 0  # What % are fraud?
total_amt = dff["amt"].sum()  # Total transaction amount
fraud_amt = dff.loc[dff["is_fraud"] == 1, "amt"].sum()  # Total fraud amount

# Classification metrics
# TP = True Positive (correctly predicted fraud)
# FP = False Positive (incorrectly flagged as fraud)
# FN = False Negative (missed fraud)
# TN = True Negative (correctly identified legitimate)
tp = int(((dff["is_fraud"]==1) & (dff["predicted_fraud"]==1)).sum())
fp = int(((dff["is_fraud"]==0) & (dff["predicted_fraud"]==1)).sum())
fn = int(((dff["is_fraud"]==1) & (dff["predicted_fraud"]==0)).sum())
tn = int(((dff["is_fraud"]==0) & (dff["predicted_fraud"]==0)).sum())

# Calculate model performance metrics
precision = tp/(tp+fp) if (tp+fp)>0 else 0  # Of flagged items, how many are truly fraud?
recall = tp/(tp+fn) if (tp+fn)>0 else 0  # Of all fraud, how many did we catch?
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0  # Balance of precision and recall
accuracy = (tp+tn)/n_total if n_total else 0  # What % did we classify correctly?


# display KPI cards
def kpi(col, cls, label, value, sub):
    """
    Display a single KPI (Key Performance Indicator) card.
    
    Parameters:
    - col: Streamlit column to place the card in
    - cls: CSS class for color (empty, red, amber, green, purple)
    - label: Title of the metric
    - value: The actual metric value to display
    - sub: Subtitle or additional info
    """
    col.markdown(f"""
    <div class="kpi-card {cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# Create 6 columns for KPI cards
k1, k2, k3, k4, k5, k6 = st.columns(6)

# Display KPI cards
kpi(k1, "red",    "Fraud Transactions", f"{n_fraud:,}",        f"{fraud_rate:.2f}% of total")
kpi(k2, "amber",  "Fraud Amount",       f"${fraud_amt:,.0f}",  f"of ${total_amt:,.0f} total")
kpi(k3, "",       "Recall",             f"{recall:.1%}",       "Fraud cases caught")
kpi(k4, "green",  "Precision",          f"{precision:.1%}",    "Flagged = real fraud")
kpi(k5, "purple", "F1 Score",           f"{f1:.3f}",           f"Threshold {threshold:.2f}")
kpi(k6, "",       "Accuracy",           f"{accuracy:.1%}",     f"{n_total:,} transactions")

st.markdown("<br>", unsafe_allow_html=True)


# page title and header 
st.markdown(f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
  <div style="font-family:Syne,sans-serif;font-size:28px;font-weight:800;color:#f0f6ff;">
    🛡️ NovaPay Fraud Intelligence
  </div>
  <span class="chip chip-blue">LIVE DASHBOARD</span>
</div>
<p style="color:#2d4a6a;font-size:11px;margin-bottom:26px;">
  Filtered transactions: <b style="color:#00e5ff">{n_total:,}</b> &nbsp;|&nbsp; Threshold: <b style="color:#ffb300">{threshold:.2f}</b>
</p>
""", unsafe_allow_html=True)

# Create 5 tabs for different analysis views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview",      # General statistics
    "⏰  Time of Day",  # Time-based analysis
    "🏪  Merchant Category",  # Category analysis
    "💰  Amount & Distance",  # Transaction features
    "🤖  Model Performance",  # Model metrics
])


# TAB 1: OVERVIEW
# This tab shows general statistics about fraud in the dataset
with tab1:
    st.markdown('<div class="sec-head">TRANSACTION OVERVIEW</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Hour chart: volume + fraud rate
    with c1:
        hourly = dff.groupby("trans_hour")["is_fraud"].agg(["sum", "count"]).reset_index()
        hourly.columns = ["hour", "fraud", "total"]
        hourly["rate"] = hourly["fraud"] / hourly["total"] * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=hourly["hour"], y=hourly["total"], name="Total",
            marker_color="#1a3050", opacity=0.9
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["rate"], name="Fraud Rate %",
            mode="lines+markers", line=dict(color=COLORS["fraud"], width=2.5), marker=dict(size=6)
        ), secondary_y=True)
        fig.update_layout(base_layout(
            title="Hourly Volume & Fraud Rate", height=320,
            xaxis=dict(title="Hour", tickmode="linear", dtick=1),
            legend=dict(orientation="h", y=-0.2)
        ))
        fig.update_yaxes(title_text="Transactions", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate %", ticksuffix="%", showgrid=False,
                         color=COLORS["fraud"], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Category chart: fraud rate by merchant category
    with c2:
        cat_s = (dff.groupby("category")["is_fraud"]
                 .agg(fraud_count="sum", total="count").reset_index())
        cat_s["fraud_rate"] = cat_s["fraud_count"] / cat_s["total"] * 100
        cat_s = cat_s.sort_values("fraud_rate", ascending=False).head(12)
        cat_s = cat_s.sort_values("fraud_rate", ascending=True)

        fig = go.Figure(go.Bar(
            y=cat_s["category"], x=cat_s["fraud_rate"], orientation="h",
            marker_color=COLORS["purple"], text=cat_s["fraud_rate"].map(lambda x: f"{x:.1f}%"),
            textposition="outside", name="Fraud Rate"
        ))
        fig.update_layout(base_layout(
            title="Top 12 Categories by Fraud Rate", height=320,
            xaxis=dict(title="Fraud Rate (%)", ticksuffix="%"),
            yaxis=dict(title="")
        ))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    # Gender chart
    with c3:
        if "gender" in dff.columns:
            gdf = dff.groupby("gender")["is_fraud"].agg(["sum", "count"]).reset_index()
            gdf.columns = ["gender", "fraud", "total"]
            gdf["fraud_rate"] = np.where(gdf["total"] > 0, gdf["fraud"] / gdf["total"] * 100, 0)
            overall_rate = dff["is_fraud"].mean() * 100
            max_total = gdf["total"].max() if len(gdf) else 0
            gdf["bubble_size"] = np.where(max_total > 0, 18 + (gdf["total"] / max_total) * 24, 18)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=gdf["total"],
                y=gdf["fraud_rate"],
                mode="markers+text",
                text=gdf["gender"],
                textposition="top center",
                marker=dict(
                    size=gdf["bubble_size"],
                    color=COLORS["purple"],
                    line=dict(color="#a78bfa", width=1.5),
                    opacity=0.82,
                ),
                customdata=gdf[["fraud", "total"]].values,
                hovertemplate="Gender=%{text}<br>Fraud Rate=%{y:.2f}%<br>Fraud Txns=%{customdata[0]:,}<br>Total Txns=%{customdata[1]:,}<extra></extra>",
                name="Gender"
            ))
            fig.add_hline(
                y=overall_rate,
                line_dash="dash",
                line_color=COLORS["amber"],
                annotation_text=f"Overall: {overall_rate:.2f}%",
                annotation_position="top right"
            )
            fig.update_layout(base_layout(
                title="Gender Risk: Fraud Rate vs Transaction Volume",
                height=320,
                xaxis=dict(title="Transaction Volume"),
                yaxis=dict(title="Fraud Rate (%)", ticksuffix="%")
            ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Gender column not available in dataset.")

    # Age chart
    with c4:
        age_bins = [18, 25, 35, 45, 55, 65, 100]
        age_labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        age_df = dff.copy()
        age_df["age_band"] = pd.cut(age_df["age"], bins=age_bins, labels=age_labels, right=False)

        adf = age_df.groupby("age_band", observed=False)["is_fraud"].agg(["sum", "count"]).reset_index()
        adf.columns = ["age_band", "fraud", "total"]
        adf = adf.dropna(subset=["age_band"])
        adf["fraud_rate"] = np.where(adf["total"] > 0, adf["fraud"] / adf["total"] * 100, 0)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=adf["age_band"],
            y=adf["total"],
            name="Total",
            marker_color="#1a3050",
            opacity=0.85
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=adf["age_band"],
            y=adf["fraud_rate"],
            name="Fraud Rate %",
            mode="lines+markers",
            line=dict(color=COLORS["fraud"], width=3),
            marker=dict(size=7)
        ), secondary_y=True)
        fig.update_layout(base_layout(
            title="Fraud Rate Across Age Bands",
            height=320,
            legend=dict(orientation="h", y=-0.2)
        ))
        fig.update_yaxes(title_text="Transactions", secondary_y=False)
        fig.update_yaxes(
            title_text="Fraud Rate (%)",
            ticksuffix="%",
            showgrid=False,
            color=COLORS["fraud"],
            secondary_y=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# TAB 2: TIME OF DAY ANALYSIS
# This tab tests if fraud happens more at certain times of day
with tab2:
    st.markdown('<div class="sec-head">TIME OF DAY ANALYSIS</div>', unsafe_allow_html=True)
    st.caption("Compare late-night risk with other hours, then review hourly and day-hour patterns.")

    hourly = (dff.groupby("trans_hour")["is_fraud"]
              .agg(fraud="sum", total="count")
              .reindex(range(24), fill_value=0)
              .reset_index()
              .rename(columns={"trans_hour": "hour"}))
    hourly["rate"] = np.where(hourly["total"] > 0, hourly["fraud"] / hourly["total"] * 100, 0)

    late  = dff[dff["trans_hour"].between(0,3)]
    other = dff[~dff["trans_hour"].between(0,3)]
    late_rate  = late["is_fraud"].mean()*100  if len(late)>0 else 0
    other_rate = other["is_fraud"].mean()*100 if len(other)>0 else 0
    cont = np.array([[late["is_fraud"].sum(), len(late)-late["is_fraud"].sum()],
                     [other["is_fraud"].sum(), len(other)-other["is_fraud"].sum()]])
    if cont.min() > 0:
        chi2, pval, _, _ = stats.chi2_contingency(cont)
    else:
        chi2, pval = 0.0, 1.0
    risk_gap = late_rate - other_rate

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Late-Night Fraud %", f"{late_rate:.2f}%")
    m2.metric("Other Hours Fraud %", f"{other_rate:.2f}%")
    m3.metric("Risk Gap", f"{risk_gap:+.2f} pp")
    m4.metric("p-value", f"{pval:.4f}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=hourly["hour"], y=hourly["total"], name="Total",
        marker_color="#1a3050", opacity=0.9
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["rate"], name="Fraud Rate %",
        mode="lines+markers", line=dict(color=COLORS["fraud"], width=2.5), marker=dict(size=6)
    ), secondary_y=True)
    fig.add_vrect(x0=-0.5, x1=3.5, fillcolor="rgba(255,69,96,0.08)", line_width=0)
    fig.add_vrect(x0=21.5, x1=23.5, fillcolor="rgba(255,69,96,0.08)", line_width=0)
    fig.update_layout(base_layout(
        title="Hourly Volume & Fraud Rate", height=340,
        xaxis=dict(title="Hour", tickmode="linear", dtick=1),
        legend=dict(orientation="h", y=-0.2)
    ))
    fig.update_yaxes(title_text="Transactions", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate %", ticksuffix="%", showgrid=False,
                     color=COLORS["fraud"], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    doh = dff.groupby(["trans_dayofweek", "trans_hour"])["is_fraud"].mean().reset_index()
    pivot = doh.pivot(index="trans_dayofweek", columns="trans_hour", values="is_fraud").fillna(0)
    dlabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(range(24)),
        y=[dlabels[i] for i in pivot.index],
        colorscale=[[0, "#0b1929"], [1, "#ff4560"]],
        colorbar=dict(title="Fraud Rate"),
    ))
    fig.update_layout(base_layout(title="Fraud Rate Heatmap — Day × Hour", height=280,
                                  xaxis=dict(title="Hour", tickmode="linear", dtick=1)))
    st.plotly_chart(fig, use_container_width=True)


# TAB 3: MERCHANT CATEGORY ANALYSIS
# This tab tests if certain merchant categories have more fraud
with tab3:
    st.markdown('<div class="sec-head">MERCHANT CATEGORY ANALYSIS</div>', unsafe_allow_html=True)
    st.caption("Compare category fraud risk, volume, and where risk concentrates.")

    cat_s = (dff.groupby("category")["is_fraud"]
             .agg(fraud_count="sum", total="count").reset_index())
    cat_s["fraud_rate"] = cat_s["fraud_count"] / cat_s["total"] * 100
    cat_s["legit_count"] = cat_s["total"] - cat_s["fraud_count"]
    cat_s = cat_s.sort_values("fraud_rate", ascending=True)

    cont_cat = cat_s[["fraud_count", "legit_count"]].values
    if cont_cat.min() > 0:
        chi2_c, p_c, dof_c, _ = stats.chi2_contingency(cont_cat)
    else:
        chi2_c, p_c, dof_c = 0.0, 1.0, 0
    top = cat_s.sort_values("fraud_rate", ascending=False).iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Highest Risk Category", f"{top['category']}")
    m2.metric("Top Fraud Rate", f"{top['fraud_rate']:.2f}%")
    m3.metric("p-value", f"{p_c:.4f}")
    m4.metric("DoF", f"{dof_c}")

    cat_display = cat_s.sort_values("fraud_rate", ascending=False).head(12).sort_values("fraud_rate", ascending=True)
    cat_vol = cat_s.sort_values("total", ascending=False).head(12)

    fig = go.Figure(go.Bar(
        y=cat_display["category"], x=cat_display["fraud_rate"], orientation="h",
        marker_color=COLORS["purple"],
        text=cat_display["fraud_rate"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig.update_layout(base_layout(title="Top 12 Categories by Fraud Rate", height=360,
                                  xaxis=dict(title="Fraud Rate (%)", ticksuffix="%"),
                                  yaxis=dict(title="")))
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cat_vol["category"], y=cat_vol["legit_count"],
                         name="Legitimate", marker_color=COLORS["legit"], opacity=0.8))
    fig.add_trace(go.Bar(x=cat_vol["category"], y=cat_vol["fraud_count"],
                         name="Fraudulent", marker_color=COLORS["fraud"]))
    fig.update_layout(base_layout(title="Top 12 Categories by Volume",
                                  barmode="stack", height=360,
                                  xaxis=dict(tickangle=-30),
                                  legend=dict(orientation="h", y=-0.22)))
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: TRANSACTION AMOUNT & DISTANCE ANALYSIS
# This tab tests if fraud has different amounts and distances
with tab4:
    st.markdown('<div class="sec-head">AMOUNT & DISTANCE ANALYSIS</div>', unsafe_allow_html=True)
    st.caption("Compare amount and distance behavior for fraud vs legitimate transactions.")

    fa = dff.loc[dff["is_fraud"]==1,"amt"].dropna()
    la = dff.loc[dff["is_fraud"]==0,"amt"].dropna()
    fd = dff.loc[dff["is_fraud"]==1,"home_merch_dist"].dropna()
    ld = dff.loc[dff["is_fraud"]==0,"home_merch_dist"].dropna()

    u_amt,  p_amt  = stats.mannwhitneyu(fa, la, alternative="greater") if len(fa)>0 else (0,1)
    u_dist, p_dist = stats.mannwhitneyu(fd, ld, alternative="greater") if len(fd)>0 else (0,1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean Amount ($)", f"{fa.mean():.2f}", delta=f"Legit {la.mean():.2f}")
    k2.metric("Median Amount ($)", f"{fa.median():.2f}", delta=f"Legit {la.median():.2f}")
    k3.metric("Mean Distance (km)", f"{fd.mean():.2f}", delta=f"Legit {ld.mean():.2f}")
    k4.metric("Median Dist (km)", f"{fd.median():.2f}", delta=f"Legit {ld.median():.2f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lab, arr, col in [("Legitimate",la,COLORS["legit"]),("Fraudulent",fa,COLORS["fraud"])]:
            fig.add_trace(go.Violin(y=arr, name=lab, box_visible=True,
                                    meanline_visible=True, fillcolor=col,
                                    opacity=0.6, line_color=col))
        fig.update_layout(base_layout(title="Transaction Amount", height=380,
                                      yaxis=dict(title="Amount ($)")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Amount test · U={u_amt:,.0f}, p={p_amt:.4f} · {'Significant' if p_amt<0.05 else 'Not Significant'}"
        )

    with c2:
        fig = go.Figure()
        for lab, arr, col in [("Legitimate",ld,COLORS["legit"]),("Fraudulent",fd,COLORS["fraud"])]:
            fig.add_trace(go.Violin(y=arr, name=lab, box_visible=True,
                                    meanline_visible=True, fillcolor=col,
                                    opacity=0.6, line_color=col))
        fig.update_layout(base_layout(title="Home → Merchant Distance", height=380,
                                      yaxis=dict(title="Distance (km)")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Distance test · U={u_dist:,.0f}, p={p_dist:.4f} · {'Significant' if p_dist<0.05 else 'Not Significant'}"
        )

    # Scatter: amount vs distance
    sample = dff.sample(min(2000, len(dff)), random_state=42)
    fig = px.scatter(
        sample, x="home_merch_dist", y="amt",
        color=sample["is_fraud"].map({0:"Legitimate",1:"Fraudulent"}),
        color_discrete_map={"Legitimate":COLORS["legit"],"Fraudulent":COLORS["fraud"]},
        opacity=0.45, title="Amount vs Home-to-Merchant Distance (sample n=2,000)",
        labels={"home_merch_dist":"Distance (km)","amt":"Amount ($)"},
    )
    fig.update_layout(base_layout(height=320, legend=dict(orientation="h", y=-0.15)))
    st.plotly_chart(fig, use_container_width=True)


# TAB 5: MODEL PERFORMANCE
# This tab shows how well our machine learning model is working
with tab5:
    st.markdown('<div class="sec-head">MODEL PERFORMANCE</div>', unsafe_allow_html=True)
    st.caption("Core classification quality and threshold behavior for the current filter selection.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.1%}")
    m2.metric("Precision", f"{precision:.1%}")
    m3.metric("Recall", f"{recall:.1%}")
    m4.metric("F1", f"{f1:.3f}")

    # Confusion matrix heatmap
    cm = np.array([[tn, fp],[fn, tp]])
    text_cm = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"],
               [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]
    fig = go.Figure(go.Heatmap(
        z=cm, text=text_cm, texttemplate="%{text}",
        colorscale=[[0,"#0b1929"],[1,"#1a3a60"]],
        showscale=False, xgap=4, ygap=4,
    ))
    fig.update_layout(base_layout(
        title=f"Confusion Matrix (threshold={threshold:.2f})", height=320,
        xaxis=dict(tickvals=[0,1], ticktext=["Pred Legit","Pred Fraud"], title=""),
        yaxis=dict(tickvals=[0,1], ticktext=["Actual Legit","Actual Fraud"], title=""),
    ))
    fig.update_traces(textfont=dict(family="Syne,sans-serif", size=14, color="#f0f6ff"))
    st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    fig = go.Figure()
    for lab, val, col in [("Legitimate",0,COLORS["legit"]),("Fraudulent",1,COLORS["fraud"])]:
        fig.add_trace(go.Histogram(x=dff[dff["is_fraud"]==val]["fraud_probability"],
                                   name=lab, opacity=0.75, marker_color=col, nbinsx=50))
    fig.add_vline(x=threshold, line_dash="dash", line_color=COLORS["amber"], line_width=2,
                  annotation_text=f"Threshold {threshold:.2f}",
                  annotation_font_color=COLORS["amber"])
    fig.update_layout(base_layout(title="Fraud Probability Score Distribution",
                                  barmode="overlay", height=290))
    st.plotly_chart(fig, use_container_width=True)

    # Threshold sensitivity curve
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        p = (dff["fraud_probability"] >= t).astype(int)
        _tp = ((dff["is_fraud"]==1) & (p==1)).sum()
        _fp = ((dff["is_fraud"]==0) & (p==1)).sum()
        _fn = ((dff["is_fraud"]==1) & (p==0)).sum()
        _tn = ((dff["is_fraud"]==0) & (p==0)).sum()
        _pr = _tp/(_tp+_fp) if (_tp+_fp)>0 else 0
        _re = _tp/(_tp+_fn) if (_tp+_fn)>0 else 0
        _f1 = 2*_pr*_re/(_pr+_re) if (_pr+_re)>0 else 0
        rows.append({"threshold":t,"precision":_pr,"recall":_re,"f1":_f1})
    tdf = pd.DataFrame(rows)
    fig = go.Figure()
    for col, color, name in [("precision",COLORS["legit"],"Precision"),
                               ("recall",COLORS["fraud"],"Recall"),
                               ("f1",COLORS["amber"],"F1 Score")]:
        fig.add_trace(go.Scatter(x=tdf["threshold"], y=tdf[col], mode="lines",
                                 name=name, line=dict(color=color, width=2)))
    fig.add_vline(x=threshold, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.update_layout(base_layout(title="Precision / Recall / F1 vs Threshold", height=300,
                                  xaxis=dict(title="Threshold", tickformat=".2f"),
                                  yaxis=dict(title="Score", tickformat=".0%"),
                                  legend=dict(orientation="h", y=-0.22)))
    st.plotly_chart(fig, use_container_width=True)



# Display information about the dashboard at the bottom
st.markdown(f"""
<div style="margin-top:40px;padding:16px 0;border-top:1px solid #1a3050;
     display:flex;justify-content:space-between;font-size:10px;color:#1e3a5f;">
  <span>🛡️ NovaPay Fraud Intelligence Platform</span>
  <span>{DATA_PATH}</span>
  <span>Streamlit · Plotly · SciPy</span>
</div>
""", unsafe_allow_html=True)
