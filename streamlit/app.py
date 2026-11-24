"""
Wholesale Customer Segmentation Dashboard
Built with Streamlit
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Wholesale Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

@st.cache_data
def load_data():
    file_path = 'dataset/clustered_customers.csv'
    if not os.path.exists(file_path):
        file_path = 'clustered_customers.csv'
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Sidebar
st.sidebar.title("🎛️ Navigation & Filter")
st.sidebar.markdown("Filter wholesale customer segments to explore behaviors and distributions.")

cluster_options = ["All Segments"] + sorted(df['Cluster_Name'].unique().tolist())
selected_cluster = st.sidebar.selectbox("Select Customer Segment:", cluster_options)

channel_options = ["All Channels", "Horeca", "Retail"]
selected_channel = st.sidebar.selectbox("Filter Channel:", channel_options)

region_options = ["All Regions", "Lisbon", "Oporto", "Other Region"]
selected_region = st.sidebar.selectbox("Filter Region:", region_options)

# Filter logic
filtered_df = df.copy()
if selected_cluster != "All Segments":
    filtered_df = filtered_df[filtered_df['Cluster_Name'] == selected_cluster]
if selected_channel != "All Channels":
    filtered_df = filtered_df[filtered_df['Channel_Desc'] == selected_channel]
if selected_region != "All Regions":
    filtered_df = filtered_df[filtered_df['Region_Desc'] == selected_region]

# Main Dashboard Header
st.markdown('<div class="main-header">Wholesale Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven behavioral clustering using K-Means (k=3) on Log-Standardized annual spending</div>', unsafe_allow_html=True)

# Top KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="Selected Customers", value=f"{len(filtered_df):,}", delta=f"{(len(filtered_df)/len(df))*100:.1f}% of Total")
with kpi2:
    total_spend = filtered_df[FEATURE_COLS].sum().sum()
    st.metric(label="Total Spending (m.u.)", value=f"{total_spend:,.0f}")
with kpi3:
    avg_customer_spend = total_spend / len(filtered_df) if len(filtered_df) > 0 else 0
    st.metric(label="Avg Customer Spend", value=f"{avg_customer_spend:,.0f} m.u.")
with kpi4:
    dominant_channel = filtered_df['Channel_Desc'].mode()[0] if len(filtered_df) > 0 else 'N/A'
    st.metric(label="Dominant Channel", value=dominant_channel)

st.markdown("---")

# Visual Section 1: PCA Projection & Cluster Sizes
col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("📍 Customer Clusters in 2D PCA Space")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cluster_colors = {
        'Broad-Category High Spenders': '#2b5c8f',
        'Fresh & Frozen / Horeca Focused': '#2ca02c',
        'Grocery & Detergents / Retail Specialists': '#d62728'
    }
    for c_name, sub in df.groupby('Cluster_Name'):
        ax.scatter(sub['PC1'], sub['PC2'], color=cluster_colors.get(c_name, '#555555'),
                   label=f"{c_name} (n={len(sub)})", alpha=0.65, edgecolors='none', s=50)

    # Highlight filtered points if filtered
    if len(filtered_df) < len(df) and len(filtered_df) > 0:
        ax.scatter(filtered_df['PC1'], filtered_df['PC2'], facecolors='none', edgecolors='gold', s=90, linewidth=1.5, label='Filtered Subset')

    ax.set_title("PCA Biplot (PC1: Pantry Goods vs PC2: Fresh/Frozen)", fontsize=11, fontweight='bold')
    ax.set_xlabel("PC1 (44.1% Variance) - Grocery, Milk, Detergents")
    ax.set_ylabel("PC2 (27.2% Variance) - Fresh, Frozen, Delicassen")
    ax.legend(fontsize=8, loc='upper right')
    st.pyplot(fig)

with col_right:
    st.subheader("📊 Customer Count by Segment")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    segment_counts = df['Cluster_Name'].value_counts()
    colors = [cluster_colors.get(name, '#333') for name in segment_counts.index]
    bars = ax.barh(range(len(segment_counts)), segment_counts.values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(range(len(segment_counts)))
    ax.set_yticklabels([name.replace(' / ', '\n') for name in segment_counts.index], fontsize=9)
    ax.invert_yaxis()
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 3, bar.get_y() + bar.get_height()/2, f"{int(w)} ({w/len(df)*100:.1f}%)", va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(segment_counts.values) * 1.25)
    ax.set_xlabel("Number of Clients")
    ax.set_title("Overall Segment Share (N=440)", fontsize=11, fontweight='bold')
    st.pyplot(fig)

# Visual Section 2: Spending Profiles (Mean & Median)
st.subheader("💰 Product Category Spending Profiles (Original Scale in m.u.)")
p_col1, p_col2 = st.columns(2)

with p_col1:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mean_data = filtered_df[FEATURE_COLS].mean()
    sns.barplot(x=mean_data.index, y=mean_data.values, ax=ax, palette='Blues_r', edgecolor='black')
    ax.set_title("Mean Annual Spending (Filtered Subset)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mean Spending (m.u.)")
    ax.tick_params(axis='x', rotation=25)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    st.pyplot(fig)

with p_col2:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    median_data = filtered_df[FEATURE_COLS].median()
    sns.barplot(x=median_data.index, y=median_data.values, ax=ax, palette='Greens_r', edgecolor='black')
    ax.set_title("Median Annual Spending (Filtered Subset)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Median Spending (m.u.)")
    ax.tick_params(axis='x', rotation=25)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', xytext=(0, 3), textcoords='offset points')
    st.pyplot(fig)

# Visual Section 3: Channel & Geographic Breakdown
st.subheader("🏢 Channel and Geographic Distribution")
cg1, cg2 = st.columns(2)

with cg1:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ch_counts = filtered_df['Channel_Desc'].value_counts()
    ax.pie(ch_counts, labels=ch_counts.index, autopct='%1.1f%%', colors=['#3498db', '#e67e22'],
           startangle=140, explode=[0.03]*len(ch_counts), textprops={'fontweight': 'bold'})
    ax.set_title("Channel Distribution", fontsize=11, fontweight='bold')
    st.pyplot(fig)

with cg2:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    reg_counts = filtered_df['Region_Desc'].value_counts()
    ax.pie(reg_counts, labels=reg_counts.index, autopct='%1.1f%%', colors=['#9b59b6', '#1abc9c', '#34495e'],
           startangle=140, explode=[0.03]*len(reg_counts), textprops={'fontweight': 'bold'})
    ax.set_title("Geographic Region Distribution", fontsize=11, fontweight='bold')
    st.pyplot(fig)

# Section 4: Data Table
st.subheader("📋 Customer Level Records")
st.dataframe(filtered_df[['Channel_Desc', 'Region_Desc', 'Cluster_Name'] + FEATURE_COLS + ['PC1', 'PC2']].head(50), width=1000)

# Section 5: Strategic Business Action Plan
st.subheader("🎯 Segment Specific Strategic Playbook")
exp1, exp2, exp3 = st.columns(3)

with exp1:
    st.markdown("""
    **Cluster 0: Broad-Category High Spenders**
    - **Focus:** Key Account Retention & Loyalty
    - **Revenue Share:** ~52.8% of total volume
    - **Strategy:** Assign dedicated account managers, provide SLA priority delivery, volume-based bundle incentives.
    """)

with exp2:
    st.markdown("""
    **Cluster 1: Fresh & Frozen / Horeca Focused**
    - **Focus:** Cold Chain Reliability & Freshness
    - **Revenue Share:** ~30.6% of total volume (98.6% Horeca)
    - **Strategy:** Daily scheduled fresh delivery, seasonal menu promotions, bundled pre-cut frozen supplies.
    """)

with exp3:
    st.markdown("""
    **Cluster 2: Grocery & Detergents / Retail Specialists**
    - **Focus:** Pallet Pricing & Inventory Turnover
    - **Revenue Share:** ~16.6% of total volume (60.2% Retail)
    - **Strategy:** Wholesale bulk discounts on pantry items, cross-selling paper/hygiene products, credit term extensions.
    """)
