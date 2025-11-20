


"""
TMA Crew Scheduler - Optimized Version
Features:
- Single tab interface for streamlined workflow
- Modular, reusable code structure
- Interactive Plotly visualizations
- Responsive design for all screen sizes
- Performance optimized with caching
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import io
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import custom functions
from functions import (
    safe_date_format, tma_df_processing, schedule_processing,
    melt_fun, add_rolling_counts, vacation_fun
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TMA Crew Scheduler",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS
# ============================================================================
MANAGEMENT_CREW = ['HJIF', 'MKHA', 'ISMS', 'AUZU', 'MRAS', 'ARMO', 'AHNA', 'DWOO']
TRAININGS = ['-', 'GRND 300', 'CRM', 'GRND 400', 'UPRT', 'MEDICAL', 'FLT TRNG', 
             'ESET', 'FMT', 'LPC', 'LINE CHECK', 'DGR', 'SMS', 'EVAC', 'FIRE', 
             'LC -400', 'OPC', 'RSQT', 'RSQC', 'LICENCE', 'TRE CERT', 'TRI Rating', 
             'TREMTR', 'TRIMTR', 'FLT TRNG', '', '   ', 'ELP']

COLUMN_RENAME = {
    'AL': 'Annual Leaves',
    1: 'Working days',
    'X': 'Off days',
    '-': 'Others'
}

COLOR_SCHEME = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'tma': '#ef4444',  # Red for TMA
    'new': '#667eea'   # Purple for New Schedule
}

# ============================================================================
# DATA LOADING & PROCESSING (CACHED)
# ============================================================================
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_static_data():
    """Load static reference data"""
    ac_data = pd.read_excel("Day Requirements.xlsx")
    tma_input_df = pd.read_excel("TMA Data.xlsx", "Data")
    return ac_data, tma_input_df

@st.cache_data(ttl=3600, show_spinner="Processing TMA data...")
def process_tma_baseline(_tma_input_df, _ac_data):
    """Process baseline TMA data"""
    expanded = tma_df_processing(_tma_input_df)
    vacation = vacation_fun(expanded)
    crew_summary = create_crew_summary(expanded)
    vacation_summary = create_vacation_summary(vacation, crew_summary)
    plan_summary = create_plan_summary(vacation_summary)
    availability = create_availability_data(expanded, _ac_data)
    
    return {
        'expanded': expanded,
        'vacation': vacation,
        'crew_summary': crew_summary,
        'vacation_summary': vacation_summary,
        'plan_summary': plan_summary,
        'availability': availability
    }

def process_uploaded_schedule(input_data, ac_data):
    """Process uploaded schedule file"""
    optimized = schedule_processing(input_data)
    expanded = melt_fun(optimized)
    vacation = vacation_fun(expanded)
    crew_summary = create_crew_summary(expanded)
    vacation_summary = create_vacation_summary(vacation, crew_summary)
    plan_summary = create_plan_summary(vacation_summary)
    availability = create_availability_data(expanded, ac_data)
    
    return {
        'optimized': optimized,
        'expanded': expanded,
        'vacation': vacation,
        'crew_summary': crew_summary,
        'vacation_summary': vacation_summary,
        'plan_summary': plan_summary,
        'availability': availability
    }

# ============================================================================
# DATA TRANSFORMATION HELPERS
# ============================================================================
def create_crew_summary(expanded_df):
    """Create crew summary with standardized column names"""
    summary = expanded_df.groupby(['Crew code', 'Status']).size().reset_index(name='count')
    summary = summary.pivot(index='Crew code', columns='Status', values='count').reset_index()
    summary = summary.rename(columns=COLUMN_RENAME)
    return summary

def create_vacation_summary(vacation_df, crew_summary):
    """Create vacation summary with statistics"""
    summary = vacation_df.groupby('Crew code').agg({
        'Duration': ['count', 'sum', 'mean', 'max', 'min'],
        'Days gap with next vacation': ['mean', 'max', 'min']
    }).reset_index()
    
    summary.columns = [
        'Crew code', 'Vacation frequency', 'Total Vacation leaves',
        'Average Vacation duration', 'Maximum Vacation duration', 'Min Vacation duration',
        'Average gap between vacations', 'Max gap between vacations', 'Min gap between vacations'
    ]
    
    summary['Average Vacation duration'] = summary['Average Vacation duration'].round(0)
    summary['Average gap between vacations'] = summary['Average gap between vacations'].round(0)
    summary = summary.merge(crew_summary, on='Crew code', how='left')
    
    return summary

def create_plan_summary(vacation_summary):
    """Create aggregated plan summary"""
    return pd.DataFrame({
        'Metric': ['Avg Off Days', 'Avg Vacation Days', 'Avg Working Days'],
        'Value': [
            round(vacation_summary['Off days'].mean()),
            round(vacation_summary['Annual Leaves'].mean()),
            round(vacation_summary['Working days'].mean())
        ]
    })

def create_availability_data(expanded_df, ac_data):
    """Create crew availability vs requirements data"""
    availability = expanded_df.groupby(["Date", "Status"]).size().reset_index()
    availability = availability[availability["Status"] == 1].reset_index(drop=True)
    availability.drop(["Status"], axis=1, inplace=True)
    availability.columns = ["Date", "Available crew"]
    availability = availability.merge(ac_data, on="Date", how="outer")
    availability["Excess crew"] = availability["Available crew"] - availability["AC"]
    availability["Category"] = availability["Excess crew"].apply(
        lambda x: "Excess" if x > 0 else ("Equal" if x == 0 else "Shortfall")
    )
    return availability

# ============================================================================
# VISUALIZATION COMPONENTS (REUSABLE)
# ============================================================================
def create_metric_card(label, value, comparison=None, is_positive=None, color_key='primary'):
    """Create a metric card with optional comparison badge"""
    color = COLOR_SCHEME[color_key]
    
    badge_html = ""
    if comparison is not None:
        badge_class = "badge-positive" if is_positive else "badge-negative"
        badge_html = f'<span class="comparison-badge {badge_class}">{comparison:+.0f} vs TMA</span>'
    
    return f'''
        <div class="metric-card" style="background: linear-gradient(135deg, {color} 0%, {COLOR_SCHEME['secondary']} 100%);">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {badge_html}
        </div>
    '''

def create_crew_wise_scatter_comparison(new_crew_summary, tma_crew_summary):
    """Create crew-wise scatter plot comparison for Off Days, Vacation Days, and Working Days"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Off Days by Crew', 'Vacation Days by Crew', 'Working Days by Crew'),
        horizontal_spacing=0.10
    )
    
    # Merge data for comparison
    comparison_df = new_crew_summary[['Crew code', 'Off days', 'Annual Leaves', 'Working days']].merge(
        tma_crew_summary[['Crew code', 'Off days', 'Annual Leaves', 'Working days']],
        on='Crew code',
        suffixes=(' New', ' TMA')
    )
    
    # Off Days comparison
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Crew code'],
            y=comparison_df['Off days TMA'],
            mode='markers',
            name='TMA',
            marker=dict(size=10, color=COLOR_SCHEME['tma'], symbol='circle'),
            hovertemplate='<b>%{x}</b><br>TMA Off Days: %{y}<extra></extra>',
            legendgroup='tma'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Crew code'],
            y=comparison_df['Off days New'],
            mode='markers',
            name='New',
            marker=dict(size=10, color=COLOR_SCHEME['new'], symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>New Off Days: %{y}<extra></extra>',
            legendgroup='new'
        ),
        row=1, col=1
    )
    
    # Vacation Days comparison
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Crew code'],
            y=comparison_df['Annual Leaves TMA'],
            mode='markers',
            name='TMA',
            marker=dict(size=10, color=COLOR_SCHEME['tma'], symbol='circle'),
            hovertemplate='<b>%{x}</b><br>TMA Vacation Days: %{y}<extra></extra>',
            legendgroup='tma',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Crew code'],
            y=comparison_df['Annual Leaves New'],
            mode='markers',
            name='New',
            marker=dict(size=10, color=COLOR_SCHEME['new'], symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>New Vacation Days: %{y}<extra></extra>',
            legendgroup='new',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Working Days comparison
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Crew code'],
            y=comparison_df['Working days TMA'],
            mode='markers',
            name='TMA',
            marker=dict(size=10, color=COLOR_SCHEME['tma'], symbol='circle'),
            hovertemplate='<b>%{x}</b><br>TMA Working Days: %{y}<extra></extra>',
            legendgroup='tma',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Crew code'],
            y=comparison_df['Working days New'],
            mode='markers',
            name='New',
            marker=dict(size=10, color=COLOR_SCHEME['new'], symbol='diamond'),
            hovertemplate='<b>%{x}</b><br>New Working Days: %{y}<extra></extra>',
            legendgroup='new',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.update_xaxes(title_text="Crew Code", row=1, col=1, tickangle=45)
    fig.update_xaxes(title_text="Crew Code", row=1, col=2, tickangle=45)
    fig.update_xaxes(title_text="Crew Code", row=1, col=3, tickangle=45)
    
    fig.update_yaxes(title_text="Days", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=1, col=2)
    fig.update_yaxes(title_text="Days", row=1, col=3)
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        title={'text': 'Crew-wise Days Comparison: TMA vs New Schedule', 'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=50, r=50, t=100, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='closest'
    )
    
    return fig

def create_distribution_chart(data, title, columns=['Working days', 'Annual Leaves', 'Off days']):
    """Create interactive distribution bar chart"""
    fig = go.Figure()
    
    colors = [COLOR_SCHEME['primary'], COLOR_SCHEME['warning'], COLOR_SCHEME['success']]
    
    for col_name, color in zip(columns, colors):
        if col_name in data.columns:
            value_counts = data[col_name].value_counts().sort_index()
            fig.add_trace(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=col_name,
                marker_color=color,
                text=value_counts.values,
                textposition='outside',
                hovertemplate='<b>%{x} days</b><br>Count: %{y}<extra></extra>'
            ))
    
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Days",
        yaxis_title="Number of Crew",
        barmode='group',
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_availability_timeline_comparison(new_availability_df, tma_availability_df):
    """Create crew availability comparison: AC required vs TMA vs New schedule"""
    fig = go.Figure()
    
    # Required crew line (AC)
    fig.add_trace(go.Scatter(
        x=new_availability_df['Date'],
        y=new_availability_df['AC'],
        mode='lines+markers',
        name='Required Crew (AC)',
        line=dict(color='#94a3b8', width=2.5, dash='dash'),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>Required: %{y}<extra></extra>'
    ))
    
    # TMA available crew line
    fig.add_trace(go.Scatter(
        x=tma_availability_df['Date'],
        y=tma_availability_df['Available crew'],
        mode='lines+markers',
        name='TMA Available Crew',
        line=dict(color=COLOR_SCHEME['tma'], width=2.5),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>TMA Available: %{y}<extra></extra>'
    ))
    
    # New available crew line
    fig.add_trace(go.Scatter(
        x=new_availability_df['Date'],
        y=new_availability_df['Available crew'],
        mode='lines+markers',
        name='New Available Crew',
        line=dict(color=COLOR_SCHEME['new'], width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>New Available: %{y}<extra></extra>'
    ))
    
    # Highlight new schedule shortfall areas
    shortfall = new_availability_df[new_availability_df['Category'] == 'Shortfall']
    if not shortfall.empty:
        fig.add_trace(go.Scatter(
            x=shortfall['Date'],
            y=shortfall['Available crew'],
            mode='markers',
            name='New Shortfall',
            marker=dict(size=14, color=COLOR_SCHEME['danger'], symbol='x', line=dict(width=2, color='white')),
            hovertemplate='<b>%{x}</b><br>⚠️ New Shortfall: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Crew Availability Comparison: AC Required vs TMA vs New Schedule', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Date',
        yaxis_title='Number of Crew',
        template='plotly_white',
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_comparison_chart(new_summary, tma_summary, new_insights, tma_insights):
    """Create side-by-side comparison bar chart using insights data for consistency"""
    metrics = ['Avg Off Days', 'Avg Vacation Days', 'Avg Working Days']
    
    # Use insights data instead of summary for consistency with KPI cards
    new_values = [
        new_insights['avg_off'],
        new_insights['avg_vacation'],
        new_insights['avg_working']
    ]
    
    tma_values = [
        tma_insights['avg_off'],
        tma_insights['avg_vacation'],
        tma_insights['avg_working']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='New Schedule',
        x=metrics,
        y=new_values,
        marker_color=COLOR_SCHEME['new'],
        text=[f"{v:.1f}" for v in new_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>New: %{y:.1f} days<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='TMA Schedule',
        x=metrics,
        y=tma_values,
        marker_color=COLOR_SCHEME['tma'],
        text=[f"{v:.1f}" for v in tma_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>TMA: %{y:.1f} days<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Schedule Comparison: New vs TMA', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='',
        yaxis_title='Days',
        barmode='group',
        template='plotly_white',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_vacation_comparison_boxplot(new_vacation_summary, tma_vacation_summary):
    """Create vacation statistics comparison box plots: TMA vs New"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Vacation Duration (Days)', 'Gap Between Vacations (Days)', 'Vacation Frequency'),
        horizontal_spacing=0.12
    )
    
    # Duration comparison
    fig.add_trace(
        go.Box(
            y=tma_vacation_summary['Average Vacation duration'],
            name='TMA',
            marker_color=COLOR_SCHEME['tma'],
            boxmean='sd',
            hovertemplate='TMA Duration: %{y} days<extra></extra>',
            legendgroup='duration'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(
            y=new_vacation_summary['Average Vacation duration'],
            name='New',
            marker_color=COLOR_SCHEME['new'],
            boxmean='sd',
            hovertemplate='New Duration: %{y} days<extra></extra>',
            legendgroup='duration'
        ),
        row=1, col=1
    )
    
    # Gap comparison
    fig.add_trace(
        go.Box(
            y=tma_vacation_summary['Average gap between vacations'],
            name='TMA',
            marker_color=COLOR_SCHEME['tma'],
            boxmean='sd',
            hovertemplate='TMA Gap: %{y} days<extra></extra>',
            legendgroup='gap',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Box(
            y=new_vacation_summary['Average gap between vacations'],
            name='New',
            marker_color=COLOR_SCHEME['new'],
            boxmean='sd',
            hovertemplate='New Gap: %{y} days<extra></extra>',
            legendgroup='gap',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Frequency comparison
    fig.add_trace(
        go.Box(
            y=tma_vacation_summary['Vacation frequency'],
            name='TMA',
            marker_color=COLOR_SCHEME['tma'],
            boxmean='sd',
            hovertemplate='TMA Frequency: %{y} times<extra></extra>',
            legendgroup='frequency',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Box(
            y=new_vacation_summary['Vacation frequency'],
            name='New',
            marker_color=COLOR_SCHEME['new'],
            boxmean='sd',
            hovertemplate='New Frequency: %{y} times<extra></extra>',
            legendgroup='frequency',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        height=450,
        template='plotly_white',
        title={'text': 'Vacation Pattern Comparison: TMA vs New Schedule', 'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Days", row=1, col=1)
    fig.update_yaxes(title_text="Days", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=3)
    
    return fig

def create_pie_chart(availability_df):
    """Create pie chart for availability categories"""
    category_counts = availability_df['Category'].value_counts()
    
    colors = {
        'Shortfall': COLOR_SCHEME['danger'],
        'Equal': COLOR_SCHEME['warning'],
        'Excess': COLOR_SCHEME['success']
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        marker=dict(colors=[colors.get(cat, COLOR_SCHEME['info']) for cat in category_counts.index]),
        hovertemplate='<b>%{label}</b><br>Days: %{value}<br>Percentage: %{percent}<extra></extra>',
        textinfo='label+percent',
        textposition='inside'
    )])
    
    fig.update_layout(
        title={'text': 'Crew Availability Status Distribution', 'x': 0.5, 'xanchor': 'center'},
        height=400,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_workload_balance_chart(new_crew_summary, tma_crew_summary):
    """Create a comprehensive workload balance comparison showing working days range"""
    fig = go.Figure()
    
    # TMA working days distribution
    fig.add_trace(go.Box(
        y=tma_crew_summary['Working days'],
        name='TMA Schedule',
        marker_color=COLOR_SCHEME['tma'],
        boxmean='sd',
        hovertemplate='TMA Working Days: %{y}<extra></extra>'
    ))
    
    # New working days distribution
    fig.add_trace(go.Box(
        y=new_crew_summary['Working days'],
        name='New Schedule',
        marker_color=COLOR_SCHEME['new'],
        boxmean='sd',
        hovertemplate='New Working Days: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Working Days Distribution: Workload Balance Analysis', 'x': 0.5, 'xanchor': 'center'},
        yaxis_title='Working Days',
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_rest_days_analysis(new_crew_summary, tma_crew_summary):
    """Create combined Off Days + Vacation analysis"""
    # Calculate total rest days (Off Days + Vacation)
    new_crew_summary['Total Rest Days'] = new_crew_summary['Off days'] + new_crew_summary['Annual Leaves']
    tma_crew_summary['Total Rest Days'] = tma_crew_summary['Off days'] + tma_crew_summary['Annual Leaves']
    
    fig = go.Figure()
    
    # TMA total rest
    fig.add_trace(go.Histogram(
        x=tma_crew_summary['Total Rest Days'],
        name='TMA Schedule',
        marker_color=COLOR_SCHEME['tma'],
        opacity=0.7,
        nbinsx=15,
        hovertemplate='TMA Rest Days: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # New total rest
    fig.add_trace(go.Histogram(
        x=new_crew_summary['Total Rest Days'],
        name='New Schedule',
        marker_color=COLOR_SCHEME['new'],
        opacity=0.7,
        nbinsx=15,
        hovertemplate='New Rest Days: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={'text': 'Total Rest Days Distribution (Off Days + Vacation)', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Total Rest Days',
        yaxis_title='Number of Crew',
        barmode='overlay',
        template='plotly_white',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_crew_schedule_breakdown(crew_summary):
    """Create clustered bar chart showing 365-day breakdown per crew"""
    # Calculate 'Others' category (365 - all other categories)
    crew_data = crew_summary.copy()
    
    # Get all columns, handle missing ones
    working_days = crew_data.get('Working days', 0)
    off_days = crew_data.get('Off days', 0)
    annual_leaves = crew_data.get('Annual Leaves', 0)
    
    # Calculate Others
    crew_data['Others'] = 365 - (working_days + off_days + annual_leaves)
    
    # Ensure no negative values
    crew_data['Others'] = crew_data['Others'].clip(lower=0)
    
    fig = go.Figure()
    
    # Add bars for each category
    fig.add_trace(go.Bar(
        name='Working Days',
        x=crew_data['Crew code'],
        y=crew_data.get('Working days', 0),
        marker_color=COLOR_SCHEME['primary'],
        hovertemplate='<b>%{x}</b><br>Working Days: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Annual Leaves',
        x=crew_data['Crew code'],
        y=crew_data.get('Annual Leaves', 0),
        marker_color=COLOR_SCHEME['warning'],
        hovertemplate='<b>%{x}</b><br>Annual Leaves: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Off Days',
        x=crew_data['Crew code'],
        y=crew_data.get('Off days', 0),
        marker_color=COLOR_SCHEME['success'],
        hovertemplate='<b>%{x}</b><br>Off Days: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='Others',
        x=crew_data['Crew code'],
        y=crew_data['Others'],
        marker_color='#94a3b8',
        hovertemplate='<b>%{x}</b><br>Others: %{y}<extra></extra>'
    ))
    
    # Add a reference line at 365
    fig.add_hline(
        y=365,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text="Total: 365 days",
        annotation_position="right"
    )
    
    fig.update_layout(
        title={'text': 'Crew Schedule Breakdown (365 Days Total)', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Crew Code',
        yaxis_title='Days',
        barmode='stack',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=100),
        hovermode='x unified'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_availability_trend_analysis(availability_df):
    """Create trend analysis for crew availability without moving averages"""
    availability_df = availability_df.sort_values('Date').copy()
    
    fig = go.Figure()
    
    # Available crew
    fig.add_trace(go.Scatter(
        x=availability_df['Date'],
        y=availability_df['Available crew'],
        mode='lines+markers',
        name='Available Crew',
        line=dict(color=COLOR_SCHEME['tma'], width=2.5),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>Available: %{y}<extra></extra>'
    ))
    
    # Required crew
    fig.add_trace(go.Scatter(
        x=availability_df['Date'],
        y=availability_df['AC'],
        mode='lines',
        name='Required Crew',
        line=dict(color='#94a3b8', width=2, dash='dash'),
        hovertemplate='<b>%{x}</b><br>Required: %{y}<extra></extra>'
    ))
    
    # Highlight shortfalls
    shortfall = availability_df[availability_df['Category'] == 'Shortfall']
    if not shortfall.empty:
        fig.add_trace(go.Scatter(
            x=shortfall['Date'],
            y=shortfall['Available crew'],
            mode='markers',
            name='Shortfall',
            marker=dict(size=12, color=COLOR_SCHEME['danger'], symbol='x', line=dict(width=2, color='white')),
            hovertemplate='<b>%{x}</b><br>⚠️ Shortfall: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': 'Crew Availability Trend Analysis', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Date',
        yaxis_title='Number of Crew',
        template='plotly_white',
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified'
    )
    
    return fig

def create_workload_inequality_index(crew_summary):
    """Calculate and visualize workload inequality using coefficient of variation"""
    import numpy as np
    
    working_days = crew_summary['Working days'].values
    mean_work = np.mean(working_days)
    std_work = np.std(working_days)
    cv = (std_work / mean_work) * 100  # Coefficient of Variation
    
    fig = go.Figure()
    
    # Distribution curve
    fig.add_trace(go.Histogram(
        x=working_days,
        name='Working Days Distribution',
        marker_color=COLOR_SCHEME['new'],
        nbinsx=20,
        hovertemplate='Working Days: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean_work,
        line_dash="dash",
        line_color=COLOR_SCHEME['tma'],
        annotation_text=f"Mean: {mean_work:.1f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title={'text': f'Workload Distribution (CV: {cv:.1f}% - Lower is Better)', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title='Working Days',
        yaxis_title='Number of Crew',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig, cv

def create_tma_insights_summary(tma_data):
    """Create comprehensive TMA insights"""
    crew_summary = tma_data['crew_summary']
    vacation_summary = tma_data['vacation_summary']
    availability = tma_data['availability']
    
    insights = {
        'total_crew': len(crew_summary),
        'avg_working': crew_summary['Working days'].mean(),
        'avg_off': crew_summary['Off days'].mean(),
        'avg_vacation': crew_summary['Annual Leaves'].mean(),
        'max_working': crew_summary['Working days'].max(),
        'min_working': crew_summary['Working days'].min(),
        'std_working': crew_summary['Working days'].std(),
        'shortfall_days': len(availability[availability['Category'] == 'Shortfall']),
        'excess_days': len(availability[availability['Category'] == 'Excess']),
        'avg_vacation_duration': vacation_summary['Average Vacation duration'].mean(),
        'avg_vacation_gap': vacation_summary['Average gap between vacations'].mean(),
        'total_rest_days': (crew_summary['Off days'] + crew_summary['Annual Leaves']).mean()
    }
    
    return insights

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================
def export_to_excel(data_dict, filename_prefix="TMA_Report"):
    """Export multiple dataframes to Excel with timestamp"""
    output = io.BytesIO()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in data_dict.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                    worksheet.set_column(idx, idx, max_length)
    
    output.seek(0)
    return output, f"{filename_prefix}_{timestamp}.xlsx"

# ============================================================================
# UI COMPONENTS
# ============================================================================
def load_custom_css():
    """Load responsive custom CSS"""
    st.markdown("""
        <style>
        /* Responsive design */
        @media (max-width: 768px) {
            .metric-card { font-size: 0.8rem; padding: 15px; }
            .metric-value { font-size: 1.8rem !important; }
            .section-header { font-size: 1.3rem !important; }
        }
        
        /* Main styling */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 0;
        }
        
        .block-container {
            padding: 1rem 2rem;
            max-width: 100%;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s ease;
            margin-bottom: 1rem;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.95;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }
        
        .comparison-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 8px;
        }
        
        .badge-positive {
            background: rgba(16, 185, 129, 0.3);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        .badge-negative {
            background: rgba(239, 68, 68, 0.3);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: white;
            margin: 30px 0 20px 0;
            padding: 15px 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            border-left: 5px solid #ffffff;
        }
        
        /* Info boxes */
        .info-box {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-weight: 600;
            width: 100%;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: white;
            border-radius: 10px;
            padding: 20px;
        }
        
        /* Hide branding */
        #MainMenu, footer, header {visibility: hidden;}
        
        /* Responsive plotly */
        .plotly-graph-div {
            width: 100% !important;
        }
        
        /* Custom metric with subtitle */
        .metric-subtitle {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 4px;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    load_custom_css()
    
    

    st.markdown("""
        <div style="text-align: center; padding: 40px 0 30px 0; margin-bottom: 30px; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <div style="font-size: 4rem; margin-bottom: 15px;">✈️</div>
            <h1 style="color: #1e293b; font-size: 2.8rem; margin-bottom: 10px; font-weight: 400; letter-spacing: 1px;">TMA Crew Scheduler</h1>
            <p style="color: #64748b; font-size: 1.1rem; font-weight: 400;">Schedule Validation & Analysis Tool</p>
        </div>
    """, unsafe_allow_html=True)


    # Load baseline data
    with st.spinner("🔄 Loading baseline data..."):
        ac_data, tma_input_df = load_static_data()
        tma_data = process_tma_baseline(tma_input_df, ac_data)
    
    st.markdown('<h1 class="section-header">📊 Schedule Validation & Analysis</h1>', unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "📁 Upload Your Schedule File",
        type=["csv", "xlsx"],
        help="Upload crew schedule in Excel or CSV format"
    )
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        
        try:
            input_data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            
            if st.button("🔍 Validate & Analyze Schedule", use_container_width=True):
                with st.spinner("🔄 Processing your schedule..."):
                    # Process uploaded schedule
                    new_data = process_uploaded_schedule(input_data, ac_data)
                
                st.success("✅ Analysis complete!")
                
                # === KPI CARDS WITH TMA COMPARISON ===
                st.markdown('<h2 class="section-header">📊 New Schedule Key Performance Indicators</h2>', unsafe_allow_html=True)
                
                # Calculate new insights
                new_insights = create_tma_insights_summary(new_data)
                tma_insights = create_tma_insights_summary(tma_data)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    crew_change = new_insights['total_crew'] - tma_insights['total_crew']
                    st.metric(
                        "👥 Total Crew",
                        new_insights['total_crew'],
                        f"{crew_change:+d} vs TMA ({tma_insights['total_crew']})"
                    )
                    
                    working_change = new_insights['avg_working'] - tma_insights['avg_working']
                    st.metric(
                        "💼 Avg Working Days",
                        f"{new_insights['avg_working']:.1f}",
                        f"{working_change:+.1f} vs TMA ({tma_insights['avg_working']:.1f})",
                        delta_color="inverse"
                    )
                
                with col2:
                    vacation_change = new_insights['avg_vacation'] - tma_insights['avg_vacation']
                    st.metric(
                        "🏖️ Avg Vacation Days",
                        f"{new_insights['avg_vacation']:.1f}",
                        f"{vacation_change:+.1f} vs TMA ({tma_insights['avg_vacation']:.1f})"
                    )
                    
                    duration_change = new_insights['avg_vacation_duration'] - tma_insights['avg_vacation_duration']
                    st.metric(
                        "📅 Avg Vacation Duration",
                        f"{new_insights['avg_vacation_duration']:.1f}",
                        f"{duration_change:+.1f} vs TMA ({tma_insights['avg_vacation_duration']:.1f})"
                    )
                
                with col3:
                    off_change = new_insights['avg_off'] - tma_insights['avg_off']
                    st.metric(
                        "😴 Avg Off Days",
                        f"{new_insights['avg_off']:.1f}",
                        f"{off_change:+.1f} vs TMA ({tma_insights['avg_off']:.1f})"
                    )
                    
                    rest_change = new_insights['total_rest_days'] - tma_insights['total_rest_days']
                    st.metric(
                        "🔄 Total Rest Days",
                        f"{new_insights['total_rest_days']:.1f}",
                        f"{rest_change:+.1f} vs TMA ({tma_insights['total_rest_days']:.1f})"
                    )
                
                with col4:
                    shortfall_change = new_insights['shortfall_days'] - tma_insights['shortfall_days']
                    st.metric(
                        "⚠️ Shortfall Days",
                        new_insights['shortfall_days'],
                        f"{shortfall_change:+d} vs TMA ({tma_insights['shortfall_days']})",
                        delta_color="inverse"
                    )
                    
                    excess_change = new_insights['excess_days'] - tma_insights['excess_days']
                    st.metric(
                        "✅ Excess Days",
                        new_insights['excess_days'],
                        f"{excess_change:+d} vs TMA ({tma_insights['excess_days']})"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # === COMPARISON CHART ===
                st.markdown('<h2 class="section-header">⚖️ Schedule Metrics Comparison</h2>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                fig_comparison = create_comparison_chart(
                    new_data['plan_summary'], 
                    tma_data['plan_summary'],
                    new_insights,
                    tma_insights
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # === CREW-WISE SCATTER PLOT COMPARISON ===
                st.markdown('<h2 class="section-header">👥 Crew-wise Days Comparison</h2>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                fig_scatter = create_crew_wise_scatter_comparison(new_data['crew_summary'], tma_data['crew_summary'])
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                
                
                # === DISTRIBUTION ANALYSIS ===
                st.markdown('<h2 class="section-header">📊 Distribution Analysis</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### 🆕 New Schedule Distribution")
                    fig_new_dist = create_distribution_chart(new_data['crew_summary'], "New Schedule")
                    st.plotly_chart(fig_new_dist, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### 📋 TMA Baseline Distribution")
                    fig_tma_dist = create_distribution_chart(tma_data['crew_summary'], "TMA Baseline")
                    st.plotly_chart(fig_tma_dist, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # === AVAILABILITY ANALYSIS ===
                st.markdown('<h2 class="section-header">👥 Crew Availability Analysis</h2>', unsafe_allow_html=True)
                
                # Timeline comparison
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                fig_timeline = create_availability_timeline_comparison(new_data['availability'], tma_data['availability'])
                st.plotly_chart(fig_timeline, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Pie charts comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### 📋 TMA Availability Status")
                    fig_pie_tma = create_pie_chart(tma_data['availability'])
                    st.plotly_chart(fig_pie_tma, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("### 🆕 New Availability Status")
                    fig_pie_new = create_pie_chart(new_data['availability'])
                    st.plotly_chart(fig_pie_new, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Statistics with improved comparison format
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                # st.markdown("### 📊 Availability Category Comparison")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate stats for both
                new_shortfall = len(new_data['availability'][new_data['availability']['Category'] == 'Shortfall'])
                tma_shortfall = len(tma_data['availability'][tma_data['availability']['Category'] == 'Shortfall'])
                shortfall_change = new_shortfall - tma_shortfall
                
                new_excess = len(new_data['availability'][new_data['availability']['Category'] == 'Excess'])
                tma_excess = len(tma_data['availability'][tma_data['availability']['Category'] == 'Excess'])
                excess_change = new_excess - tma_excess
                
                new_equal = len(new_data['availability'][new_data['availability']['Category'] == 'Equal'])
                tma_equal = len(tma_data['availability'][tma_data['availability']['Category'] == 'Equal'])
                equal_change = new_equal - tma_equal
                
                total = len(new_data['availability'])
                avg_excess = new_data['availability']['Excess crew'].mean()
                
                with col1:
                    # Calculate percentage change
                    if tma_shortfall > 0:
                        pct_change = ((new_shortfall - tma_shortfall) / tma_shortfall) * 100
                    else:
                        pct_change = 0 if new_shortfall == 0 else 100
                    
                    delta_text = f"{pct_change:+.1f}% vs TMA ({tma_shortfall})"
                    delta_color = "inverse" if shortfall_change > 0 else "normal"
                    st.metric(
                        "⚠️ Shortfall Days", 
                        new_shortfall, 
                        delta_text,
                        delta_color=delta_color
                    )
                
                with col2:
                    # Calculate percentage change
                    if tma_excess > 0:
                        pct_change = ((new_excess - tma_excess) / tma_excess) * 100
                    else:
                        pct_change = 0 if new_excess == 0 else 100
                    
                    delta_text = f"{pct_change:+.1f}% vs TMA ({tma_excess})"
                    delta_color = "normal" if excess_change > 0 else "inverse"
                    st.metric(
                        "✅ Excess Days", 
                        new_excess, 
                        delta_text,
                        delta_color=delta_color
                    )
                
                with col3:
                    # Calculate percentage change
                    if tma_equal > 0:
                        pct_change = ((new_equal - tma_equal) / tma_equal) * 100
                    else:
                        pct_change = 0 if new_equal == 0 else 100
                    
                    delta_text = f"{pct_change:+.1f}% vs TMA ({tma_equal})"
                    st.metric(
                        "⚖️ Equal Days", 
                        new_equal, 
                        delta_text
                    )
                
                with col4:
                    st.metric("📊 Avg Excess Crew", f"{avg_excess:.1f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # === VACATION PATTERNS ===
                st.markdown('<h2 class="section-header">🏖️ Vacation Pattern Analysis</h2>', unsafe_allow_html=True)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                fig_vacation = create_vacation_comparison_boxplot(new_data['vacation_summary'], tma_data['vacation_summary'])
                st.plotly_chart(fig_vacation, use_container_width=True)
                
                # Vacation statistics comparison
                # st.markdown("### 📊 Statistical Comparison")
                col1, col2, col3 = st.columns(3)
                
                # Duration comparison
                with col1:
                    new_avg_duration = new_data['vacation_summary']['Average Vacation duration'].mean()
                    tma_avg_duration = tma_data['vacation_summary']['Average Vacation duration'].mean()
                    duration_diff = new_avg_duration - tma_avg_duration
                    st.metric(
                        "📅 Avg Vacation Duration", 
                        f"{new_avg_duration:.0f} days",
                        f"{duration_diff:+.0f} vs TMA ({tma_avg_duration:.0f})"
                    )
                
                # Gap comparison
                with col2:
                    new_avg_gap = new_data['vacation_summary']['Average gap between vacations'].mean()
                    tma_avg_gap = tma_data['vacation_summary']['Average gap between vacations'].mean()
                    gap_diff = new_avg_gap - tma_avg_gap
                    st.metric(
                        "⏳ Avg Gap Between Vacations", 
                        f"{new_avg_gap:.0f} days",
                        f"{gap_diff:+.0f} vs TMA ({tma_avg_gap:.0f})"
                    )
                
                # Frequency comparison
                with col3:
                    new_avg_freq = new_data['vacation_summary']['Vacation frequency'].mean()
                    tma_avg_freq = tma_data['vacation_summary']['Vacation frequency'].mean()
                    freq_diff = new_avg_freq - tma_avg_freq
                    st.metric(
                        "🔢 Avg Vacation Frequency", 
                        f"{new_avg_freq:.1f} times",
                        f"{freq_diff:+.1f} vs TMA ({tma_avg_freq:.1f})"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # === DETAILED DATA ===
                with st.expander("📋 View Detailed Data Tables", expanded=False):
                    tab1, tab2, tab3, tab4 = st.tabs(["Crew Summary", "Vacation Summary", "Availability", "Raw Data"])
                    
                    with tab1:
                        st.dataframe(new_data['crew_summary'], use_container_width=True, height=400)
                    
                    with tab2:
                        st.dataframe(new_data['vacation_summary'], use_container_width=True, height=400)
                    
                    with tab3:
                        st.dataframe(new_data['availability'], use_container_width=True, height=400)
                    
                    with tab4:
                        st.dataframe(new_data['expanded'].head(100), use_container_width=True, height=400)
                        st.info(f"Showing first 100 rows of {len(new_data['expanded'])} total rows")
                
                # === DOWNLOAD REPORTS ===
                st.markdown('<h2 class="section-header">💾 Export Reports</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Comprehensive report
                    export_data = {
                        'Plan_Summary': new_data['plan_summary'],
                        'Crew_Summary': new_data['crew_summary'],
                        'Vacation_Summary': new_data['vacation_summary'],
                        'Availability': new_data['availability'],
                        'Vacation_Details': new_data['vacation'],
                        'Expanded_Data': new_data['expanded']
                    }
                    output, filename = export_to_excel(export_data, "TMA_Full_Report")
                    
                    st.download_button(
                        label="📥 Download Complete Report",
                        data=output,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    # Summary report
                    summary_data = {
                        'Plan_Summary': new_data['plan_summary'],
                        'Crew_Summary': new_data['crew_summary'],
                        'Vacation_Summary': new_data['vacation_summary']
                    }
                    output_summary, filename_summary = export_to_excel(summary_data, "TMA_Summary")
                    
                    st.download_button(
                        label="📊 Download Summary Only",
                        data=output_summary,
                        file_name=filename_summary,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file has the correct format and required columns.")
    
    else:
        st.info("👆 Please upload a schedule file to begin analysis")
        
        # === TMA BASELINE INSIGHTS ===
        st.markdown('<h2 class="section-header">📊 TMA Baseline Insights</h2>', unsafe_allow_html=True)
        
        # Calculate insights
        tma_insights = create_tma_insights_summary(tma_data)
        
        # KPI Cards
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Crew", tma_insights['total_crew'])
            st.metric("💼 Avg Working Days", f"{tma_insights['avg_working']:.1f}")
        with col2:
            st.metric("🏖️ Avg Vacation Days", f"{tma_insights['avg_vacation']:.1f}")
            st.metric("📅 Avg Vacation Duration", f"{tma_insights['avg_vacation_duration']:.1f}")
        with col3:
            st.metric("😴 Avg Off Days", f"{tma_insights['avg_off']:.1f}")
            st.metric("🔄 Total Rest Days", f"{tma_insights['total_rest_days']:.1f}")
        with col4:
            st.metric("⚠️ Shortfall Days", tma_insights['shortfall_days'])
            st.metric("✅ Excess Days", tma_insights['excess_days'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Crew Schedule Breakdown (365 days)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Crew Schedule Breakdown")
        fig_breakdown = create_crew_schedule_breakdown(tma_data['crew_summary'])
        st.plotly_chart(fig_breakdown, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Availability Trend
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📈 Availability Trend Analysis")
        fig_trend = create_availability_trend_analysis(tma_data['availability'])
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()