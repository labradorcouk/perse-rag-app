# eda_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from fpdf import FPDF
from io import BytesIO
import base64
import tempfile
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
APP_DEBUG = os.getenv('APP_DEBUG', 'False').lower() == 'true'
APP_ENV = os.getenv('APP_ENV', 'development')
MAX_SAMPLE_SIZE = int(os.getenv('MAX_SAMPLE_SIZE', '1000'))
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '600'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.2'))
REPORT_TITLE = os.getenv('REPORT_TITLE', 'Energy Report QA Analysis')
REPORT_FONT = os.getenv('REPORT_FONT', 'Arial')
REPORT_FONT_SIZE = int(os.getenv('REPORT_FONT_SIZE', '16'))

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.set_page_config(page_title="Advanced EDA Tool", layout="wide")
st.title("Automated Exploratory Data Analysis Tool")

# Add session state for persistent data
if 'df' not in st.session_state:
    st.session_state.df = None

# File upload with additional format support
uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "feather"])
use_sample = st.checkbox("Use random sample (for large datasets)", value=False)

def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]

def generate_energy_report(df, expected_months):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font(REPORT_FONT, 'B', REPORT_FONT_SIZE)
    
    # Header
    pdf.cell(0, 10, REPORT_TITLE, 0, 1, 'C')
    pdf.ln(10)
    
    # Function to add images
    def add_plot(fig):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight')
            pdf.image(tmpfile.name, x=10, w=190)
            plt.close(fig)
        pdf.ln(10)
    
    # 1. Basic Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Basic Metrics', 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Total Unique Meters: {df['meterNo'].nunique()}", 0, 1)
    
    # 2. Fuel Type Distribution
    fuel_dist = df.groupby('fuelType')['meterNo'].nunique().reset_index()
    fig = plt.figure()
    sns.barplot(y='fuelType', x='meterNo', data=fuel_dist)
    plt.title('Meter Distribution by Fuel Type')
    add_plot(fig)
    
    # 3. Estimated vs Actual
    est_actual = df.pivot_table(index='meterNo', columns='readingType', values='consumption', aggfunc='sum')
    pdf.cell(0, 10, f"Overall Estimated/Actual Ratio: {(est_actual['Estimated'].sum()/est_actual['Actual'].sum()*100):.1f}%", 0, 1)
    
    # 4. Monthly Coverage
    monthly_counts = df.groupby('meterNo')['month_year'].nunique().reset_index()
    coverage = (monthly_counts['month_year'].mean()/expected_months*100).round(1)
    pdf.cell(0, 10, f"Average Monthly Coverage: {coverage}%", 0, 1)
    
    # 5. Missing Data
    missing = df['consumption'].isnull().sum()
    pdf.cell(0, 10, f"Missing Consumption Records: {missing} ({missing/len(df)*100:.1f}%)", 0, 1)
    
    # 6. Duplicates
    duplicates = df.duplicated(subset=['meterNo', 'month', 'year']).sum()
    pdf.cell(0, 10, f"Duplicate Records: {duplicates}", 0, 1)
    
    # Save to bytes buffer
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

def generate_llm_insights(client, df, user_prompt, context=""):
    """Generate insights using OpenAI's LLM with dynamic context analysis"""
    if not client:
        return "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
        
    try:
        # Step 1: Perform dynamic dataset analysis
        analysis_report = analyze_dataset(df, user_prompt)
        
        # Step 2: Prepare context-aware prompt
        context = f"""
        Dataset Context:
        - Columns: {', '.join(df.columns)}
        - Total Records: {len(df)}
        
        Question-Specific Analysis:
        {json.dumps(analysis_report, indent=2)[:2000]}  # Truncate to stay within token limits
        """
        
        # Step 3: Generate LLM response
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": f"""You are a senior data analyst. Follow these steps:
1. Analyze the provided dataset context and question-specific analysis
2. Identify key patterns, anomalies, and relationships
3. Provide detailed numerical insights with exact values from the analysis
4. Use bullet points with emojis for clarity
5. Highlight potential data quality issues"""},
                {"role": "user", "content": f"Question: {user_prompt}\n\n{context}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        if APP_DEBUG:
            return f"Error generating insights: {str(e)}"
        return "An error occurred while generating insights. Please try again."

def analyze_dataset(df, question):
    """Dynamically analyze dataset based on user question context"""
    analysis = {}
    
    # Detect numeric columns mentioned in question
    mentioned_cols = [col for col in df.columns if col.lower() in question.lower()]
    numeric_cols = df.columns[
        df.columns.isin(mentioned_cols) & 
        df.columns.isin(df.select_dtypes(include=np.number).columns)
    ].tolist()
    
    # Helper function to convert pandas/numpy types to native Python types
    def convert_to_native(obj):
        if isinstance(obj, (np.generic)):
            return obj.item()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj
    
    # 1. Generate summary stats for mentioned numeric columns
    if numeric_cols:
        stats = df[numeric_cols].describe().applymap(convert_to_native).to_dict()
        analysis["summary_stats"] = stats
        
    # 2. Check distributions
    distributions = {}
    for col in numeric_cols:
        try:
            distributions[col] = {
                "skewness": round(float(df[col].skew()), 2),
                "kurtosis": round(float(df[col].kurtosis()), 2),
                "outliers": int(detect_outliers(df[col]))
            }
        except Exception as e:
            continue
    if distributions:
        analysis["distributions"] = distributions
    
    # 3. Correlation analysis for mentioned columns
    if len(numeric_cols) > 1:
        analysis["correlations"] = (
            df[numeric_cols]
            .corr()
            .applymap(convert_to_native)
            .to_dict()
        )
    
    # 4. Missing values analysis
    if mentioned_cols:
        missing = (
            df[mentioned_cols]
            .isnull()
            .sum()
            .apply(convert_to_native)
            .to_dict()
        )
        analysis["missing_values"] = missing
    
    # 5. Time series analysis if date columns exist
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols and numeric_cols:
        try:
            ts_data = (
                df.groupby(date_cols[0])[numeric_cols]
                .mean()
                .reset_index()
                .applymap(convert_to_native)
                .to_dict(orient='list')
            )
            analysis["time_series"] = {"trends": ts_data}
        except Exception as e:
            pass
    
    return analysis

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.feather'):
            df = pd.read_feather(uploaded_file)
        
        if use_sample:
            df = df.sample(min(MAX_SAMPLE_SIZE, len(df)))
        
        st.session_state.df = df.copy()
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

if st.session_state.df is not None:
    df = st.session_state.df
    
    # ===== Data Overview Section =====
    with st.expander("📊 Data Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            dup_count = df.duplicated().sum()
            st.metric("Duplicate Rows", dup_count, delta_color="off")
        
        if st.checkbox("Show sample data"):
            st.dataframe(df.head(10), height=300)
    
    # ===== Data Quality Analysis =====
    with st.expander("🔍 Data Quality Analysis"):
        # Missing values analysis
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_pct})
        st.subheader("Missing Values Analysis")
        st.dataframe(missing_df.style.background_gradient(cmap='Reds', axis=0), height=400)
        
        # Data types analysis
        st.subheader("Data Type Distribution")
        dtype_counts = df.dtypes.value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        fig, ax = plt.subplots()
        sns.barplot(x='Count', y='Data Type', data=dtype_counts, ax=ax)
        st.pyplot(fig)
        
        # Duplicates handling
        if dup_count > 0:
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.session_state.df = df
                st.experimental_rerun()
    
    # ===== Feature Profiling =====
    with st.expander("📈 Feature Profiling"):
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        tab1, tab2 = st.tabs(["Numerical Features", "Categorical Features"])
        
        with tab1:
            selected_num = st.multiselect("Select numerical features", num_cols)
            if selected_num:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribution Plot")
                    fig = plt.figure()
                    sns.histplot(df[selected_num[0]], kde=True)
                    st.pyplot(fig)
                with col2:
                    st.subheader("Box Plot")
                    fig = plt.figure()
                    sns.boxplot(y=df[selected_num[0]])
                    st.pyplot(fig)
        
        with tab2:
            selected_cat = st.multiselect("Select categorical features", cat_cols)
            if selected_cat:
                st.subheader("Frequency Distribution")
                fig = plt.figure(figsize=(10, 4))
                sns.countplot(y=df[selected_cat[0]], order=df[selected_cat[0]].value_counts().index)
                st.pyplot(fig)
    
    # ===== Advanced Visualizations =====
    with st.expander("🔬 Advanced Analysis"):
        # Correlation analysis
        st.subheader("Correlation Matrix")
        corr_matrix = df.select_dtypes(include=['number']).corr()
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig)
        
        # Pair plot for selected numerical features
        if len(num_cols) >= 2:
            st.subheader("Pairwise Relationships")
            pair_cols = st.multiselect("Select up to 5 numerical features", num_cols, default=num_cols[:2])
            if pair_cols:
                fig = sns.pairplot(df[pair_cols].sample(min(500, len(df))))
                st.pyplot(fig)
    
    # ===== Outlier Detection =====
    with st.expander("📌 Outlier Detection"):
        method = st.radio("Select detection method", ["IQR", "Z-score"])
        outlier_cols = st.multiselect("Select numerical columns", num_cols)
        
        if outlier_cols:
            results = []
            for col in outlier_cols:
                if method == "IQR":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))]
                else:
                    outliers = df[np.abs(stats.zscore(df[col])) > 3]
                
                results.append({
                    'Feature': col,
                    'Outlier Count': len(outliers),
                    'Outlier Percentage': f"{(len(outliers)/len(df))*100:.2f}%"
                })
            
            outliers_df = pd.DataFrame(results)
            st.dataframe(outliers_df.style.highlight_max(color='#ff8080'))
    
    # ===== Time Series Analysis =====
    if date_cols:
        with st.expander("⏳ Time Series Analysis"):
            date_col = st.selectbox("Select date column", date_cols)
            value_col = st.selectbox("Select value column", num_cols)
            
            if date_col and value_col:
                ts_df = df.set_index(date_col)[value_col]
                st.subheader(f"Time Series of {value_col}")
                st.line_chart(ts_df)
    
    # ===== Automated Insights =====
    with st.expander("💡 Automated Insights"):
        insights = []
        
        # Missing values insight
        high_missing = missing_pct[missing_pct > 30].index.tolist()
        if high_missing:
            insights.append(f"🚨 High missing values (>30%) in: {', '.join(high_missing)}")
        
        # Skewness detection
        skewed = df[num_cols].apply(lambda x: stats.skew(x.dropna())).abs()
        skewed = skewed[skewed > 1].index.tolist()
        if skewed:
            insights.append(f"⚠️ Potential skewness in: {', '.join(skewed)}")
        
        # Constant features check
        constant = [col for col in df.columns if df[col].nunique() == 1]
        if constant:
            insights.append(f"🔔 Constant features detected: {', '.join(constant)}")
        
        # Display insights
        if insights:
            st.write("\n\n".join(insights))
        else:
            st.success("No major data quality issues detected!")
    
    # ===== AI-Powered Insights =====
    with st.expander("💡 AI-Powered Insights", expanded=True):
        # Existing automated checks
        insights = []
        
        # Missing values insight
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 30].index.tolist()
        if high_missing:
            insights.append(f"🚨 High missing values (>30%) in: {', '.join(high_missing)}")
        
        # Skewness detection
        num_cols = df.select_dtypes(include=['number']).columns
        skewed = df[num_cols].apply(lambda x: stats.skew(x.dropna())).abs()
        skewed = skewed[skewed > 1].index.tolist()
        if skewed:
            insights.append(f"⚠️ Potential skewness in: {', '.join(skewed)}")
        
        # Constant features check
        constant = [col for col in df.columns if df[col].nunique() == 1]
        if constant:
            insights.append(f"🔔 Constant features detected: {', '.join(constant)}")
        
        # LLM Integration
        st.subheader("Ask About Your Data")
        user_question = st.text_input("Enter your analysis question (e.g., 'What patterns do you see in sales data?')", 
                                    key="llm_input")
        
        # Add API key and model selection
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        model_name = st.selectbox("Select Model", 
                                ["gpt-3.5-turbo", "gpt-4", "gpt-4o"], 
                                index=0,
                                help="GPT-4 requires special API access")
        
        if api_key and user_question:
            try:
                client = OpenAI(api_key=api_key)
                
                with st.spinner("Analyzing data with AI..."):
                    # Get mentioned columns from question
                    mentioned_columns = [col for col in df.columns if col.lower() in user_question.lower()]
                    
                    if mentioned_columns:
                        st.warning(f"Analysing mentioned columns: {', '.join(mentioned_columns)}")
                    
                    # Generate LLM response with dynamic context
                    llm_response = generate_llm_insights(
                        client=client,
                        df=df,
                        user_prompt=user_question
                    )
                    
                    st.subheader("AI Insights")
                    st.markdown(llm_response)
                    
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
        
        # Display basic automated insights
        if insights:
            st.subheader("Automated Quality Checks")
            st.write("\n\n".join(insights))
        else:
            st.success("No major data quality issues detected!")
                
    # ===== Energy Report QA Analysis =====
    with st.expander("🔋 Energy Report QA"):
        required_cols = ['meterNo', 'fuelType', 'consumption', 'readingType', 'month', 'year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Missing required columns for Energy QA: {', '.join(missing_cols)}")
        else:
            # 1. Unique Meters
            unique_meters = df['meterNo'].nunique()
            st.metric("Total Unique Meters", unique_meters)
            
            # 2. Meters per Fuel Type
            st.subheader("Meter Distribution by Fuel Type")
            fuel_dist = df.groupby('fuelType')['meterNo'].nunique().reset_index()
            fuel_dist.columns = ['Fuel Type', 'Meter Count']
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(fuel_dist.style.background_gradient(cmap='Blues', axis=0))
            with col2:
                fig = plt.figure()
                sns.barplot(y='Fuel Type', x='Meter Count', data=fuel_dist)
                st.pyplot(fig)
            
            # 3. Estimated vs Actual Consumption
            st.subheader("Estimated vs Actual Consumption Analysis")
            consumption_comparison = df.pivot_table(
                index='meterNo',
                columns='readingType',
                values='consumption',
                aggfunc='sum'
            ).reset_index()
            
            consumption_comparison['Estimated_vs_Actual'] = (
                (consumption_comparison['Estimated'] / consumption_comparison['Actual']) * 100
            ).round(2)
            
            # Display metrics
            total_estimated = consumption_comparison['Estimated'].sum()
            total_actual = consumption_comparison['Actual'].sum()
            overall_percentage = (total_estimated / total_actual * 100).round(2)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Estimated Consumption", f"{total_estimated:,.2f} units")
            col2.metric("Total Actual Consumption", f"{total_actual:,.2f} units")
            col3.metric("Overall Estimated/Actual", f"{overall_percentage}%")
            
            # 4. Estimated vs Actual Per Fuel Type
            st.subheader("Fuel Type Consumption Comparison")
            fuel_comparison = df.pivot_table(
                index='fuelType',
                columns='readingType',
                values='consumption',
                aggfunc='sum'
            ).reset_index()
            
            fuel_comparison['Estimated_vs_Actual'] = (
                (fuel_comparison['Estimated'] / fuel_comparison['Actual']) * 100
            ).round(2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(
                    fuel_comparison.style.format({
                        'Estimated': '{:,.2f}',
                        'Actual': '{:,.2f}',
                        'Estimated_vs_Actual': '{:.2f}%'
                    }).background_gradient(
                        subset=['Estimated_vs_Actual'], 
                        cmap='YlOrRd'
                    )
                )
            with col2:
                fig = plt.figure(figsize=(10, 6))
                fuel_comparison_melt = fuel_comparison.melt(id_vars='fuelType', 
                                                            value_vars=['Estimated', 'Actual'],
                                                            var_name='Type', value_name='Consumption')
                sns.barplot(x='Consumption', y='fuelType', hue='Type', data=fuel_comparison_melt)
                st.pyplot(fig)
            
            # 5. Monthly Data Coverage
            st.subheader("📅 Monthly Data Coverage Analysis")
            expected_months = st.number_input("Enter expected number of months per meter", 
                                            min_value=1, max_value=36, value=12)
            
            df['month_year'] = df['month'].astype(str) + '-' + df['year'].astype(str)
            monthly_counts = df.groupby('meterNo')['month_year'].nunique().reset_index()
            monthly_counts.columns = ['Meter', 'Months Available']
            monthly_counts['Months Missing'] = expected_months - monthly_counts['Months Available']
            monthly_counts['Coverage'] = (
                (monthly_counts['Months Available'] / expected_months) * 100
            ).round(1)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Meters with Complete Data ({expected_months} months):**")
                complete_meters = monthly_counts[monthly_counts['Months Available'] == expected_months]
                st.metric("Count", len(complete_meters))
            
            with col2:
                st.write("**Coverage Summary**")
                avg_coverage = monthly_counts['Coverage'].mean().round(1)
                st.metric("Average Coverage", f"{avg_coverage}%")
            
            st.write("**Detailed Coverage Report:**")
            st.dataframe(
                monthly_counts.style.background_gradient(
                    subset=['Coverage'], 
                    cmap='RdYlGn', 
                    vmin=0, 
                    vmax=100
                )
            )
            
            # 6. Meters Unavailable
            st.subheader("🚫 Meters with Missing Data")
            missing_consumption = df[df['consumption'].isnull()]
            missing_meters = missing_consumption['meterNo'].nunique()
            missing_percentage = round((missing_meters / unique_meters * 100), 1)
            
            col1, col2 = st.columns(2)
            col1.metric("Meters with Missing Consumption", missing_meters)
            col2.metric("Percentage of Total Meters", f"{missing_percentage}%")
            
            if missing_meters > 0:
                st.write("Affected Meters:")
                st.write(missing_consumption['meterNo'].unique())
            
            # 7. Duplicate Data Alert
            st.subheader("🔔 Duplicate Records Detection")
            duplicates = df.groupby(['meterNo', 'month', 'year']).size().reset_index(name='counts')
            duplicates = duplicates[duplicates['counts'] > 1]
            
            if not duplicates.empty:
                st.error(f"Found {len(duplicates)} duplicate time periods!")
                st.write("Duplicate Records:")
                st.dataframe(
                    duplicates.style.highlight_max(color='red'),
                    hide_index=True
                )
            else:
                st.success("No duplicate records found!")
    
    if st.button("📥 Generate PDF Report"):
        with st.spinner("Generating PDF Report..."):
            # Get expected months from session state
            expected_months = st.session_state.get('expected_months', 12)
            
            # Generate PDF
            pdf_bytes = generate_energy_report(df, expected_months)
            
            # Create download button
            st.download_button(
                label="Download Energy Report",
                data=pdf_bytes,
                file_name="energy_report.pdf",
                mime="application/pdf"
            )

# Add footer
st.markdown("---")
# st.markdown("*Advanced EDA Tool* | Inspired by [Advanced EDA Techniques](https://miykael.github.io/blog/2022/advanced_eda/)")