import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Power Demand Trend Analysis")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Power Demand Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # Combine Date and Time into a single datetime column
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))

        # Sort data
        df.sort_values(by=['State', 'Datetime'], inplace=True)

        # Show raw data
        st.subheader("📄 Raw Data Preview")
        st.dataframe(df)

        # Plot trend analysis
        st.subheader("📊 Power Demand Trend by State")
        fig = px.line(df, x='Datetime', y='Power Demand (MW)', color='State',
                      title='Power Demand Trend by State Over Time',
                      labels={'Power Demand (MW)': 'Power Demand (MW)', 'Datetime': 'Time'})
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload an Excel file to begin analysis.")
