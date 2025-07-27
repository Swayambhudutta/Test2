import streamlit as st
import pandas as pd

# Set the title of the dashboard
st.title("Excel Viewer Dashboard")

# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)

        # Display the contents of the Excel file
        st.subheader("Excel File Contents")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
else:
    st.info("Please upload an Excel file to view its contents.")
