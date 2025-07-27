import streamlit as st
import pandas as pd

st.title("Excel Viewer Dashboard")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Try importing openpyxl
        import openpyxl

        # Read the Excel file
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.subheader("Excel File Contents")
        st.dataframe(df)

    except ImportError:
        st.error("Missing dependency 'openpyxl'. Please install it using `pip install openpyxl`.")
    except Exception as e:
        st.error(f"Error reading the Excel file: {e}")
else:
    st.info("Please upload an Excel file to view its contents.")
