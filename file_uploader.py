import streamlit as st
import pandas as pd


def file_uploader_create(txt):
    uploaded_file = st.file_uploader(txt, type=['csv'])
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        return dataframe
