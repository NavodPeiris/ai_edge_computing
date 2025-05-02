import streamlit as st
import pandas as pd

# config is in .streamlit/secrets.toml
conn = st.connection("mysql", type="sql") 


def fetch_latest_data():
    query = """
        SELECT * FROM events
    """

    df = conn.query(query)
    print(df)
    return df

'''
fetch_latest_data()
'''