import streamlit as st
import base64
from PIL import Image, ImageDraw, ImageFont
import torch

import crowd_crawler
import weather_crawler
import traffic_crawler

import future_crowd_crawler
import future_weather_crawler
import future_traffic_crawler

import event_crawler
import holidays_crawler
import pandas as pd
from mlflow.models import Model
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from utils import edge_server_url, mlflow_tracking_uri, client, train_fn, inf

# used for supervised classification tasks
def create_supervised_dialog(model, model_path):
    """Factory function to create a supervised training dialog for a specific model."""
    @st.dialog(f"Upload Data for {model['name']}")
    def supervised_model_train_dialog():
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        label = st.text_input("Label Name:")
        rounds = str(int(st.number_input("Num of Rounds to train:", min_value=1, step=1)))

        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date Column Name:")

            enable_crowd_count = st.toggle("Add crowd count data", key="enable_crowd_count")
            if enable_crowd_count:
                cc_df = crowd_crawler.fetch_latest_data()

                cc_locs = cc_df["location"].unique().tolist()
                cc_selected_loc = st.multiselect("Pick locations", cc_locs, key="cc_selected_loc")

                # Filter the DataFrame based on selected locations
                cc_df = cc_df[cc_df["location"].isin(cc_selected_loc)]
                cc_df = cc_df.groupby("date").agg({
                    "count": "sum"
                }).reset_index()

                cc_df = cc_df.rename(columns={"date": date_col_name, "count": "crowd_count"})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime


            enable_traffic_count = st.toggle("Add traffic count data", key="enable_traffic_count")
            if enable_traffic_count:
                tc_df = traffic_crawler.fetch_latest_data()

                tc_locs = tc_df["location"].unique().tolist()
                tc_selected_loc = st.multiselect("Pick locations", tc_locs, key="tc_selected_loc")

                # Filter the DataFrame based on selected locations
                tc_df = tc_df[tc_df["location"].isin(tc_selected_loc)]
                tc_df = tc_df.groupby("date").agg({
                    "vehicles_coming_in": "sum",
                    "vehicles_going_out": "sum"
                }).reset_index()

                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            
            enable_weather_data = st.toggle("Add weather data", key="enable_weather_data")
            if enable_weather_data:
                w_df = weather_crawler.fetch_latest_data()

                w_locs = w_df["location"].unique().tolist()
                w_selected_loc = st.multiselect("Pick locations", w_locs, key="w_selected_loc")

                # Filter the DataFrame based on selected locations
                w_df = w_df[w_df["location"].isin(w_selected_loc)]
                w_df = w_df.groupby("date").agg({
                    'humidity': 'mean',
                    'rain': 'mean',
                    'temperature': 'mean'
                }).reset_index()

                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            
            enable_events_data = st.toggle("Add events data", key="enable_events_data")
            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()

                e_locs = e_df["location"].unique().tolist()
                e_selected_loc = st.multiselect("Pick locations", e_locs, key="e_selected_loc")

                # Filter the DataFrame based on selected locations
                e_df = e_df[e_df["location"].isin(e_selected_loc)]
                e_df = e_df.groupby("date").agg({
                    "estimated_visitors": "sum",
                    "location": "count"
                }).rename(columns={"location": "event_count"}).reset_index()

                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

                print(e_df)

            
            enable_holidays_data = st.toggle("Add holidays data", key="enable_holidays_data")
            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})

                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

        
        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                print("label:", label)
                print("rounds:", rounds)
                print("task:", model["task"])
                print("save_path:", model_path)
                if df is not None:
                    
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            print(df)
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                            
                    
                    st.write("Data after merging:")
                    st.dataframe(df)

        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            with st.spinner("Training in progress..."):
                train_fn(df, model_path, model["task"], [label], rounds, edge_server_url)
                st.write("Training Status: success")
                st.rerun()
                    
    return supervised_model_train_dialog


# used for unsupervised classification tasks and anomaly detection tasks
def create_unsupervised_dialog(model, model_path):
    """Factory function to create an unsupervised training dialog for a specific model."""
    @st.dialog(f"Upload Data for {model['name']}")
    def unsupervised_model_train_dialog():
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        labels = []
        rounds = str(int(st.number_input("Num of Rounds to train:", min_value=1, step=1)))

        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date Column Name:")

            enable_crowd_count = st.toggle("Add crowd count data", key="enable_crowd_count")
            if enable_crowd_count:
                cc_df = crowd_crawler.fetch_latest_data()

                cc_locs = cc_df["location"].unique().tolist()
                cc_selected_loc = st.multiselect("Pick locations", cc_locs, key="cc_selected_loc")

                # Filter the DataFrame based on selected locations
                cc_df = cc_df[cc_df["location"].isin(cc_selected_loc)]
                cc_df = cc_df.groupby("date").agg({
                    "count": "sum"
                }).reset_index()

                cc_df = cc_df.rename(columns={"date": date_col_name, "count": "crowd_count"})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime


            enable_traffic_count = st.toggle("Add traffic count data", key="enable_traffic_count")
            if enable_traffic_count:
                tc_df = traffic_crawler.fetch_latest_data()

                tc_locs = tc_df["location"].unique().tolist()
                tc_selected_loc = st.multiselect("Pick locations", tc_locs, key="tc_selected_loc")

                # Filter the DataFrame based on selected locations
                tc_df = tc_df[tc_df["location"].isin(tc_selected_loc)]
                tc_df = tc_df.groupby("date").agg({
                    "vehicles_coming_in": "sum",
                    "vehicles_going_out": "sum"
                }).reset_index()

                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            
            enable_weather_data = st.toggle("Add weather data", key="enable_weather_data")
            if enable_weather_data:
                w_df = weather_crawler.fetch_latest_data()

                w_locs = w_df["location"].unique().tolist()
                w_selected_loc = st.multiselect("Pick locations", w_locs, key="w_selected_loc")

                # Filter the DataFrame based on selected locations
                w_df = w_df[w_df["location"].isin(w_selected_loc)]
                w_df = w_df.groupby("date").agg({
                    'humidity': 'mean',
                    'rain': 'mean',
                    'temperature': 'mean'
                }).reset_index()

                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            
            enable_events_data = st.toggle("Add events data", key="enable_events_data")
            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()

                e_locs = e_df["location"].unique().tolist()
                e_selected_loc = st.multiselect("Pick locations", e_locs, key="e_selected_loc")

                # Filter the DataFrame based on selected locations
                e_df = e_df[e_df["location"].isin(e_selected_loc)]
                e_df = e_df.groupby("date").agg({
                    "estimated_visitors": "sum",
                    "location": "count"
                }).rename(columns={"location": "event_count"}).reset_index()

                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

                print(e_df)

            
            enable_holidays_data = st.toggle("Add holidays data", key="enable_holidays_data")
            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})

                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

        
        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                print("label:", "N/A")
                print("rounds:", rounds)
                print("task:", model["task"])
                print("save_path:", model_path)
                if df is not None:
                    
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            print(df)
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                            
                    
                    st.write("Data after merging:")
                    st.dataframe(df)

        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            with st.spinner("Training in progress..."):
                train_fn(df, model_path, model["task"], labels, rounds, edge_server_url)
                st.write("Training Status: success")
                st.rerun()

    return unsupervised_model_train_dialog


# used for forecasting tasks and regression tasks
def create_forecasting_dialog(model, model_path):
    """Factory function to create a supervised training dialog for a specific model."""
    @st.dialog(f"Upload Data for {model['name']}")
    def forecasting_model_train_dialog():
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        labels = st.text_input("Forecast column names (comma seperated):")
        rounds = str(int(st.number_input("Num of Rounds to train:", min_value=1, step=1)))

        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date Column Name:")

            enable_crowd_count = st.toggle("Add crowd count data", key="enable_crowd_count")
            if enable_crowd_count:
                cc_df = crowd_crawler.fetch_latest_data()

                cc_locs = cc_df["location"].unique().tolist()
                cc_selected_loc = st.multiselect("Pick locations", cc_locs, key="cc_selected_loc")

                # Filter the DataFrame based on selected locations
                cc_df = cc_df[cc_df["location"].isin(cc_selected_loc)]
                cc_df = cc_df.groupby("date").agg({
                    "count": "sum"
                }).reset_index()

                cc_df = cc_df.rename(columns={"date": date_col_name, "count": "crowd_count"})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime


            enable_traffic_count = st.toggle("Add traffic count data", key="enable_traffic_count")
            if enable_traffic_count:
                tc_df = traffic_crawler.fetch_latest_data()

                tc_locs = tc_df["location"].unique().tolist()
                tc_selected_loc = st.multiselect("Pick locations", tc_locs, key="tc_selected_loc")

                # Filter the DataFrame based on selected locations
                tc_df = tc_df[tc_df["location"].isin(tc_selected_loc)]
                tc_df = tc_df.groupby("date").agg({
                    "vehicles_coming_in": "sum",
                    "vehicles_going_out": "sum"
                }).reset_index()

                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            
            enable_weather_data = st.toggle("Add weather data", key="enable_weather_data")
            if enable_weather_data:
                w_df = weather_crawler.fetch_latest_data()

                w_locs = w_df["location"].unique().tolist()
                w_selected_loc = st.multiselect("Pick locations", w_locs, key="w_selected_loc")

                # Filter the DataFrame based on selected locations
                w_df = w_df[w_df["location"].isin(w_selected_loc)]
                w_df = w_df.groupby("date").agg({
                    'humidity': 'mean',
                    'rain': 'mean',
                    'temperature': 'mean'
                }).reset_index()

                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            
            enable_events_data = st.toggle("Add events data", key="enable_events_data")
            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()

                e_locs = e_df["location"].unique().tolist()
                e_selected_loc = st.multiselect("Pick locations", e_locs, key="e_selected_loc")

                # Filter the DataFrame based on selected locations
                e_df = e_df[e_df["location"].isin(e_selected_loc)]
                e_df = e_df.groupby("date").agg({
                    "estimated_visitors": "sum",
                    "location": "count"
                }).rename(columns={"location": "event_count"}).reset_index()

                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

                print(e_df)

            
            enable_holidays_data = st.toggle("Add holidays data", key="enable_holidays_data")
            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})

                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

                
        labels = labels.split(",") if labels else []

        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                print("labels:", labels)
                print("rounds:", rounds)
                print("task:", model["task"])
                print("save_path:", model_path)
                if df is not None:
                    
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            print(df)
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                            
                    
                    st.write("Data after merging:")
                    st.dataframe(df)

        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            with st.spinner("Training in progress..."):
                train_fn(df, model_path, model["task"], labels, rounds, edge_server_url)
                st.write("Training Status: success")
                st.rerun()
                    
    return forecasting_model_train_dialog


# used for supervised classification tasks
def create_supervised_model_pred_dialog(model, model_path):
    @st.dialog(f"Upload Data for {model['name']}")
    def supervised_model_pred_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        # Create a text input field
        label = st.text_input("Prediction Column Name:")

        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date Column Name:")

            enable_crowd_count = st.toggle("Add predicted crowd count data", key="enable_crowd_count")
            if enable_crowd_count:
                cc_df = future_crowd_crawler.fetch_latest_data()

                cc_locs = cc_df["location"].unique().tolist()
                cc_selected_loc = st.multiselect("Pick locations", cc_locs, key="cc_selected_loc")

                # Filter the DataFrame based on selected locations
                cc_df = cc_df[cc_df["location"].isin(cc_selected_loc)]
                cc_df = cc_df.groupby("date").agg({
                    "count": "sum"
                }).reset_index()

                cc_df = cc_df.rename(columns={"date": date_col_name, "count": "crowd_count"})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime


            enable_traffic_count = st.toggle("Add predicted traffic count data", key="enable_traffic_count")
            if enable_traffic_count:
                tc_df = future_traffic_crawler.fetch_latest_data()

                tc_locs = tc_df["location"].unique().tolist()
                tc_selected_loc = st.multiselect("Pick locations", tc_locs, key="tc_selected_loc")

                # Filter the DataFrame based on selected locations
                tc_df = tc_df[tc_df["location"].isin(tc_selected_loc)]
                tc_df = tc_df.groupby("date").agg({
                    "vehicles_coming_in": "sum",
                    "vehicles_going_out": "sum"
                }).reset_index()

                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            
            enable_weather_data = st.toggle("Add predicted weather data", key="enable_weather_data")
            if enable_weather_data:
                w_df = future_weather_crawler.fetch_latest_data()

                w_locs = w_df["location"].unique().tolist()
                w_selected_loc = st.multiselect("Pick locations", w_locs, key="w_selected_loc")

                # Filter the DataFrame based on selected locations
                w_df = w_df[w_df["location"].isin(w_selected_loc)]
                w_df = w_df.groupby("date").agg({
                    'humidity': 'mean',
                    'rain': 'mean',
                    'temperature': 'mean'
                }).reset_index()

                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            
            enable_events_data = st.toggle("Add events data", key="enable_events_data")
            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()

                e_locs = e_df["location"].unique().tolist()
                e_selected_loc = st.multiselect("Pick locations", e_locs, key="e_selected_loc")

                # Filter the DataFrame based on selected locations
                e_df = e_df[e_df["location"].isin(e_selected_loc)]
                e_df = e_df.groupby("date").agg({
                    "estimated_visitors": "sum",
                    "location": "count"
                }).rename(columns={"location": "event_count"}).reset_index()

                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

                print(e_df)

            
            enable_holidays_data = st.toggle("Add holidays data", key="enable_holidays_data")
            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})

                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

        
        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                print("label:", "N/A")
                print("task:", model["task"])
                print("save_path:", model_path)
                if df is not None:
                    
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            print(df)
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                            
                    
                    st.write("Data after merging:")
                    st.dataframe(df)

        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            try:
                # Reset file pointer to beginning
                file.seek(0)
                
                # Check if file is empty
                if file.tell() == file.seek(0, 2):  # seek to end and check position
                    st.error("The uploaded file is empty.")
                    return
                
                # Reset file pointer again
                file.seek(0)
                
                try:
                    # Try to read with pandas
                    df = pd.read_csv(file, encoding='utf-8')  # Explicitly specify encoding
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a different encoding
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')
                except pd.errors.EmptyDataError:
                    st.error("The file appears to be empty or corrupted. Please check the file content.")
                    return
                except Exception as e:
                    st.error(f"Error reading the file: {str(e)}\nPlease make sure it's a valid CSV file.")
                    return
                
                # Verify that we have data
                if df.empty:
                    st.error("The uploaded file contains no data.")
                    return
                
                df = df.head(10000)
                
                # Show status updates
                with st.spinner("Inference in progress..."):
                    results = inf(df, [label], model_path, model["task"], edge_server_url)
                    st.write("Prediction Results:")
                    st.dataframe(results)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                
    return supervised_model_pred_dialog


# used for unsupervised classification tasks
def create_unsupervised_model_pred_dialog(model, model_path):
    @st.dialog(f"Upload Data for {model['name']}")
    def unsupervised_model_pred_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        label = ""

        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date Column Name:")

            enable_crowd_count = st.toggle("Add predicted crowd count data", key="enable_crowd_count")
            if enable_crowd_count:
                cc_df = future_crowd_crawler.fetch_latest_data()

                cc_locs = cc_df["location"].unique().tolist()
                cc_selected_loc = st.multiselect("Pick locations", cc_locs, key="cc_selected_loc")

                # Filter the DataFrame based on selected locations
                cc_df = cc_df[cc_df["location"].isin(cc_selected_loc)]
                cc_df = cc_df.groupby("date").agg({
                    "count": "sum"
                }).reset_index()

                cc_df = cc_df.rename(columns={"date": date_col_name, "count": "crowd_count"})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime


            enable_traffic_count = st.toggle("Add predicted traffic count data", key="enable_traffic_count")
            if enable_traffic_count:
                tc_df = future_traffic_crawler.fetch_latest_data()

                tc_locs = tc_df["location"].unique().tolist()
                tc_selected_loc = st.multiselect("Pick locations", tc_locs, key="tc_selected_loc")

                # Filter the DataFrame based on selected locations
                tc_df = tc_df[tc_df["location"].isin(tc_selected_loc)]
                tc_df = tc_df.groupby("date").agg({
                    "vehicles_coming_in": "sum",
                    "vehicles_going_out": "sum"
                }).reset_index()

                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            
            enable_weather_data = st.toggle("Add predicted weather data", key="enable_weather_data")
            if enable_weather_data:
                w_df = future_weather_crawler.fetch_latest_data()

                w_locs = w_df["location"].unique().tolist()
                w_selected_loc = st.multiselect("Pick locations", w_locs, key="w_selected_loc")

                # Filter the DataFrame based on selected locations
                w_df = w_df[w_df["location"].isin(w_selected_loc)]
                w_df = w_df.groupby("date").agg({
                    'humidity': 'mean',
                    'rain': 'mean',
                    'temperature': 'mean'
                }).reset_index()

                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            
            enable_events_data = st.toggle("Add events data", key="enable_events_data")
            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()

                e_locs = e_df["location"].unique().tolist()
                e_selected_loc = st.multiselect("Pick locations", e_locs, key="e_selected_loc")

                # Filter the DataFrame based on selected locations
                e_df = e_df[e_df["location"].isin(e_selected_loc)]
                e_df = e_df.groupby("date").agg({
                    "estimated_visitors": "sum",
                    "location": "count"
                }).rename(columns={"location": "event_count"}).reset_index()

                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

                print(e_df)

            
            enable_holidays_data = st.toggle("Add holidays data", key="enable_holidays_data")
            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})

                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

        
        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                print("label:", "N/A")
                print("task:", model["task"])
                print("save_path:", model_path)
                if df is not None:
                    
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            print(df)
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                            
                    
                    st.write("Data after merging:")
                    st.dataframe(df)

        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            try:
                # Reset file pointer to beginning
                file.seek(0)
                
                # Check if file is empty
                if file.tell() == file.seek(0, 2):  # seek to end and check position
                    st.error("The uploaded file is empty.")
                    return
                
                # Reset file pointer again
                file.seek(0)
                
                try:
                    # Try to read with pandas
                    df = pd.read_csv(file, encoding='utf-8')  # Explicitly specify encoding
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a different encoding
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')
                except pd.errors.EmptyDataError:
                    st.error("The file appears to be empty or corrupted. Please check the file content.")
                    return
                except Exception as e:
                    st.error(f"Error reading the file: {str(e)}\nPlease make sure it's a valid CSV file.")
                    return
                
                # Verify that we have data
                if df.empty:
                    st.error("The uploaded file contains no data.")
                    return
                
                df = df.head(10000)
                
                # Show status updates
                with st.spinner("Inference in progress..."):
                    results = inf(df, [label], model_path, model["task"], edge_server_url)
                    st.write("Prediction Results:")
                    # Display in Streamlit
                    st.title("UMAP Visualization of Encoded Features")
                    # Create a Matplotlib figure
                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(results["umap_1"], results["umap_2"], c=np.arange(len(results)), cmap="viridis", alpha=0.7)
                    ax.set_xlabel("UMAP 1")
                    ax.set_ylabel("UMAP 2")
                    ax.set_title("UMAP Projection of Latent Space")
                    plt.colorbar(scatter, ax=ax)
                    # Show plot in Streamlit
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")

    return unsupervised_model_pred_dialog


# used for anomaly detection tasks
def create_anomaly_model_pred_dialog(model, model_path):
    @st.dialog(f"Upload Data for {model['name']}")
    def anomaly_model_pred_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        label = "Anomaly"

        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date Column Name:")

            enable_crowd_count = st.toggle("Add predicted crowd count data", key="enable_crowd_count")
            if enable_crowd_count:
                cc_df = future_crowd_crawler.fetch_latest_data()

                cc_locs = cc_df["location"].unique().tolist()
                cc_selected_loc = st.multiselect("Pick locations", cc_locs, key="cc_selected_loc")

                # Filter the DataFrame based on selected locations
                cc_df = cc_df[cc_df["location"].isin(cc_selected_loc)]
                cc_df = cc_df.groupby("date").agg({
                    "count": "sum"
                }).reset_index()

                cc_df = cc_df.rename(columns={"date": date_col_name, "count": "crowd_count"})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime


            enable_traffic_count = st.toggle("Add predicted traffic count data", key="enable_traffic_count")
            if enable_traffic_count:
                tc_df = future_traffic_crawler.fetch_latest_data()

                tc_locs = tc_df["location"].unique().tolist()
                tc_selected_loc = st.multiselect("Pick locations", tc_locs, key="tc_selected_loc")

                # Filter the DataFrame based on selected locations
                tc_df = tc_df[tc_df["location"].isin(tc_selected_loc)]
                tc_df = tc_df.groupby("date").agg({
                    "vehicles_coming_in": "sum",
                    "vehicles_going_out": "sum"
                }).reset_index()

                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            
            enable_weather_data = st.toggle("Add predicted weather data", key="enable_weather_data")
            if enable_weather_data:
                w_df = future_weather_crawler.fetch_latest_data()

                w_locs = w_df["location"].unique().tolist()
                w_selected_loc = st.multiselect("Pick locations", w_locs, key="w_selected_loc")

                # Filter the DataFrame based on selected locations
                w_df = w_df[w_df["location"].isin(w_selected_loc)]
                w_df = w_df.groupby("date").agg({
                    'humidity': 'mean',
                    'rain': 'mean',
                    'temperature': 'mean'
                }).reset_index()

                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            
            enable_events_data = st.toggle("Add events data", key="enable_events_data")
            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()

                e_locs = e_df["location"].unique().tolist()
                e_selected_loc = st.multiselect("Pick locations", e_locs, key="e_selected_loc")

                # Filter the DataFrame based on selected locations
                e_df = e_df[e_df["location"].isin(e_selected_loc)]
                e_df = e_df.groupby("date").agg({
                    "estimated_visitors": "sum",
                    "location": "count"
                }).rename(columns={"location": "event_count"}).reset_index()

                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

                print(e_df)

            
            enable_holidays_data = st.toggle("Add holidays data", key="enable_holidays_data")
            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})

                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

        
        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                print("label:", "N/A")
                print("task:", model["task"])
                print("save_path:", model_path)
                if df is not None:
                    
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            print(df)
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                            
                    
                    st.write("Data after merging:")
                    st.dataframe(df)

        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            try:
                # Reset file pointer to beginning
                file.seek(0)
                
                # Check if file is empty
                if file.tell() == file.seek(0, 2):  # seek to end and check position
                    st.error("The uploaded file is empty.")
                    return
                
                # Reset file pointer again
                file.seek(0)
                
                try:
                    # Try to read with pandas
                    df = pd.read_csv(file, encoding='utf-8')  # Explicitly specify encoding
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a different encoding
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')
                except pd.errors.EmptyDataError:
                    st.error("The file appears to be empty or corrupted. Please check the file content.")
                    return
                except Exception as e:
                    st.error(f"Error reading the file: {str(e)}\nPlease make sure it's a valid CSV file.")
                    return
                
                # Verify that we have data
                if df.empty:
                    st.error("The uploaded file contains no data.")
                    return
                
                df = df.head(10000)
                
                # Show status updates
                with st.spinner("Inference in progress..."):
                    results = inf(df, [label], model_path, model["task"], edge_server_url)
                    st.dataframe(results)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    return anomaly_model_pred_dialog


# used for forecasting tasks and regression tasks
def create_forecasting_model_pred_dialog(model, model_path):
    @st.dialog(f"Upload Data for {model['name']}")
    def forecasting_model_pred_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        labels = st.text_input("Forecast column names (comma seperated):")
        
        labels = labels.split(",") if labels else []

        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date Column Name:")

            enable_crowd_count = st.toggle("Add predicted crowd count data", key="enable_crowd_count")
            if enable_crowd_count:
                cc_df = future_crowd_crawler.fetch_latest_data()

                cc_locs = cc_df["location"].unique().tolist()
                cc_selected_loc = st.multiselect("Pick locations", cc_locs, key="cc_selected_loc")

                # Filter the DataFrame based on selected locations
                cc_df = cc_df[cc_df["location"].isin(cc_selected_loc)]
                cc_df = cc_df.groupby("date").agg({
                    "count": "sum"
                }).reset_index()

                cc_df = cc_df.rename(columns={"date": date_col_name, "count": "crowd_count"})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime


            enable_traffic_count = st.toggle("Add predicted traffic count data", key="enable_traffic_count")
            if enable_traffic_count:
                tc_df = future_traffic_crawler.fetch_latest_data()

                tc_locs = tc_df["location"].unique().tolist()
                tc_selected_loc = st.multiselect("Pick locations", tc_locs, key="tc_selected_loc")

                # Filter the DataFrame based on selected locations
                tc_df = tc_df[tc_df["location"].isin(tc_selected_loc)]
                tc_df = tc_df.groupby("date").agg({
                    "vehicles_coming_in": "sum",
                    "vehicles_going_out": "sum"
                }).reset_index()

                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            
            enable_weather_data = st.toggle("Add predicted weather data", key="enable_weather_data")
            if enable_weather_data:
                w_df = future_weather_crawler.fetch_latest_data()

                w_locs = w_df["location"].unique().tolist()
                w_selected_loc = st.multiselect("Pick locations", w_locs, key="w_selected_loc")

                # Filter the DataFrame based on selected locations
                w_df = w_df[w_df["location"].isin(w_selected_loc)]
                w_df = w_df.groupby("date").agg({
                    'humidity': 'mean',
                    'rain': 'mean',
                    'temperature': 'mean'
                }).reset_index()

                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            
            enable_events_data = st.toggle("Add events data", key="enable_events_data")
            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()

                e_locs = e_df["location"].unique().tolist()
                e_selected_loc = st.multiselect("Pick locations", e_locs, key="e_selected_loc")

                # Filter the DataFrame based on selected locations
                e_df = e_df[e_df["location"].isin(e_selected_loc)]
                e_df = e_df.groupby("date").agg({
                    "estimated_visitors": "sum",
                    "location": "count"
                }).rename(columns={"location": "event_count"}).reset_index()

                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

                print(e_df)

            
            enable_holidays_data = st.toggle("Add holidays data", key="enable_holidays_data")
            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})

                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

        
        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                print("label:", "N/A")
                print("task:", model["task"])
                print("save_path:", model_path)
                if df is not None:
                    
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            print(df)
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                            
                    
                    st.write("Data after merging:")
                    st.dataframe(df)

        else:
            st.button("Merge", key="merge-disabled", disabled=True)


        if st.button("Submit"):
            try:
                # Reset file pointer to beginning
                file.seek(0)
                
                # Check if file is empty
                if file.tell() == file.seek(0, 2):  # seek to end and check position
                    st.error("The uploaded file is empty.")
                    return
                
                # Reset file pointer again
                file.seek(0)
                
                try:
                    # Try to read with pandas
                    df = pd.read_csv(file, encoding='utf-8')  # Explicitly specify encoding
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with a different encoding
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')
                except pd.errors.EmptyDataError:
                    st.error("The file appears to be empty or corrupted. Please check the file content.")
                    return
                except Exception as e:
                    st.error(f"Error reading the file: {str(e)}\nPlease make sure it's a valid CSV file.")
                    return
                
                # Verify that we have data
                if df.empty:
                    st.error("The uploaded file contains no data.")
                    return
                
                df = df.head(10000)
                
                # Show status updates
                with st.spinner("Inference in progress..."):
                    results = inf(df, labels, model_path, model["task"], edge_server_url)
                    st.dataframe(results)
            except Exception as e:
                st.error(f"Error: {e}")

    return forecasting_model_pred_dialog


def create_registry_model_pred_dialog(model_name, model_path, task_type):
    @st.dialog(f"Use {model_name}")
    def registry_model_pred_dialog():
        try:
            model = mlflow.pyfunc.load_model(f"models/registry/{model_name}")
        except Exception as e:
            try:
                print(e)
                model = mlflow.transformers.load_model(f"models/registry/{model_name}")
            except Exception as e:
                print(e)
                model = mlflow.tensorflow.load_model(f"models/registry/{model_name}")

        # Load metadata (no model loaded)
        model_metadata = Model.load(model_path)

        # Access signature
        signature = model_metadata.signature
        #print("Input schema:", signature.inputs)
        #print("Output schema:", signature.outputs)

        # Get input schema
        input_schema = signature.inputs.to_dict()

        # Get output schema
        output_schema = signature.outputs

        print("Input schema:", input_schema)
        print("Output schema:", output_schema)

        input_dict = {}
        input_list = []

        for i in range(len(input_schema)):
            if input_schema[i]["type"] == "string":
                input = st.text_input("Input(string):", key=f"input-{i}")
                if input_schema[i].get("name", None) is not None:
                    input_dict[input_schema[i]["name"]] = input
                else:
                    input_list.append(input)

            elif input_schema[i]["type"] == "integer":
                input = int(st.number_input("Input(float/double):", min_value=0, step=1, key=f"input-{i}"))
                if input_schema[i].get("name", None) is not None:
                    input_dict[input_schema[i]["name"]] = input
                else:
                    input_list.append(input)

            elif input_schema[i]["type"] == "float" or input_schema[i]["type"] == "double":
                input = st.number_input("Input(float/double):", key=f"input-{i}")
                if input_schema[i].get("name", None) is not None:
                    input_dict[input_schema[i]["name"]] = input
                else:
                    input_list.append(input)

            elif input_schema[i]["type"] == "binary":
                input = st.file_uploader("Upload file:", key=f"input-{i}")
                if input is not None:
                    # Read the file and convert to bytes
                    input_bytes = input.read()
                    base64_str = base64.b64encode(input_bytes).decode('utf-8')
                    if input_schema[i].get("name", None) is not None:
                        input_dict[input_schema[i]["name"]] = base64_str
                    else:
                        input_list.append(base64_str)


        if st.button("Submit"):
            if input is not None:
                try:
                    # Show status updates
                    with st.spinner("Inference in progress..."):
                        if len(input_list) == 0:
                            results = model.predict(input_dict)
                        else:
                            results = model.predict(*input_list)
                        st.write("Results:")

                        if task_type == "object-detection":
                            # Annotate image
                            image = Image.open(input).convert("RGB")
                            draw = ImageDraw.Draw(image)
                            font = ImageFont.load_default(size=20)

                            for det in results:
                                box = det["box"]
                                label = f"{det['label']} ({det['score']:.2f})"
                                
                                draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline="yellow", width=4)
                                draw.text((box["xmin"], box["ymin"] - 20), label, fill="yellow", font=font)

                            # Streamlit display
                            st.image(image, caption="Detected Objects")
                        
                        elif isinstance(results, list):
                            for result in results:
                                for key, value in result.items():
                                    if isinstance(value, Image.Image):
                                        st.subheader(f"{key}:")
                                        st.image(value, caption=key)
                                    elif isinstance(value, torch.Tensor):
                                        st.subheader(f"{key}:")
                                        st.write(value.cpu().numpy())  # convert to readable format
                                    else:
                                        st.subheader(f"{key}:")
                                        st.write(value)
                        else:
                            for key, value in results.items():
                                if isinstance(value, Image.Image):
                                    st.subheader(f"{key}:")
                                    st.image(value, caption=key)
                                elif isinstance(value, torch.Tensor):
                                    st.subheader(f"{key}:")
                                    st.write(value.cpu().numpy())  # convert to readable format
                                else:
                                    st.subheader(f"{key}:")
                                    st.write(value)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    return registry_model_pred_dialog

def create_supervised_model_test_dialog(model, model_path):
    @st.dialog(f"Test {model['name']}")
    def supervised_model_test_dialog():
        # File uploader for test data
        file = st.file_uploader("Upload test data (Excel or CSV file)", type=["xlsx", "xls", "csv"])
        label = st.text_input("Target column name:")
        
        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date column name:")
            enable_crowd_data = st.toggle("Include crowd data", key="enable_crowd_data")
            enable_traffic_data = st.toggle("Include traffic data", key="enable_traffic_data")
            enable_weather_data = st.toggle("Include weather data", key="enable_weather_data")
            enable_events_data = st.toggle("Include events data", key="enable_events_data")
            enable_holidays_data = st.toggle("Include holidays data", key="enable_holidays_data")

            if enable_crowd_data:
                cc_df = crowd_crawler.fetch_latest_data()
                cc_df = cc_df.rename(columns={"date": date_col_name})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])   # Convert to datetime

            if enable_traffic_data:
                tc_df = traffic_crawler.fetch_latest_data()
                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])   # Convert to datetime

            if enable_weather_data:
                w_df = weather_crawler.fetch_latest_data()
                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])   # Convert to datetime

            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()
                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])   # Convert to datetime

            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})
                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])   # Convert to datetime

        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                if df is not None:
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])   # Convert to datetime

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                    
                    st.write("Data after merging:")
                    st.dataframe(df)
        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            try:
                # Store actual values
                actual_values = df[label].values
                # Drop the target column for prediction
                test_df = df.drop(columns=[label])
                
                # Get predictions
                with st.spinner("Testing in progress..."):
                    results = inf(test_df, [label], model_path, model["task"], edge_server_url)
                    predictions = results[label].values
                    
                    # Calculate accuracy
                    if model["task"] == "classification":
                        accuracy = np.mean(predictions == actual_values) * 100
                        st.write(f"Test Accuracy: {accuracy:.2f}%")
                        
                        # Create confusion matrix
                        cm = confusion_matrix(actual_values, predictions)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                        plt.title('Confusion Matrix')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        st.pyplot(fig)
                        
                    else:  # For regression tasks
                        mse = mean_squared_error(actual_values, predictions)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(actual_values, predictions)
                        r2 = r2_score(actual_values, predictions)
                        
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"Root Mean Squared Error: {rmse:.4f}")
                        st.write(f"Mean Absolute Error: {mae:.4f}")
                        st.write(f"R² Score: {r2:.4f}")
                        
                        # Plot actual vs predicted values
                        fig, ax = plt.subplots()
                        plt.scatter(actual_values, predictions)
                        plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'r--', lw=2)
                        plt.title('Actual vs Predicted Values')
                        plt.xlabel('Actual Values')
                        plt.ylabel('Predicted Values')
                        st.pyplot(fig)
                    
                    # Show detailed results
                    results_df = pd.DataFrame({
                        'Actual': actual_values,
                        'Predicted': predictions
                    })
                    st.write("Detailed Results:")
                    st.dataframe(results_df)
                    
            except Exception as e:
                st.error(f"Error: {e}")
                
    return supervised_model_test_dialog

def create_forecasting_model_test_dialog(model, model_path):
    @st.dialog(f"Test {model['name']}")
    def forecasting_model_test_dialog():
        # File uploader for test data
        file = st.file_uploader("Upload test data (Excel or CSV file)", type=["xlsx", "xls", "csv"])
        labels = st.text_input("Target column names (comma separated):")
        
        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date column name:")
            enable_crowd_data = st.toggle("Include crowd data", key="enable_crowd_data")
            enable_traffic_data = st.toggle("Include traffic data", key="enable_traffic_data")
            enable_weather_data = st.toggle("Include weather data", key="enable_weather_data")
            enable_events_data = st.toggle("Include events data", key="enable_events_data")
            enable_holidays_data = st.toggle("Include holidays data", key="enable_holidays_data")

            if enable_crowd_data:
                cc_df = crowd_crawler.fetch_latest_data()
                cc_df = cc_df.rename(columns={"date": date_col_name})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])

            if enable_traffic_data:
                tc_df = traffic_crawler.fetch_latest_data()
                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])

            if enable_weather_data:
                w_df = weather_crawler.fetch_latest_data()
                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])

            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()
                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])

            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})
                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])

        labels = labels.split(",") if labels else []
        
        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                if df is not None:
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                    
                    st.write("Data after merging:")
                    st.dataframe(df)
        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            try:
                # Store actual values for each target
                actual_values = {label: df[label].values for label in labels}
                # Drop the target columns for prediction
                test_df = df.drop(columns=labels)
                
                # Get predictions
                with st.spinner("Testing in progress..."):
                    results = inf(test_df, labels, model_path, model["task"], edge_server_url)
                    
                    # For each target variable
                    for label in labels:
                        predictions = results[label].values
                        st.subheader(f"Results for {label}")
                        
                        # Calculate metrics
                        mse = mean_squared_error(actual_values[label], predictions)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(actual_values[label], predictions)
                        r2 = r2_score(actual_values[label], predictions)
                        
                        st.write(f"Mean Squared Error: {mse:.4f}")
                        st.write(f"Root Mean Squared Error: {rmse:.4f}")
                        st.write(f"Mean Absolute Error: {mae:.4f}")
                        st.write(f"R² Score: {r2:.4f}")
                        
                        # Plot actual vs predicted values
                        fig, ax = plt.subplots()
                        plt.scatter(actual_values[label], predictions)
                        plt.plot([actual_values[label].min(), actual_values[label].max()], [actual_values[label].min(), actual_values[label].max()], 'r--', lw=2)
                        plt.title(f'Actual vs Predicted Values - {label}')
                        plt.xlabel('Actual Values')
                        plt.ylabel('Predicted Values')
                        st.pyplot(fig)
                        
                        # Show detailed results
                        results_df = pd.DataFrame({
                            'Actual': actual_values[label],
                            'Predicted': predictions
                        })
                        st.write("Detailed Results:")
                        st.dataframe(results_df)
                    
            except Exception as e:
                st.error(f"Error: {e}")
                
    return forecasting_model_test_dialog

def create_unsupervised_model_test_dialog(model, model_path):
    @st.dialog(f"Test {model['name']}")
    def unsupervised_model_test_dialog():
        st.warning("Testing for unsupervised models is not available as they don't have ground truth labels.")
        st.info("Please use the 'Use Model' button to visualize the clustering results.")
                
    return unsupervised_model_test_dialog

def create_anomaly_model_test_dialog(model, model_path):
    @st.dialog(f"Test {model['name']}")
    def anomaly_model_test_dialog():
        # File uploader for test data
        file = st.file_uploader("Upload test data (Excel or CSV file)", type=["xlsx", "xls", "csv"])
        label = st.text_input("Anomaly label column name (1 for anomaly, 0 for normal):")
        
        has_datecol = st.toggle("Does dataset have a date column?", key="has_datecol")

        date_col_name = None
        cc_df = None
        tc_df = None
        w_df = None
        e_df = None
        h_df = None

        if has_datecol:
            date_col_name = st.text_input("Date column name:")
            enable_crowd_data = st.toggle("Include crowd data", key="enable_crowd_data")
            enable_traffic_data = st.toggle("Include traffic data", key="enable_traffic_data")
            enable_weather_data = st.toggle("Include weather data", key="enable_weather_data")
            enable_events_data = st.toggle("Include events data", key="enable_events_data")
            enable_holidays_data = st.toggle("Include holidays data", key="enable_holidays_data")

            if enable_crowd_data:
                cc_df = crowd_crawler.fetch_latest_data()
                cc_df = cc_df.rename(columns={"date": date_col_name})
                cc_df[date_col_name] = pd.to_datetime(cc_df[date_col_name])

            if enable_traffic_data:
                tc_df = traffic_crawler.fetch_latest_data()
                tc_df = tc_df.rename(columns={"date": date_col_name})
                tc_df[date_col_name] = pd.to_datetime(tc_df[date_col_name])

            if enable_weather_data:
                w_df = weather_crawler.fetch_latest_data()
                w_df = w_df.rename(columns={"date": date_col_name})
                w_df[date_col_name] = pd.to_datetime(w_df[date_col_name])

            if enable_events_data:
                e_df = event_crawler.fetch_latest_data()
                e_df = e_df.rename(columns={"date": date_col_name})
                e_df[date_col_name] = pd.to_datetime(e_df[date_col_name])

            if enable_holidays_data:
                h_df = holidays_crawler.fetch_latest_data()
                h_df["holiday_name"] = True
                h_df = h_df.rename(columns={"holiday_name": "is_holiday"})
                h_df = h_df.rename(columns={"date": date_col_name})
                h_df[date_col_name] = pd.to_datetime(h_df[date_col_name])

        df = None
        if file is not None:
            df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            df = df.head(10000)

        if cc_df is not None or tc_df is not None or w_df is not None or e_df is not None or h_df is not None:
            if st.button("Merge", key="merge-enabled"):
                if df is not None:
                    if date_col_name is not None:
                        df[date_col_name] = pd.to_datetime(df[date_col_name])

                        if cc_df is not None:
                            df = pd.merge(df, cc_df, on=date_col_name, how="left")
                            for col in ['crowd_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if tc_df is not None:
                            df = pd.merge(df, tc_df, on=date_col_name, how="left")
                            for col in ['vehicles_coming_in', 'vehicles_going_out']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if w_df is not None:
                            df = pd.merge(df, w_df, on=date_col_name, how="left")
                            for col in ['humidity', 'rain', 'temperature']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0
                            
                        if e_df is not None:
                            df = pd.merge(df, e_df, on=date_col_name, how="left")
                            for col in ['estimated_visitors', 'event_count']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(df[col].mean()).round().astype(int)
                                else:
                                    df[col] = 0

                        if h_df is not None:
                            df = pd.merge(df, h_df, on=date_col_name, how="left")
                            for col in ['is_holiday']:
                                if col in df.columns and not df[col].isna().all():
                                    df[col] = df[col].replace([float('inf'), float('-inf')], pd.NA).fillna(False)
                                else:
                                    df[col] = False
                    
                    st.write("Data after merging:")
                    st.dataframe(df)
        else:
            st.button("Merge", key="merge-disabled", disabled=True)

        if st.button("Submit"):
            try:
                # Store actual anomaly labels
                actual_values = df[label].values
                # Drop the label column for prediction
                test_df = df.drop(columns=[label])
                
                # Get predictions
                with st.spinner("Testing in progress..."):
                    results = inf(test_df, [label], model_path, model["task"], edge_server_url)
                    predictions = results[label].values
                    
                    # Calculate metrics
                    accuracy = accuracy_score(actual_values, predictions) * 100
                    precision = precision_score(actual_values, predictions) * 100
                    recall = recall_score(actual_values, predictions) * 100
                    f1 = f1_score(actual_values, predictions) * 100
                    
                    st.write(f"Accuracy: {accuracy:.2f}%")
                    st.write(f"Precision: {precision:.2f}%")
                    st.write(f"Recall: {recall:.2f}%")
                    st.write(f"F1 Score: {f1:.2f}%")
                    
                    # Create confusion matrix
                    cm = confusion_matrix(actual_values, predictions)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    st.pyplot(fig)
                    
                    # Show detailed results
                    results_df = pd.DataFrame({
                        'Actual': actual_values,
                        'Predicted': predictions
                    })
                    st.write("Detailed Results:")
                    st.dataframe(results_df)
                    
            except Exception as e:
                st.error(f"Error: {e}")
                
    return anomaly_model_test_dialog
    return anomaly_model_test_dialog