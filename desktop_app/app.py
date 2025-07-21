import streamlit as st
import pandas as pd
import os
import json
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import shutil
from mlflow.models import Model
from utils import edge_server_url, grafana_url, mlflow_tracking_uri, client
from dialogs import (
    create_supervised_dialog,
    create_supervised_model_pred_dialog,
    create_unsupervised_dialog,
    create_unsupervised_model_pred_dialog,
    create_forecasting_dialog,
    create_forecasting_model_pred_dialog,
    create_anomaly_model_pred_dialog,
    create_registry_model_pred_dialog
)

import requests
import pickle
import shutil

# Set page configuration
st.set_page_config(
    page_title="Edge Runner",
    initial_sidebar_state="expanded",
    page_icon="bizsuite_logo_no_background.ico",
    layout="wide",  # Use wide layout
)


def load_session_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
                loaded_state = pickle.load(f)
                for k, v in loaded_state.items():
                    st.session_state[k] = v
    else:
        print("session state file not found.")

#load_session_state("session_state.pkl")


if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You are not logged in.")
    st.stop()

st.title("Welcome to the Edge Runner Home Page!")
st.success(f"Welcome {st.session_state['username']}!")

os.makedirs(f"models/{st.session_state.user_id}", exist_ok=True)
os.makedirs(f"models/registry", exist_ok=True)
os.makedirs(f"{st.session_state.user_id}", exist_ok=True)

if st.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_id = ""
    if os.path.exists("session_state.pkl"):
        os.remove("session_state.pkl")
    st.switch_page("pages/Login.py") 

if not os.path.exists(f"{st.session_state.user_id}/models.json"):
    # if json is on server it will send it, otherwise a json with empty models list will be saved
    response = requests.get(f"{edge_server_url}/deliver_models_json/{st.session_state.user_id}")

    if response.status_code == 200:
        with open(f"{st.session_state.user_id}/models.json", "wb") as f:
            f.write(response.content)
        print(f"File saved to models.json")
    elif response.status_code == 404:
        default_models_json = {
            "models": [

            ]
        }
        with open(f"{st.session_state.user_id}/models.json", "w") as f:
            json.dump(default_models_json, f, indent=4)
    else:
        print(f"Failed to download. Status code: {response.status_code}")

# Read JSON file into a dictionary
with open(f"{st.session_state.user_id}/models.json", "r") as file:
    models = json.load(file)


# Inject custom CSS for layout adjustments
st.markdown("""
    <style>
        /* Increase padding around the main container */
        .main {
            padding: 2rem 2rem; /* Adjusted padding values */
        }

        /* Adjust the page width */
        section.main > div {
            max-width: 95%;
            padding: 2rem; /* Adjusted padding values */
        }

        /* Increase padding for block container */
        .block-container {
            padding: 2rem 1rem; /* Adjusted padding values */
        }
    </style>
""", unsafe_allow_html=True)

# Inject custom CSS to hide the Streamlit state
hide_streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add Bootstrap for styling
st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN" crossorigin="anonymous">', unsafe_allow_html=True)

# Create a sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Models", "Registry", "Dashboards"])

# Page content based on selection
if page == "Home":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Edge Runner</h1>
            <h3>Welcome to Edge Runner!</h3>
            <h4>Start Training Models and Gain Valuable Insights</h4>
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "Models":

    st.title("Add Model")

    with st.container(border=True):  # Add a border around each card    
        model_name = st.text_input("Enter Model Name")
        model_description = st.text_input("Enter Model Description")
        model_folder_name = model_name.replace(" ", "_").lower()
        model_task = st.selectbox(label="Select Model task", options=("supervised classification", "unsupervised classification", "forecasting", "anomaly detection"))

        if model_task == "supervised classification":
            model_task = "classification"

        model_item = {
            "name": model_name,
            "description": model_description,
            "model_folder": model_folder_name,
            "task": model_task
        }

        if st.button("Add Model", key=f"model-add"):
            try:
                with st.spinner(f"Adding model..."):
                    models["models"].insert(0, model_item)
                    with open(f"{st.session_state.user_id}/models.json", "w") as f:
                        json.dump(models, f, indent=4)

                    try:
                        # send updated model.json to server
                        res_json_upload = requests.post(f"{edge_server_url}/upload_models_json/{st.session_state.user_id}", json=models)
                        st.success("Successfully added the model")
                    except Exception as e:
                        st.error(f"Error: {e}")

            except Exception as e:
                print(e)

    st.title("Models Available")

    # Create a grid layout for cards
    cols = st.columns(3)  # 3 cards per row

    for i, model in enumerate(models["models"]):
        model_path = f"models/{st.session_state.user_id}/{model['model_folder']}/model.h5"
        scaler_path = f"models/{st.session_state.user_id}/{model['model_folder']}/scaler.pkl"
        encoder_path = f"models/{st.session_state.user_id}/{model['model_folder']}/encoder.pkl"

        model_available_locally = os.path.exists(model_path)
        model_available_res = requests.post(f"{edge_server_url}/check_model_availability/", json={
            "model_path": model_path
        })

        if model_available_res.status_code == 200:
            model_available_remote = True
        else:
            model_available_remote = False

        with cols[i % 3]:  # Distribute cards across columns
            with st.container(border=True):  # Add a border around each card
                
                st.markdown(f"### {model['name']}")  # Model title
                st.markdown(f"{model['description']}")  # Model description

                btn_cols = st.columns(3)

                with btn_cols[0]:
                    if st.button("Train Model", key=f"model-train-{i}"):
                        if model["task"] == "unsupervised classification" or  model["task"] == "anomaly detection":
                            create_unsupervised_dialog(model, model_path)()  
                        elif model["task"] == "forecasting":
                            create_forecasting_dialog(model, model_path)()
                        elif model["task"] == "classification":
                            create_supervised_dialog(model, model_path)()  
                        

                with btn_cols[1]:
                    if model_available_locally:
                        if st.button("Use Model", key=f"model-use-{i}"):

                            if model["task"] == "unsupervised classification":
                                create_unsupervised_model_pred_dialog(model, model_path)()
                            elif model["task"] == "anomaly detection":
                                create_anomaly_model_pred_dialog(model, model_path)()
                            elif model["task"] == "classification":
                                create_supervised_model_pred_dialog(model, model_path)()
                            elif model["task"] == "forecasting":
                                create_forecasting_model_pred_dialog(model, model_path)()
                    elif model_available_remote:
                        if st.button("Download Model", key=f"model-use-{i}"):
                            with st.spinner(f"downloading model..."):
                                model_file_res = requests.post(f"{edge_server_url}/deliver_model/", json={
                                    "model_path": model_path
                                })
                                scaler_file_res = requests.post(f"{edge_server_url}/deliver_scaler/", json={
                                    "scaler_path": scaler_path
                                })

                                encoder_file_res = requests.post(f"{edge_server_url}/deliver_encoder/", json={
                                    "encoder_path": encoder_path
                                })

                                if model_file_res.status_code != 200:
                                    st.error(f"Error: {model_file_res}")
                                elif scaler_file_res.status_code != 200:
                                    st.error(f"Error: {scaler_file_res}")
                                elif encoder_file_res.status_code != 200:
                                    st.error(f"Error: {encoder_file_res}")
                                else:
                                    with open(model_path, "wb") as f:
                                        f.write(model_file_res.content)
                                    with open(scaler_path, "wb") as f:
                                        f.write(scaler_file_res.content)
                                    with open(encoder_path, "wb") as f:
                                        f.write(encoder_file_res.content)
                                
                                st.success("model downloaded successfully")
                    else:
                        st.button("Use Model", key=f"model-{i}", disabled=True)

            
                with btn_cols[2]:  
                    if st.button("Delete Model", key=f"model-delete-{i}"):
                        with st.spinner(f"Adding model..."):
                            # remove local model folder
                            if os.path.exists(f"models/{st.session_state.user_id}/{model['model_folder']}"):
                                shutil.rmtree(f"models/{st.session_state.user_id}/{model['model_folder']}")
                            # remove model record and save model json
                            models["models"].remove(model)
                            with open(f"{st.session_state.user_id}/models.json", "w") as f:
                                json.dump(models, f, indent=4)

                            try:
                                # send updated model.json to server
                                res_json_upload = requests.post(f"{edge_server_url}/upload_models_json/{st.session_state.user_id}", json=models)
                                # delete remote model and scaler
                                res_del_model = requests.delete(f"{edge_server_url}/delete_model/", json={
                                    "model_path": f"models/{st.session_state.user_id}/{model['model_folder']}"
                                })
                                st.success("Successfully deleted the model")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    

# added new page called registry
elif page == "Registry":
    st.title("Model Registry")
    
    try:
        # getting all registered models
        registered_models = client.search_registered_models()
        
        if not registered_models:
            st.info("No models found in the registry. Train some models and register them first.")
        else:
            # creating grid layout for cards
            cols = st.columns(3)  # 3 cards per row
            
            for i, model in enumerate(registered_models):
                print(model)
                model_name = model.name
                
                # get latest version
                versions = client.get_latest_versions(model_name)
                if not versions:
                    continue
                    
                latest_version = versions[0]
                
                # extract model details
                run_id = latest_version.run_id
                version_num = latest_version.version
                
                # get run details
                run = client.get_run(run_id)
                task_type = run.data.tags.get("task_type", "Unknown")
                description = run.data.tags.get("description", "No description available")
                
                # path where the model would be saved
                save_dir = f"models/registry"
                model_path = f"{save_dir}/{model_name}"
                
                # check if model is downloaded AND exists in registry_models.json
                model_is_downloaded = os.path.exists(model_path)
                
                # create card
                with cols[i % 3]:
                    with st.container(border=True):
                        
                        st.markdown(f"### {model_name}")
                        st.markdown(f"**Version:** {version_num}")
                        st.markdown(f"**Task:** {task_type}")
                        st.markdown(f"**Description:** {description}")
                        
                        # create buttons side by side using columns
                        btn_cols = st.columns(2)
                        
                        # download Button (always visible)
                        with btn_cols[0]:
                            # show different button text based on download status
                            button_text = "Downloaded ✓" if model_is_downloaded else "Download Model"
                            download_button = st.button(button_text, key=f"download-{model_name}-{version_num}", 
                                                       disabled=model_is_downloaded)
                            
                            if download_button:
                                # show download dialog
                                with st.spinner(f"Downloading {model_name} version {version_num}..."):
                                    os.makedirs(save_dir, exist_ok=True)
                                    
                                    down_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/{model_name}",dst_path=save_dir)
                                    st.success(f"Model downloaded to {down_path}")
                                    
                                    # force a rerun to update the UI
                                    st.rerun()
                        
                        # use model button (only visible if model is downloaded)
                        with btn_cols[1]:
                            if model_is_downloaded:
                                if st.button("Use Model", key=f"use-reg-{model_name}-{version_num}"):
                                    
                                    create_registry_model_pred_dialog(model_name, model_path, task_type)()

                            else:
                                # show disabled button or text indicating download needed
                                st.button("Use Model", key=f"use-reg-{model_name}-{version_num}", disabled=True)
    
    except Exception as e:
        st.error(f"Error connecting to MLflow server: {e}")
        st.info("Make sure MLflow server is running at " + mlflow_tracking_uri)

        
elif page == "Dashboards":
    # Embed Grafana view in the Streamlit app
    st.markdown(f"""
        <iframe src="{grafana_url}" width="100%" height="1080px"></iframe>
    """, unsafe_allow_html=True)
