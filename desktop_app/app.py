import streamlit as st
import pandas as pd
import os
import tensorflow as tf
from inference import infer
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import shutil
from mlflow.models import Model
import base64
from PIL import Image, ImageDraw, ImageFont
import torch

# Set page configuration
st.set_page_config(
    page_title="Edge Runner",
    initial_sidebar_state="expanded",
    page_icon="bizsuite_logo_no_background.ico",
    layout="wide",  # Use wide layout
)

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("You are not logged in.")
    st.stop()

st.title("Welcome to the Edge Runner Home Page!")
st.success(f"Welcome {st.session_state['username']}!")

if st.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.switch_page("Login") 

edge_server_url = "http://127.0.0.1:8001"

# MLflow server address
mlflow_tracking_uri = "http://127.0.0.1:5001"

# initializing MLflow client
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient()


# Read JSON file into a dictionary
with open("models.json", "r") as file:
    models = json.load(file)


def inf(df, label, model_path, task_type, edge_server_url): 
    churn_list = infer(df, label, task_type, model_path, edge_server_url)
    return churn_list


def train_fn(df, save_path, task_type, label, rounds, edge_server_url): 
    # Save DataFrame to CSV
    os.makedirs("tmp", exist_ok=True)
    tmp_path = "tmp/data.xlsx"
    df.to_excel(tmp_path, index=False)
    
    command = [
        "python",
        "flwr_client.py",  # Replace with your actual script path
        "--file_path", str(tmp_path),
        "--save_path", str(save_path),
        "--task_type", str(task_type),
        "--label", str(label),
        "--rounds", str(rounds),
        "--edge_server_url", str(edge_server_url)
    ]
    
    # Start the external Python script as a subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    os.remove(tmp_path)
    
    print(stderr)

    return "success"

def create_supervised_model_pred_dialog(model, model_path):
    @st.dialog(f"Upload Data for {model['name']}")
    def supervised_model_pred_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        # Create a text input field
        label = st.text_input("Prediction Column Name:")
        if st.button("Submit"):
            if file is not None:
                # Check the file type and read it appropriately
                if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                    df = pd.read_excel(file)  # Read Excel file
                else:
                    df = pd.read_csv(file)  # Read CSV file
                df = df.head(10000)
                try:
                    # Show status updates
                    with st.spinner("Inference in progress..."):
                        results = inf(df, label, model_path, model["task"], edge_server_url)
                        st.write("Prediction Results:")
                        st.dataframe(results)
                except Exception as e:
                    st.error(f"Error: {e}")
    return supervised_model_pred_dialog

def create_unsupervised_model_pred_dialog(model, model_path):
    @st.dialog(f"Upload Data for {model['name']}")
    def unsupervised_model_pred_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        label = ""
        if st.button("Submit"):
            if file is not None:
                # Check the file type and read it appropriately
                if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                    df = pd.read_excel(file)  # Read Excel file
                else:
                    df = pd.read_csv(file)  # Read CSV file
                df = df.head(10000)
                try:
                    # Show status updates
                    with st.spinner("Inference in progress..."):
                        results = inf(df, label, model_path, model["task"], edge_server_url)
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

def create_anomaly_model_pred_dialog(model, model_path):
    @st.dialog(f"Upload Data for {model['name']}")
    def anomaly_model_pred_dialog():
        # File uploader for Home page
        file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
        label = "Anomaly"
        if st.button("Submit"):
            if file is not None:
                # Check the file type and read it appropriately
                if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
                    df = pd.read_excel(file)  # Read Excel file
                else:
                    df = pd.read_csv(file)  # Read CSV file
                df = df.head(10000)
                try:
                    # Show status updates
                    with st.spinner("Inference in progress..."):
                        results = inf(df, label, model_path, model["task"], edge_server_url)
                        st.dataframe(results)
                except Exception as e:
                    st.error(f"Error: {e}")
    return anomaly_model_pred_dialog


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


# Set page configuration
st.set_page_config(
    page_title="Edge Runner",
    initial_sidebar_state="expanded",
    page_icon="bizsuite_logo_no_background.ico",
    layout="wide",  # Use wide layout
)


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
page = st.sidebar.selectbox("Select Page", ["Home", "Models", "Registry", "Dashboards", "About"])

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
    st.title("Models Available")
    
    # Create a grid layout for cards
    cols = st.columns(3)  # 3 cards per row

    def create_supervised_dialog(model, model_path):
        """Factory function to create a supervised training dialog for a specific model."""
        @st.dialog(f"Upload Data for {model['name']}")
        def supervised_model_train_dialog():
            file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
            label = st.text_input("Label Name:")
            rounds = str(int(st.number_input("Num of Rounds to train:", min_value=1, step=1)))

            if st.button("Submit"):
                print("label:", label)
                print("rounds:", rounds)
                print("task:", model["task"])
                print("save_path:", model_path)
                if file is not None:
                    df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
                    df = df.head(10000)

                    with st.spinner("Training in progress..."):
                        train_fn(df, model_path, model["task"], label, rounds, edge_server_url)

                    st.write("Training Status: success")
        return supervised_model_train_dialog

    def create_unsupervised_dialog(model):
        """Factory function to create an unsupervised training dialog for a specific model."""
        @st.dialog(f"Upload Data for {model['name']}")
        def unsupervised_model_train_dialog():
            file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
            label = ""
            rounds = str(int(st.number_input("Num of Rounds to train:", min_value=1, step=1)))

            if st.button("Submit"):
                print("label:", label)
                print("rounds:", rounds)
                print("task:", model["task"])
                print("save_path:", model_path)
                if file is not None:
                    df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
                    df = df.head(10000)

                    with st.spinner("Training in progress..."):
                        train_fn(df, model_path, model["task"], label, rounds, edge_server_url)

                    st.write("Training Status: success")
        return unsupervised_model_train_dialog
    

    for i, model in enumerate(models["models"]):
        model_path = f"models/user_id/{model['model_folder']}/model.h5"
        model_is_trained = os.path.exists(model_path)

        with cols[i % 3]:  # Distribute cards across columns
            with st.container(border=True):  # Add a border around each card
                st.image(model["image"], use_container_width=True)  # Display model image
                st.markdown(f"### {model['name']}")  # Model title
                st.markdown(f"{model['description']}")  # Model description

                btn_cols = st.columns(2)

                with btn_cols[0]:
                    if st.button("Train Model", key=f"model-train-{i}"):
                        if model["task"] == "unsupervised classification" or  model["task"] == "anomaly detection":
                            create_unsupervised_dialog(model, model_path)()  # Call the function factory and execute it
                        else:
                            create_supervised_dialog(model, model_path)()  # Call the function factory and execute it

                # Clickable button inside the card
                with btn_cols[1]:
                    if model_is_trained:
                        if st.button("Use Model", key=f"model-use-{i}"):

                            if model["task"] == "unsupervised classification":
                                create_unsupervised_model_pred_dialog(model, model_path)()
                            elif model["task"] == "anomaly detection":
                                create_anomaly_model_pred_dialog(model, model_path)()
                            elif model["task"] == "classification":
                                create_supervised_model_pred_dialog(model, model_path)()
                    else:
                        st.button("Use Model", key=f"model-{i}", disabled=True)


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
                        # placeholder image based on task type
                        if "classification" in task_type.lower():
                            img_path = "img.jpg"
                        elif "regression" in task_type.lower():
                            img_path = "img.jpg"
                        elif "anomaly" in task_type.lower():
                            img_path = "img.jpg"
                        else:
                            img_path = "img.jpg"
                            
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        
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
    # URL of the Grafana dashboard or panel (make sure it's publicly accessible or authenticated)
    grafana_url = "http://localhost:3003"

    # Embed Grafana view in the Streamlit app
    st.markdown(f"""
        <iframe src="{grafana_url}" width="100%" height="1080px"></iframe>
    """, unsafe_allow_html=True)


elif page == "About":
    st.title("About")
    st.write("This is the About page for Edge Runner.")