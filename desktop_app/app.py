import streamlit as st
import pandas as pd
import os
from inference import infer
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np

edge_server_url = "http://127.0.0.1:8001"

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
page = st.sidebar.selectbox("Select Page", ["Home", "Models", "Train", "Dashboards", "About"])

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

    def create_supervised_model_pred_dialog(model):
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
                            results = inf(df, label, model["path"], model["task"], edge_server_url)
                            st.write("Prediction Results:")
                            st.dataframe(results)
                    except Exception as e:
                        st.error(f"Error: {e}")

        return supervised_model_pred_dialog
    

    def create_unsupervised_model_pred_dialog(model):
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
                            results = inf(df, label, model["path"], model["task"], edge_server_url)
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
    

    def create_anomaly_model_pred_dialog(model):
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
                            results = inf(df, label, model["path"], model["task"], edge_server_url)
                            st.dataframe(results)
                    except Exception as e:
                        st.error(f"Error: {e}")

        return anomaly_model_pred_dialog


    for i, model in enumerate(models["models"]):
        with cols[i % 3]:  # Distribute cards across columns
            with st.container(border=True):  # Add a border around each card
                st.image(model["image"], use_container_width=True)  # Display model image
                st.markdown(f"### {model['name']}")  # Model title
                st.markdown("Upload data to run predictions.")  # Model description
                
                # Clickable button inside the card
                if st.button("Use Model", key=f"model-{i}"):

                    if model["task"] == "unsupervised classification":
                        create_unsupervised_model_pred_dialog(model)()
                    elif model["task"] == "anomaly detection":
                        create_anomaly_model_pred_dialog(model)()
                    elif model["task"] == "classification":
                        create_supervised_model_pred_dialog(model)()


elif page == "Train":
    st.title("Train")

    # Create a grid layout for cards
    cols = st.columns(3)  # 3 cards per row

    def create_supervised_dialog(model):
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
                print("save_path:", model["path"])
                if file is not None:
                    df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
                    df = df.head(10000)

                    with st.spinner("Training in progress..."):
                        train_fn(df, model["path"], model["task"], label, rounds, edge_server_url)

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
                print("save_path:", model["path"])
                if file is not None:
                    df = pd.read_excel(file) if file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
                    df = df.head(10000)

                    with st.spinner("Training in progress..."):
                        train_fn(df, model["path"], model["task"], label, rounds, edge_server_url)

                    st.write("Training Status: success")
        return unsupervised_model_train_dialog

    # Iterate through models and generate UI
    for i, model in enumerate(models["models"]):
        with cols[i % 3]:  # Distribute cards across columns
            with st.container(border=True):  # Add a border around each card
                st.image(model["image"], use_container_width=True)  # Display model image
                st.markdown(f"### {model['name']}")  # Model title
                st.markdown("Upload data to train model.")  # Model description
                
                # Clickable button inside the card
                if st.button("Use Model", key=f"model-{i}"):

                    if model["task"] == "unsupervised classification" or  model["task"] == "anomaly detection":
                        create_unsupervised_dialog(model)()  # Call the function factory and execute it
                    else:
                        create_supervised_dialog(model)()  # Call the function factory and execute it



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
