import streamlit as st
import hashlib
import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

# --- Set up MySQL connection using Streamlit's built-in connection ---
# config is in .streamlit/secrets.toml
conn = st.connection("mysql", type="sql") 

# Create users table if not exists
with conn.engine.begin() as connection:
    connection.execute(text('''
        CREATE TABLE IF NOT EXISTS users (
            username VARCHAR(255) PRIMARY KEY,
            password VARCHAR(255),
            user_id VARCHAR(255)
        )
    '''))

# --- Password hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Check user credentials ---
def check_credentials(username, password):
    query = """
        SELECT user_id FROM users WHERE username = :username AND password = :password
    """
    params = {"username": username, "password": hash_password(password)}
    result = conn.query(query, params=params)
    return result["user_id"][0] if not result.empty else None

# --- Register user ---
def register_user(username, password):
    try:
        user_id = hash_password(username)
        with conn.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO users (username, password, user_id) VALUES (:u, :p, :id)"),
                {"u": username, "p": hash_password(password), "id": user_id}
            )
        return True
    except IntegrityError:
        return False

# --- Session State Setup ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- UI ---
st.title("🔐 Login or Register")

tab1, tab2 = st.tabs(["Login", "Register"])

with tab1:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_id = check_credentials(username, password)
        if user_id:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_id = user_id
            st.success("Login successful!")
            st.switch_page("app.py")  
        else:
            st.error("Invalid username or password")

with tab2:
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if new_password != confirm:
            st.error("Passwords do not match")
        elif register_user(new_username, new_password):
            st.success("Registered successfully. Please login.")
        else:
            st.error("Username already exists")
