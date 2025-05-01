import streamlit as st
import sqlite3
import hashlib

# DB Setup
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
conn.commit()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
              (username, hash_password(password)))
    return c.fetchone()

def register_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                  (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

st.title("🔐 Login or Register")

tab1, tab2 = st.tabs(["Login", "Register"])

with tab1:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.switch_page("app.py")  # ✅ will redirect to Home
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
