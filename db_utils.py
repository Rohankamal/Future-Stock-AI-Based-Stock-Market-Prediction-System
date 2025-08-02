# db_utils.py
# db_utils.py (SQLite version)
import sqlite3
import bcrypt
import streamlit as st

DB_PATH = "users.db"

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to SQLite DB: {e}")
        return None

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except sqlite3.Error as e:
        st.error(f"Error initializing SQLite database: {e}")

def add_user(username, email, password) -> bool:
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                       (username, email, hashed_password.decode('utf-8')))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        st.error("Username or Email already exists.")
        conn.close()
        return False
    except sqlite3.Error as e:
        st.error(f"Error adding user: {e}")
        conn.close()
        return False

def verify_user(email, password) -> bool:
    conn = get_db_connection()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            stored_hash = result[0].encode('utf-8')
            return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
        return False
    except sqlite3.Error as e:
        st.error(f"Error verifying user: {e}")
        conn.close()
        return False

