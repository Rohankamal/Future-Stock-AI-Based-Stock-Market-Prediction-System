# db_utils.py
import mysql.connector # Use the MySQL connector library
import bcrypt # For password hashing
import streamlit as st
import os # For environment variables, which is a better practice

# --- MySQL Database Configuration ---
# IMPORTANT: It is best practice to use environment variables for credentials
# For demonstration purposes, we will use hardcoded values as requested.
# For a production application, please store these securely.
DATABASE_HOST = "localhost" # Assuming MySQL is running on localhost
DATABASE_USER = "root"
DATABASE_PASSWORD = "Test@0987"
DATABASE_NAME = "futurestockai"

def get_db_connection():
    """
    Establishes a connection to the MySQL database.
    Returns the connection object.
    """
    try:
        conn = mysql.connector.connect(
            host=DATABASE_HOST,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD,
            database=DATABASE_NAME
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

def init_db():
    """
    Initializes the MySQL database and creates the users table with the correct schema.
    """
    try:
        conn = mysql.connector.connect(
            host=DATABASE_HOST,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD
        )
        cursor = conn.cursor()
        
        # Create the database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}")
        conn.database = DATABASE_NAME
        
        # FIX: Removed the DROP TABLE command to prevent data from being overwritten
        # cursor.execute("DROP TABLE IF EXISTS users") 
        
        # Create the users table with the 'password' column for bcrypt hash
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) NOT NULL UNIQUE,
                email VARCHAR(255) NOT NULL UNIQUE,
                password VARCHAR(255) NOT NULL
            )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as e:
        st.error(f"Error initializing MySQL database: {e}")

def add_user(username, email, password) -> bool:
    """
    Adds a new user to the database after hashing the password.
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # The INSERT query is already correct
        query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, email, hashed_password.decode('utf-8')))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as e:
        if e.errno == 1062: # 1062 is the error code for duplicate entry
            st.error("Username or Email already exists. Please choose a different one.")
        else:
            st.error(f"Error adding user: {e}")
        conn.close()
        return False

def verify_user(email, password) -> bool:
    """
    Verifies user credentials against the database.
    """
    conn = get_db_connection()
    if not conn: return False
    
    try:
        cursor = conn.cursor()
        # The SELECT query is already correct
        query = "SELECT password FROM users WHERE email = %s"
        cursor.execute(query, (email,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()

        if result:
            stored_password_hash = result[0].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
                return True
        return False
    except mysql.connector.Error as e:
        st.error(f"Error verifying user: {e}")
        conn.close()
        return False
