import os
import random
import streamlit as st
from datetime import datetime
from Sentiment_analysis import sentiment_analysis, store_sentiment_to_mysql
from Crisis_detection import analyze_text
from daily_affirmations import daily_affirmations
import google.generativeai as genai
from dotenv import load_dotenv
import mysql.connector
import hashlib
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


st.set_page_config(page_title="Serenity.AI", layout="wide")

import base64
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(r"C:\Users\Dhanashri Masram\OneDrive\Desktop\Mental_Health Chatbot\wmremove-transformed.jpeg")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Dshr!2022",
        database="mh_chatbot"
    )

def generate_mood_reflection(user_id):
    conn = get_connection()
    df = pd.read_sql(f"SELECT sentiment, timestamp FROM sentiments WHERE user_id = {user_id} ORDER BY timestamp DESC LIMIT 7", conn)
    conn.close()

    mood_log = "\n".join([
        f"- {row['timestamp'].strftime('%Y-%m-%d')}: {row['sentiment'].title()}"
        for _, row in df.iterrows()
    ])

    prompt = f"""
You are a kind mental health assistant. Based on the user's recent mood entries, 
give a gentle and supportive reflection. Encourage them if they're doing well, 
or offer compassion and tips to feel better if they're going through a tough time.

Mood Log:
{mood_log}
"""
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

def generate_mood_graph(user_id):
    conn = get_connection()
    df = pd.read_sql(f"SELECT sentiment, timestamp FROM sentiments WHERE user_id = {user_id}", conn)
    conn.close()

    if df.empty:
        return None

    df['sentiment'] = df['sentiment'].str.strip().str.title().replace({'Nuetral': 'Neutral'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['month_year'] = df['timestamp'].dt.to_period('M').astype(str)

    sentiment_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    df['sentiment'] = df['sentiment'].astype(CategoricalDtype(categories=sentiment_order, ordered=True))

    sns.set(style="whitegrid")
    g = sns.catplot(
        data=df,
        x='sentiment',
        kind='count',
        col='month_year',
        col_wrap=3,
        order=sentiment_order,
        palette='pastel',
        height=4,
        aspect=1
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Mood Trends by Month")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return buffer

# Session state
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "chat"

# Login / Signup
if st.session_state["user_id"] is None:
    st.title("🔐 Serenity.AI Login")
    auth_mode = st.radio("Choose mode:", ["Login", "Sign Up"], horizontal=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Continue"):
        conn = get_connection()
        cursor = conn.cursor()
        if auth_mode == "Sign Up":
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                st.error("Username already exists.")
            else:
                hashed = hash_password(password)
                cursor.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed))
                conn.commit()
                cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
                user_id = cursor.fetchone()[0]
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username
                st.rerun()
        else:
            hashed = hash_password(password)
            cursor.execute("SELECT id FROM users WHERE username = %s AND password_hash = %s", (username, hashed))
            result = cursor.fetchone()
            if result:
                st.session_state["user_id"] = result[0]
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid credentials.")
        cursor.close()
        conn.close()
    st.stop()

# Sidebar 
with st.sidebar:
    st.markdown(f"### Hello, **{st.session_state['username']}**")
    if st.session_state["view_mode"] == "chat":
        if st.button("📊 View Moodboard"):
            st.session_state["view_mode"] = "moodboard"
    elif st.session_state["view_mode"] == "moodboard":
        if st.button("💬 Back to Chat"):
            st.session_state["view_mode"] = "chat"
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# Moodboard Page
if st.session_state["view_mode"] == "moodboard":
    st.title("📊 Your Moodboard")
    st.subheader("🪞 Reflection")
    reflection = generate_mood_reflection(st.session_state["user_id"])
    st.info(reflection)

    st.subheader("📈 Mood Trend Graph")
    graph = generate_mood_graph(st.session_state["user_id"])
    if graph:
        st.image(graph, use_container_width=True)
    else:
        st.warning("Not enough data to show mood trends.")
    st.stop()

# Main Chat Page
st.title("🌿 Serenity.AI")
st.caption("Your supportive mental health companion")

st.info(f"🌞 Daily Affirmation: {random.choice(daily_affirmations)}")

# Chat input
user_input = st.chat_input("How are you feeling today?")
if user_input:
    st.session_state["chat_history"].append(("user", user_input))
    sentiment = sentiment_analysis(user_input)[0]
    store_sentiment_to_mysql(sentiment, st.session_state["user_id"])

    try:
        crisis = analyze_text(user_input)["result"]["crisis_detected"]
        if crisis:
            reply = ("⚠️ It sounds like you're struggling. Please talk to someone you trust. You’re not alone 💗"    "⚠️ It sounds like you're struggling. Please talk to someone you trust. You’re not alone 💗\n\n"
    "🆘 **Crisis Helplines:**"
    "• 📞 iCall (India): +91 9152987821"
    "• 📞 AASRA: +91 9820466726"
    "• 📞 Vandrevala: 1860 266 2345 / 9999 666 555"
    "• 📞 Mental Health Helpline (24x7): 1800 599 0019\n"
    "Please take care of yourself and seek help if needed. 🤍"
)
            
        else:
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            chat = model.start_chat()
            chat.send_message("You are Serenity.AI, a gentle, emotionally supportive chatbot. You never give medical advice.")
            response = chat.send_message(user_input)
            reply = response.text
    except Exception:
        reply = "Sorry, something went wrong."

    st.session_state["chat_history"].append(("bot", reply))

# Display chat
for role, msg in st.session_state["chat_history"]:
    with st.chat_message("🤖" if role == "bot" else "🧍"):
        st.markdown(msg)
st. caption("❤️ You're not alone.")
st.caption("DISCLAIMER: This chatbot is not a substitute for professional mental health support. If you are in crisis, please reach out to a qualified mental health professional or helpline.")
