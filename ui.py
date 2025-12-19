import streamlit as st
import json
import pickle
import random
import torch
import streamlit.components.v1 as components
from train import (
    ChatbotModel,
    classify_intent,
    fetch_response,
    execute_training_loop
    
)

TTS_ENABLED = True

def speak_text(text):
    escaped_text = text.replace('"', '\\"')
    components.html(f""" <script>
    var msg = new SpeechSynthesisUtterance("{escaped_text}");
    window.speechSynthesis.speak(msg); </script>
    """, height=0)

st.set_page_config(page_title="MindHaven Therapy Clinic Chatbot", page_icon="👩‍⚕️", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #1E2B64;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #1E2B64;
    }
    [data-testid="stSidebar"] {
        background-color: #2980B9;
    }
    .stChatMessage {
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    form button[type="submit"] {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #28a745;
    }
    input[type="text"] {
    background-color: #2C3E50 !important;
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px;
}
    </style>
""", unsafe_allow_html=True)

if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'greeted' not in st.session_state:
    st.session_state.greeted = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("👩‍💻 Train the Chatbot")
with st.sidebar.form("train_form", clear_on_submit=False):
    st.markdown("📇**Enter Your Name**", unsafe_allow_html=True)
    name = st.text_input(" ", value=st.session_state.username or "", placeholder="Type your name here", label_visibility="collapsed")

    st.markdown("🔄**Epochs (between 100–30000, max 5 digits):**")
    epoch_input = st.text_input(" ", value="1000", max_chars=5, label_visibility="collapsed")

    try:
        epochs = int(epoch_input)
    except ValueError:
        epochs = 0

    submitted = st.form_submit_button("🚀 Train")

if submitted:
    invalid_input = False

    if not name.strip():
        st.sidebar.error("❌ Please enter your name.")
        invalid_input = True

    if epochs < 100 or epochs > 30000:
        st.sidebar.error("❌ Epochs must be between 100 and 30000.(Please enter numeric value)")
        invalid_input = True

    if len(str(int(epochs))) > 5:
        st.sidebar.error("❌ Epochs must be a number with at most 5 digits.")
        invalid_input = True

    if not invalid_input:
        st.session_state.username = name.strip()
        with st.sidebar:
            with st.spinner("Training the therapy chatbot..."):
                progress_bar = st.progress(0)
                execute_training_loop(epochs, progress_bar)

        st.session_state.trained = True
        st.session_state.greeted = False
        st.session_state.chat_history = []
        st.session_state.last_epochs = epochs
        st.sidebar.success(f"✅ Training complete, {st.session_state.username}! ({epochs} epochs used)")
    else:
        st.session_state.trained = False

st.title("💬 MindHaven Therapy Clinic Virtual Assistant")
st.markdown("Let our MindHaven chatbot help your emotional well-being. You can also ask anything about us 🤗")

if st.session_state.trained:
    st.subheader(f"Welcome to MindHaven Chatting space ,{st.session_state.username}!!")

    intents = json.load(open('intents.json'))
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = ChatbotModel(len(words), 512, len(classes))
    model.load_state_dict(torch.load("best_chatbot.pth"))
    model.eval()

    if not st.session_state.greeted:
        greeting = f"Hi {st.session_state.username}, welcome to MindHaven Therapy. I'm your digital therapy assistant, here to provide support, comfort, and guidance wherever you are. While I’m not a licensed therapist, I can share tips, suggest self-care techniques, and help you learn more about our services. Let’s take a small step toward healing together."
        st.session_state.chat_history.append(("assistant", greeting))
        if TTS_ENABLED:
            speak_text(greeting)
        st.session_state.greeted = True

    bot_avatar = "🤖"
    user_avatar = "🧑‍⚕️"

    for role, message in st.session_state.chat_history:
        with st.chat_message(role, avatar=(bot_avatar if role == "assistant" else user_avatar)):
            st.markdown(f'<span style="color:white; font-weight:bold;">{message}</span>', unsafe_allow_html=True)

    user_input = st.chat_input("Type something you'd like to talk about...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        intent_tag = classify_intent(user_input, model, words, classes)
        response = fetch_response(intent_tag, intents)
        st.session_state.chat_history.append(("assistant", response))

        with st.chat_message("user", avatar=user_avatar):
            st.markdown(f'<span style="color:white; font-weight:bold;">{user_input}</span>', unsafe_allow_html=True)
        with st.chat_message("assistant", avatar=bot_avatar):
            st.markdown(f'<span style="color:white; font-weight:bold;">{response}</span>', unsafe_allow_html=True)

        if TTS_ENABLED:
            speak_text(response)
else:
    st.info("👉 Please enter your name and a valid number of epochs (100–30000) in the sidebar to train your therapy chatbot.")
