import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

LangChain ভার্সন সামঞ্জস্যপূর্ণ করার ফিক্স

try:
from langchain.output_parsers import PydanticOutputParser
except ImportError:
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
import random
import time
import json
import io
import base64 # অডিও প্লেব্যাকের জন্য
import traceback # গ্লোবাল এক্সেপশন UI-এর জন্য

Lottie, Mic Recorder, OpenAI (Whisper)

from streamlit_lottie import st_lottie
from streamlit_mic_recorder import mic_recorder
import openai

অটো-রিফ্রেশ

from streamlit_autorefresh import st_autorefresh

QR Code

import qrcode
from PIL import Image

ElevenLabs

try:
from elevenlabs import ElevenLabs
except ImportError:
st.error("❌ ElevenLabs library missing. Please add elevenlabs in requirements.txt")
st.stop()

---------------- CONFIG ----------------

ফিক্স: Deprecated অপশন অপসারণ করা হয়েছে

st.set_option('client.showErrorDetails', False)

st.set_page_config(
page_title="Arc Guardian AI Agent | Team Believer",
page_icon="assets/favicon.png", # অ্যাসেট পাথ (আপনার ফোল্ডারে আছে)
layout="wide"
)

------------------------------------------------------------

🔐 SECRETS & API KEYS

------------------------------------------------------------

OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")
ARC_API_KEY = st.secrets.get("arc", {}).get("api_key")
ELEVENLABS_API_KEY = st.secrets.get("elevenlabs", {}).get("api_key")

ফিক্স: Arc API URL (উদাহরণ; আপনার আসল URL দিন)

ARC_API_URL = "https.api.arc.example.com/v1/transactions"

------------------------------------------------------------

🎨 UI POLISH (CSS INJECTION)

------------------------------------------------------------

st.markdown("""
<style>
/* Gradient buttons /
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[kind="secondary"] {
background: linear-gradient(90deg, #00bcd4, #00e5ff);
color: #000000;
border: none;
font-weight: bold;
transition: all 0.3s ease-in-out;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
box-shadow: 0 0 15px 5px #00bcd4;
transform: scale(1.02);
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
opacity: 0.8;
}
/ Glowing sidebar */
[data-testid="stSidebar"] {
border-right: 2px solid #00bcd4;
box-shadow: 0 0 15px 5px #00bcd4;
animation: pulse 2.5s infinite alternate;
}
@keyframes pulse {
from { box-shadow: 0 0 10px 2px #00bcd4; }
to { box-shadow: 0 0 20px 7px #00e5ff; }
}
</style>
""", unsafe_allow_html=True)

------------------------------------------------------------

🤖 MODEL SETUP

------------------------------------------------------------

@st.cache_resource
def get_llm():
"""LLM রিসোর্স ক্যাশ করে (ফলব্যাক সহ)।"""
try:
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
return llm
except Exception as e:
st.warning(f"gpt-4o-mini failed (Error: {e}). Falling back to gpt-3.5-turbo.")
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
return llm

@st.cache_resource
def get_elevenlabs_client():
"""ElevenLabs ক্লায়েন্ট ক্যাশ করে।"""
if not ELEVENLABS_API_KEY:
st.warning("🔑 ElevenLabs API key missing in secrets.toml. Voice will be disabled.")
return None
return ElevenLabs(api_key=ELEVENLABS_API_KEY)

try:
llm = get_llm()
client = openai.OpenAI(api_key=OPENAI_API_KEY)
eleven_client = get_elevenlabs_client() # নতুন ক্লায়েন্ট লোড করা
except Exception as e:
st.error(f"API Key setup error: {e}")
st.stop()

------------------------------------------------------------

🔊 TTS HELPER FUNCTION (SDK v2 ফিক্সড)

------------------------------------------------------------

@st.cache_data
def generate_tts(text: str, voice_name="Adam"):
"""ElevenLabs ব্যবহার করে ভয়েস জেনারেট করে এবং বাইটস রিটার্ন করে।"""
if not eleven_client:
st.warning("🔑 ElevenLabs client not available. Skipping TTS.")
return None
try:
audio_bytes_iterator = eleven_client.text_to_speech.convert(
voice_id=voice_name.lower(),
model_id="eleven_multilingual_v2",
text=text
)
audio_bytes = b"".join([chunk for chunk in audio_bytes_iterator])
return audio_bytes

except Exception as e:  
    st.error(f"TTS Generation failed: {e}")  
    return None

def play_tts_response(text, key="tts_playback", voice_override=None):
"""জেনারেট করা অডিও বাইটকে st.audio দিয়ে প্লে করে।"""
selected_voice = voice_override if voice_override else st.session_state.get("selected_voice", "Adam")

with st.spinner(f"🎧 Generating AI voice ({selected_voice})..."):  
    audio_bytes = generate_tts(text, voice_name=selected_voice)  
      
if audio_bytes:  
    b64 = base64.b64encode(audio_bytes).decode()  
    audio_html = f"""  
        <audio autoplay="true" style="display: none;">  
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">  
        </audio>  
        """  
    st.markdown(audio_html, unsafe_allow_html=True)  
else:  
    st.info("TTS unavailable – check API key or cloud environment.")

============================================================

🧠 ARC GUARDIAN — PART B: AGENTS SETUP

============================================================

--- Pydantic Models ---

class Transaction(BaseModel):
receiver: str = Field(description="Wallet address, must start with 0x")
amount: float = Field(description="Amount of USDC to send")
currency: str = Field(default="USDC")

class AIPlan(BaseModel):
reasoning: str = Field(description="Step-by-step reasoning for parsing the request.")
transactions: List[Transaction] = Field(description="List of parsed transactions.")
action: str = Field(description="Recognized intent: TRANSACT, CHECK_BALANCE, UNKNOWN")

--- Agent 1: Parser Agent ---

try:
parser_parser = PydanticOutputParser(pydantic_object=AIPlan)
parser_prompt = PromptTemplate(
template="""
You are Agent 1 (Parser). Your job is to parse a user's request into a structured JSON plan.
Available actions: TRANSACT, CHECK_BALANCE, UNKNOWN.
Rules:
1️⃣ Identify intent.
2️⃣ If TRANSACT → extract receiver & amount.
3️⃣ If CHECK_BALANCE → set action accordingly.
4️⃣ Reject invalid or unclear inputs.
5️⃣ Only use USDC as currency.
6️⃣ If amount ≤ 0 or > 100 → flag in reasoning.
User Input: {user_input}
{format_instructions}
""",
input_variables=["user_input"],
partial_variables={"format_instructions": parser_parser.get_format_instructions()}
)
chain_parser = parser_prompt | llm | parser_parser
except Exception as e:
st.error(f"Parser Agent setup error: {e}")
st.stop()

--- Agent 2: Audit Agent ---

try:
auditor_prompt = PromptTemplate(
template="""
You are Agent 2 (Auditor). Review the transaction plan for risk or fraud.
Rules:
- >50 USDC = FLAGGED
- Address 0xDEADBEEF or 0x0000000 = FRAUD → REJECTED
- Otherwise APPROVED.
Respond ONLY as JSON:
{{
"audit_result": "APPROVED" | "FLAGGED" | "REJECTED",
"audit_comment": "Short reason (max 15 words)"
}}
The Plan:
{plan_string}
""",
input_variables=["plan_string"]
)
audit_output_parser = StrOutputParser()
chain_auditor = auditor_prompt | llm | audit_output_parser
except Exception as e:
st.error(f"Audit Agent setup error: {e}")
st.stop()

============================================================

⚙️ ARC GUARDIAN — PART C: SESSION STATE

============================================================

if "transactions" not in st.session_state:
st.session_state["transactions"] = []
if "ai_plan" not in st.session_state:
st.session_state["ai_plan"] = None
if "audit_result" not in st.session_state:
st.session_state["audit_result"] = None
if "reasoning_log" not in st.session_state:
st.session_state["reasoning_log"] = []
if "correct_pin" not in st.session_state:
st.session_state["correct_pin"] = str(random.randint(1000, 9999))
if "simulation_mode" not in st.session_state:
st.session_state["simulation_mode"] = True
if "user_prompt" not in st.session_state:
st.session_state["user_prompt"] = ""
if "mock_balance" not in st.session_state:
st.session_state["mock_balance"] = 120.0
if "enable_audit" not in st.session_state:
st.session_state["enable_audit"] = True
if "selected_voice" not in st.session_state:
st.session_state["selected_voice"] = "Adam"
if "processing" not in st.session_state:
st.session_state["processing"] = False

============================================================

⚙️ ARC GUARDIAN — PART D: HELPER FUNCTIONS

============================================================

def safe_execute(func, *args, **kwargs):
"""Global error handler wrapper."""
try:
return func(*args, **kwargs)
except Exception as e:
st.error(f"⚠️ Unexpected Runtime Error: {e}")
st.code(traceback.format_exc()) # Shows traceback

@st.cache_data(show_spinner=False)
def transcribe_audio(audio_bytes):
"""Transcribes audio to text using OpenAI Whisper."""
try:
audio_file = io.BytesIO(audio_bytes)
audio_file.name = "recording.wav"
transcript_response = client.audio.transcriptions.create(
model="whisper-1",
file=audio_file
)
return transcript_response.text
except Exception as e:
st.error(f"Voice transcription failed: {e}")
return ""

def load_lottiefile(filepath: str):
"""Loads Lottie file (warns if not found)."""
try:
with open(filepath, "r") as f:
return json.load(f)
except FileNotFoundError:
st.warning(f"Lottie file not found at: {filepath}")
return None

def get_asset_as_base64(path):
"""লোকাল অ্যাসেট ফাইলকে Base64 Data URI-তে কনভার্ট করে।"""
try:
with open(path, "rb") as f:
data = f.read()
if path.endswith(".mp4"):
mime_type = "video/mp4"
elif path.endswith(".png"):
mime_type = "image/png"
else:
mime_type = "application/octet-stream"
b64 = base64.b64encode(data).decode()
return f"data:{mime_type};base64,{b64}"
except FileNotFoundError:
st.warning(f"Asset file not found: {path}")
return None

def check_balance():
"""Simulates a dynamic mock balance."""
return f"Current wallet balance: {st.session_state['mock_balance']:.2f} USDC (dynamic simulation)"

def log_transaction(receiver, amount, status, detail="N/A"):
"""Saves transaction log to session state."""
st.session_state["transactions"].append({
"receiver": receiver,
"amount": amount,
"status": status,
"detail": detail,
"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})

def log_reasoning(agent, reasoning):
"""Saves AI agent reasoning log to session state."""
st.session_state["reasoning_log"].append({
"agent": agent,
"reasoning": reasoning,
"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
})

@st.cache_data
def analyze_command_cached(user_input):
"""Calls Agent 1 (Parser)."""
try:
return chain_parser.invoke({"user_input": user_input})
except Exception as e:
st.error(f"AI Parsing Error: {e}")
return None

@st.cache_data
def analyze_audit_cached(plan_string):
"""Calls Agent 2 (Auditor)."""
try:
response_string = chain_auditor.invoke({"plan_string": plan_string})
return response_string
except Exception as e:
st.error(f"AI Audit Error: {e}")
return None

--- Asset Loading ---

(আপনার ফোল্ডারে আছে)

success_anim = load_lottiefile("assets/success.json")
APP_URL = "https.arc-guardian.streamlit.app"

def execute_transactions(transactions: List[Transaction]):
"""Simulates or executes the transaction via Arc API."""
is_simulation = st.session_state["simulation_mode"]
headers = {"Authorization": f"Bearer {ARC_API_KEY}"}

for txn in transactions:  
    if not txn.receiver.startswith("0x") or txn.amount <= 0 or txn.amount > 100:  
        st.warning(f"⚠️ Invalid transaction skipped: {txn.amount} to {txn.receiver} (Amount must be > 0 and <= 100)")  
        log_transaction(txn.receiver, txn.amount, "failed", "Invalid parameters")  
        continue  

    payload = {"amount": txn.amount, "currency": "USDC", "receiver": txn.receiver}  
      
    with st.spinner(f"Processing {txn.amount} USDC → {txn.receiver}..."):  
        if is_simulation:  
            time.sleep(1.5)  
            st.success(f"✅ [SIMULATED] Sent {txn.amount} USDC to {txn.receiver}")  
            log_transaction(txn.receiver, txn.amount, "success", "SIMULATED_TXN_ID")  
            st.toast(f"Sent {txn.amount} USDC successfully! ✅")  
              
            if st.session_state["selected_voice"] == "Domi":  
                tts_text = "লেনদেন সম্পন্ন হয়েছে, ধন্যবাদ।"  
            else:  
                tts_text = f"Transaction completed successfully! Sent {txn.amount} USDC to address ending with {txn.receiver[-4:]}."  
            play_tts_response(tts_text, key="tts_exec_sim")  
              
            if success_anim:  
                st_lottie(success_anim, height=180, key=f"success_{txn.receiver}_{random.randint(0, 1000)}")  
            else:  
                st.balloons()  
        else:  
            # Real API Call  
            if not ARC_API_KEY:  
                st.error("❌ Cannot execute in Real Mode: Arc API Key is missing.")  
                log_transaction(txn.receiver, txn.amount, "failed", "Missing API Key")  
                continue  
              
            try:  
                time.sleep(1)   
                response = requests.post(ARC_API_URL, headers=headers, json=payload)  
                data = response.json()  
                txn_id = data.get("id")  
                  
                if response.status_code == 200 and txn_id:  
                    st.success(f"✅ Sent {txn.amount} USDC to {txn.receiver} (ID: {txn_id})")  
                    log_transaction(txn.receiver, txn.amount, "success", txn_id)  
                    st.toast(f"Sent {txn.amount} USDC successfully! ✅")  
                      
                    if st.session_state["selected_voice"] == "Domi":  
                        tts_text = "লেনদেন সম্পন্ন হয়েছে, ধন্যবাদ।"  
                    else:  
                        tts_text = f"Transaction completed successfully! Sent {txn.amount} USDC to address ending with {txn.receiver[-4:]}."  
                    play_tts_response(tts_text, key="tts_exec_real")  
                      
                    if success_anim:  
                        st_lottie(success_anim, height=180, key=f"success_{txn.receiver}_{random.randint(0, 1000)}")  
                    else:  
                        st.balloons()  
                else:  
                    error_msg = data.get("message", f"API Error {response.status_code}")  
                    st.error(f"❌ API Error for {txn.receiver}: {error_msg}")  
                    log_transaction(txn.receiver, txn.amount, "failed", error_msg)  
                      
            except Exception as e:  
                st.error(f"Transaction failed for {txn.receiver}: {e}")  
                log_transaction(txn.receiver, txn.amount, "failed", str(e))

============================================================

⚙️ ARC GUARDIAN — PART E: SIDEBAR UI

============================================================

with st.sidebar:
try:
# (আপনার ফোল্ডারে আছে)
st.image("assets/team_logo.png", width=150)
except FileNotFoundError:
st.warning("assets/team_logo.png not found.")

# --- Updated: Use Lottie animation instead of GIF ---  
ai_logo_anim = load_lottiefile("assets/ai_logo.json")  
if ai_logo_anim:  
    st_lottie(ai_logo_anim, height=200, key="ai_logo_sidebar")  
else:  
    st.image("assets/team_logo.png", width=150)  
# --- End fix ---  

st.header("🧭 Control Center")  
  
st.markdown("[🎥 Watch Demo](http.googleusercontent.com/youtube/com/2)")  
st.info("API keys loaded from `.streamlit/secrets.toml`")  
  
if not OPENAI_API_KEY: st.error("OpenAI API Key not found.")  
if not ARC_API_KEY: st.warning("Arc API Key not found.")  
else: st.success("API keys loaded successfully.")  
  
if not ELEVENLABS_API_KEY:  
    st.warning("ElevenLabs API Key not found. Voice output will be skipped.")  
  
st.toggle("🧪 Simulation Mode", value=st.session_state["simulation_mode"], key="simulation_mode",   
          help="If on, no real API calls will be made.")  
  
st.divider()  
  
st.subheader("🤖 Agent Controls")  
st.toggle("🛡️ Enable Audit Agent", value=st.session_state["enable_audit"], key="enable_audit",  
          help="If disabled, transactions will be approved automatically (DANGEROUS).")  

st.subheader("🗣️ Voice Language")  
st.selectbox(  
    "AI Voice (English/Bangla)",  
    options=["Adam", "Domi", "Rachel"], # Adam (Eng), Domi (Multi/Bangla)  
    key="selected_voice"  
)  
  
st.divider()  
  
st.subheader("💰 Wallet Status")  
st_autorefresh(interval=60000, key="refresh_balance")  
  
if not st.session_state["ai_plan"]:  
    st.session_state["mock_balance"] += random.uniform(-0.5, 0.5)   
st.metric("Current Balance (USDC)", f"{st.session_state['mock_balance']:.2f}")  
  
st.divider()  
st.subheader("🔑 Demo PIN")  
st.info(f"Use this PIN for 2FA: **{st.session_state['correct_pin']}**")  
  
st.divider()  
st.subheader("📱 Scan for Demo")  
try:  
    qr = qrcode.make(APP_URL)  
    buf = io.BytesIO()  
    qr.save(buf)  
    st.image(Image.open(buf), caption="Scan to test live on Streamlit Cloud", width=150)  
except Exception as e:  
    st.error(f"QR Code Error: {e}")  
st.divider()  
st.caption("© 2025 Team Believer")

============================================================

⚙️ ARC GUARDIAN — PART F: MAIN APP UI

============================================================

st.title("💰 Arc Guardian AI Agent")
st.caption("Built by Zahid Hasan | Team Believer 🧠 AI x FinTech Hackathon 2025")
st.markdown("<div style='background:linear-gradient(90deg,#00bcd4,#673ab7);padding:6px;border-radius:8px;text-align:center;color:white;'>💸 Arc Guardian | Secure AI Payments</div>", unsafe_allow_html=True)

st.markdown(f"<p style='color: #00e5ff; text-align: center; font-weight: bold;'>🧠 Mode: {'Audit On (Secure)' if st.session_state['enable_audit'] else 'Audit Off (Fast Mode)'}</p>", unsafe_allow_html=True)

--- Main Tabs ---

tab1, tab2 = st.tabs(["🤖 New Transaction", "📊 Dashboard & History"])

--- Tab 1: New Transaction ---

with tab1:
st.markdown("## 🎥 Hackathon Demo Voice")

demo_script = """  
# --- Tab 1: Continue (New Transaction) ---
    You can use your voice or text. Try saying: 'Send 10 USDC to 0x1234...' or 'Check my balance'.
    """

    if st.button("▶️ Play Demo Voice", use_container_width=True, type="primary"):
        play_tts_response(demo_script, key="tts_demo", voice_override="Adam")

    st.divider()
    st.markdown("### 🎙️ Input Command")

    col_mic, col_text = st.columns([1, 4])
    with col_mic:
        audio = mic_recorder(
            start_prompt="🎤 রেকর্ড শুরু করুন",
            stop_prompt="⏹️ বন্ধ করুন",
            key='recorder',
            format="wav"
        )

    with col_text:
        st.text_input("অথবা টাইপ করুন:", key="user_prompt", placeholder="e.g., Send 10 USDC to 0x1234...")

    if audio:
        st.session_state["user_prompt"] = transcribe_audio(audio["bytes"])
        st.rerun()

    if st.button("🤖 Process Command", type="primary", use_container_width=True):
        user_input = st.session_state["user_prompt"]
        plan = analyze_command_cached(user_input)
        st.session_state["ai_plan"] = plan

        if not plan:
            st.error("❌ রিকোয়েস্ট বোঝা যায়নি।")
        else:
            st.success(f"✅ Intent: {plan.action}")
            if plan.action == "TRANSACT":
                audit = analyze_audit_cached(str(plan))
                if audit:
                    audit_json = json.loads(audit)
                    st.session_state["audit_result"] = audit_json
                    st.info(f"🛡️ Audit Result: {audit_json['audit_result']}")
                else:
                    st.warning("⚠️ Audit failed, skipping...")

# --- Tab 2: Dashboard & History ---
with tab2:
    st.subheader("📊 Transaction Dashboard")

    df = pd.DataFrame(st.session_state["transactions"])
    if not df.empty:
        st.dataframe(df)
        status_counts = df['status'].value_counts()
        fig, ax = plt.subplots()
        status_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.info("এখনও কোনো লেনদেন রেকর্ড করা হয়নি।")

    st.divider()
    st.subheader("🧠 AI Reasoning Logs")
    if st.session_state["reasoning_log"]:
        log_df = pd.DataFrame(st.session_state["reasoning_log"])
        st.dataframe(log_df)
    else:
        st.info("শেষ কমান্ডের জন্য কোনো AI বিশ্লেষণ লগ করা হয়নি।")

st.success("✅ Arc Guardian AI Agent fully loaded!")
