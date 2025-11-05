import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# LangChain version compatibility
try:
    from langchain.output_parsers import PydanticOutputParser
except ImportError:
    from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import random
import time
import json
import io
import base64
import traceback

# Extra UI / media
from streamlit_lottie import st_lottie
from streamlit_mic_recorder import mic_recorder
import openai
from streamlit_autorefresh import st_autorefresh
import qrcode
from PIL import Image

# ElevenLabs
try:
    from elevenlabs import ElevenLabs
except ImportError:
    st.error("❌ ElevenLabs library missing. Add `elevenlabs` to requirements.txt")
    st.stop()

# ---------------- CONFIG ----------------
st.set_option('client.showErrorDetails', False)

st.set_page_config(
    page_title="Arc Guardian AI Agent | Team Believer",
    page_icon="assets/favicon.png",
    layout="wide"
)

# ------------------------------------------------------------
# 🔐 SECRETS & API KEYS
# ------------------------------------------------------------
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")
ARC_API_KEY = st.secrets.get("arc", {}).get("api_key")
ELEVENLABS_API_KEY = st.secrets.get("elevenlabs", {}).get("api_key")

# ✅ Put your real Arc API URL here (https://…)
ARC_API_URL = st.secrets.get("arc", {}).get("base_url", "https://api.example.com") + "/v1/transactions"

# ------------------------------------------------------------
# 🎨 UI POLISH (CSS)
# ------------------------------------------------------------
st.markdown("""
    <style>
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button[kind="secondary"] {
        background: linear-gradient(90deg, #00bcd4, #00e5ff);
        color: #000;
        border: none;
        font-weight: 700;
        transition: all .2s ease-in-out;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        box-shadow: 0 0 15px 5px #00bcd4;
        transform: scale(1.02);
    }
    [data-testid="stSidebar"] {
        border-right: 2px solid #00bcd4;
        box-shadow: 0 0 15px 5px #00bcd4;
        animation: pulse 2.5s ease-in-out infinite alternate;
    }
    @keyframes pulse { from { box-shadow: 0 0 10px 2px #00bcd4; } to { box-shadow: 0 0 20px 7px #00e5ff; } }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 🤖 MODEL SETUP
# ------------------------------------------------------------
@st.cache_resource
def get_llm():
    try:
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"gpt-4o-mini failed ({e}). Falling back to gpt-3.5-turbo.")
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

@st.cache_resource
def get_elevenlabs_client():
    if not ELEVENLABS_API_KEY:
        st.warning("🔑 ElevenLabs API key missing in secrets.toml. Voice will be disabled.")
        return None
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)

try:
    llm = get_llm()
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    eleven_client = get_elevenlabs_client()
except Exception as e:
    st.error(f"API Key setup error: {e}")
    st.stop()

# ------------------------------------------------------------
# 🔊 TTS HELPER (ElevenLabs v2)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_tts(text: str, voice_name="Adam"):
    if not eleven_client:
        return None
    try:
        # ElevenLabs SDK v2: stream iterator -> bytes
        audio_bytes_iterator = eleven_client.text_to_speech.convert(
            voice_id=voice_name.lower(),  # "adam", "domi", "rachel"
            model_id="eleven_multilingual_v2",
            text=text
        )
        return b"".join(chunk for chunk in audio_bytes_iterator)
    except Exception as e:
        st.error(f"TTS Generation failed: {e}")
        return None

def play_tts_response(text, key="tts_playback", voice_override: Optional[str] = None):
    voice = voice_override or st.session_state.get("selected_voice", "Adam")
    with st.spinner(f"🎧 Generating AI voice ({voice})..."):
        audio_bytes = generate_tts(text, voice_name=voice)
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(
            f"""
            <audio autoplay="true" style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """, unsafe_allow_html=True
        )
    else:
        st.info("TTS unavailable – check ElevenLabs API key.")

# ============================================================
# 🧠 AGENTS: Parser & Auditor
# ============================================================

class Transaction(BaseModel):
    receiver: str = Field(description="Wallet address, must start with 0x")
    amount: float = Field(description="Amount of USDC to send")
    currency: str = Field(default="USDC")

class AIPlan(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for parsing the request.")
    transactions: List[Transaction] = Field(description="List of parsed transactions.")
    action: str = Field(description="Recognized intent: TRANSACT, CHECK_BALANCE, UNKNOWN")

# Parser
try:
    parser_parser = PydanticOutputParser(pydantic_object=AIPlan)
    parser_prompt = PromptTemplate(
        template="""
You are Agent 1 (Parser). Parse the user's request into a structured JSON plan.
Actions: TRANSACT, CHECK_BALANCE, UNKNOWN.
Rules:
1) Identify intent.
2) If TRANSACT → extract receiver & amount.
3) If CHECK_BALANCE → set action accordingly.
4) Reject invalid or unclear inputs.
5) Only USDC as currency.
6) If amount ≤ 0 or > 100 → flag in reasoning.

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

# Auditor
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

# ============================================================
# ⚙️ SESSION STATE
# ============================================================
for key, default in {
    "transactions": [],
    "ai_plan": None,
    "audit_result": None,
    "reasoning_log": [],
    "correct_pin": str(random.randint(1000, 9999)),
    "simulation_mode": True,
    "user_prompt": "",
    "mock_balance": 120.0,
    "enable_audit": True,
    "selected_voice": "Adam",
    "processing": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# 🧰 HELPERS
# ============================================================
def safe_execute(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"⚠️ Unexpected Runtime Error: {e}")
        st.code(traceback.format_exc())

@st.cache_data(show_spinner=False)
def transcribe_audio(audio_bytes):
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
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_asset_as_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        if path.endswith(".mp4"):
            mime = "video/mp4"
        elif path.endswith(".png"):
            mime = "image/png"
        elif path.endswith(".gif"):
            mime = "image/gif"
        else:
            mime = "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError:
        return None

def check_balance():
    return f"Current wallet balance: {st.session_state['mock_balance']:.2f} USDC (dynamic simulation)"

def log_transaction(receiver, amount, status, detail="N/A"):
    st.session_state["transactions"].append({
        "receiver": receiver,
        "amount": amount,
        "status": status,
        "detail": detail,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def log_reasoning(agent, reasoning):
    st.session_state["reasoning_log"].append({
        "agent": agent,
        "reasoning": reasoning,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@st.cache_data(show_spinner=False)
def analyze_command_cached(user_input):
    try:
        return chain_parser.invoke({"user_input": user_input})
    except Exception as e:
        st.error(f"AI Parsing Error: {e}")
        return None

@st.cache_data(show_spinner=False)
def analyze_audit_cached(plan_string):
    try:
        return chain_auditor.invoke({"plan_string": plan_string})
    except Exception as e:
        st.error(f"AI Audit Error: {e}")
        return None

# Assets
success_anim = load_lottiefile("assets/success.json")
APP_URL = "https://arc-guardian.streamlit.app"  # your deployed URL

def execute_transactions(transactions: List[Transaction]):
    is_simulation = st.session_state["simulation_mode"]
    headers = {"Authorization": f"Bearer {ARC_API_KEY}"} if ARC_API_KEY else {}

    for txn in transactions:
        if not txn.receiver.startswith("0x") or txn.amount <= 0 or txn.amount > 100:
            st.warning(f"⚠️ Invalid transaction skipped: {txn.amount} → {txn.receiver} "
                       f"(Amount must be > 0 and ≤ 100; address must start with 0x)")
            log_transaction(txn.receiver, txn.amount, "failed", "Invalid parameters")
            continue

        payload = {"amount": txn.amount, "currency": "USDC", "receiver": txn.receiver}

        with st.spinner(f"Processing {txn.amount} USDC → {txn.receiver}..."):
            if is_simulation:
                time.sleep(1.2)
                st.success(f"✅ [SIMULATED] Sent {txn.amount} USDC to {txn.receiver}")
                log_transaction(txn.receiver, txn.amount, "success", "SIMULATED_TXN_ID")
                st.toast(f"Sent {txn.amount} USDC successfully! ✅")
                tts_text = "লেনদেন সম্পন্ন হয়েছে, ধন্যবাদ।" if st.session_state["selected_voice"] == "Domi" \
                    else f"Transaction completed. Sent {txn.amount} USDC to {txn.receiver[-4:]}."
                play_tts_response(tts_text, key=f"tts_sim_{txn.receiver}")
                if success_anim: st_lottie(success_anim, height=160, key=f"success_{txn.receiver}_{random.randint(1,9999)}")
                else: st.balloons()
            else:
                if not ARC_API_KEY:
                    st.error("❌ Real Mode requires ARC_API_KEY in secrets.")
                    log_transaction(txn.receiver, txn.amount, "failed", "Missing API Key")
                    continue
                try:
                    response = requests.post(ARC_API_URL, headers=headers, json=payload, timeout=20)
                    data = response.json() if response.content else {}
                    txn_id = data.get("id")
                    if response.ok and txn_id:
                        st.success(f"✅ Sent {txn.amount} USDC to {txn.receiver} (ID: {txn_id})")
                        log_transaction(txn.receiver, txn.amount, "success", txn_id)
                        st.toast("Payment successful! ✅")
                        tts_text = "লেনদেন সম্পন্ন হয়েছে, ধন্যবাদ।" if st.session_state["selected_voice"] == "Domi" \
                            else f"Transaction completed. Sent {txn.amount} USDC to {txn.receiver[-4:]}."
                        play_tts_response(tts_text, key=f"tts_real_{txn.receiver}")
                        if success_anim: st_lottie(success_anim, height=160, key=f"success_real_{txn_id}")
                    else:
                        error_msg = data.get("message", f"API Error {response.status_code}")
                        st.error(f"❌ API Error: {error_msg}")
                        log_transaction(txn.receiver, txn.amount, "failed", error_msg)
                except Exception as e:
                    st.error(f"Transaction failed for {txn.receiver}: {e}")
                    log_transaction(txn.receiver, txn.amount, "failed", str(e))

# ============================================================
# 🧭 SIDEBAR
# ============================================================
with st.sidebar:
    try:
        st.image("assets/team_logo.png", width=150)
    except FileNotFoundError:
        st.warning("assets/team_logo.png not found.")

    # Safe GIF load via base64 (fixes MediaFileStorageError)
    gif_b64 = get_asset_as_base64("assets/ai_brain.gif")
    if gif_b64:
        st.markdown(f'<img src="{gif_b64}" alt="AI Brain" style="border-radius:8px;max-width:240px;width:100%;">',
                    unsafe_allow_html=True)
    else:
        # Fallback Lottie
        ai_logo_anim = load_lottiefile("assets/ai_logo.json")
        if ai_logo_anim:
            st_lottie(ai_logo_anim, height=180, key="ai_logo")
        else:
            st.info("Add assets/ai_brain.gif or assets/ai_logo.json for sidebar animation.")

    st.header("🧭 Control Center")
    st.info("API keys loaded from `.streamlit/secrets.toml`")

    if not OPENAI_API_KEY: st.error("OpenAI API Key not found.")
    if not ARC_API_KEY: st.warning("Arc API Key not found (Real Mode disabled).")
    else: st.success("Arc API Key present.")

    if not ELEVENLABS_API_KEY:
        st.warning("ElevenLabs API Key not found. Voice output will be skipped.")

    st.toggle("🧪 Simulation Mode", value=st.session_state["simulation_mode"], key="simulation_mode",
              help="If ON, no real API calls will be made.")

    st.divider()

    st.subheader("🤖 Agent Controls")
    st.toggle("🛡️ Enable Audit Agent", value=st.session_state["enable_audit"], key="enable_audit",
              help="If OFF, transactions auto-approve (DANGEROUS).")

    st.subheader("🗣️ Voice")
    st.selectbox("AI Voice", options=["Adam", "Domi", "Rachel"], key="selected_voice")

    st.divider()

    st.subheader("💰 Wallet Status")
    st_autorefresh(interval=60_000, key="refresh_balance")
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
        st.image(Image.open(buf), caption="Open the live app", width=150)
    except Exception as e:
        st.error(f"QR Code Error: {e}")

    st.caption("© 2025 Team Believer")

# ============================================================
# 🖥️ MAIN
# ============================================================
st.title("💰 Arc Guardian AI Agent")
st.caption("Built by Zahid Hasan | Team Believer 🧠 AI x FinTech Hackathon 2025")
st.markdown("<div style='background:linear-gradient(90deg,#00bcd4,#673ab7);padding:6px;border-radius:8px;text-align:center;color:white;'>💸 Arc Guardian | Secure AI Payments</div>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#00e5ff;text-align:center;font-weight:bold;'>🧠 Mode: {'Audit On (Secure)' if st.session_state['enable_audit'] else 'Audit Off (Fast Mode)'}</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🤖 New Transaction", "📊 Dashboard & History"])

# --- Tab 1: New Transaction ---
with tab1:
    st.markdown("## 🎥 Hackathon Demo Voice")
    demo_script = (
        "Welcome Judges! This is Arc Guardian AI, built by Team Believer. "
        "We use AI agents to parse, audit, and execute USDC payments securely on the Arc platform. "
        "You can use your voice or text. Try saying: 'Send 10 USDC to 0x1234…' or 'Check my balance'."
    )

    if st.button("▶️ Play 30-Second Demo Voice (Judges Start Here)", use_container_width=True, type="primary",
                 disabled=st.session_state["processing"]):
        voice = st.session_state.get("selected_voice", "Adam")
        play_tts_response(demo_script, key="tts_demo", voice_override=voice)

    st.divider()
    st.markdown("### 1️⃣ আপনার কমান্ড দিন (ভয়েস বা টেক্সট)")

    col_mic, col_txt = st.columns([1, 3])
    with col_mic:
        try:
            audio = mic_recorder(start_prompt="🎤 রেকর্ড করুন", stop_prompt="⏹️ বন্ধ করুন",
                                 key="recorder", format="wav", use_container_width=True,
                                 disabled=st.session_state["processing"])
        except Exception as e:
            st.warning(f"মাইক রেকর্ডার লোড হয়নি (HTTPS প্রয়োজন): {e}")
            audio = None

    with col_txt:
        st.text_input(
            "অথবা টাইপ করুন:",
            key="user_prompt",
            placeholder="e.g., Send 10 USDC to 0x1234...ABCD | Check balance",
            disabled=st.session_state["processing"]
        )

    if audio and not st.session_state["processing"]:
        st.session_state["processing"] = True
        with st.spinner("🔄 ভয়েস ট্রান্সক্রাইব হচ্ছে..."):
            st.session_state["user_prompt"] = transcribe_audio(audio["bytes"])
        st.session_state["processing"] = False
        st.success("✅ ট্রান্সক্রিপ্ট রেডি!")

    st.markdown("### 2️⃣ প্রসেস করুন")
    if st.button("🤖 প্রসেস কমান্ড", key="process_btn", type="primary", use_container_width=True,
                 disabled=(not st.session_state["user_prompt"] or st.session_state["processing"])):
        st.session_state["processing"] = True
        st.session_state["ai_plan"] = None
        st.session_state["audit_result"] = None
        st.session_state["reasoning_log"] = []

        user_input = st.session_state["user_prompt"]

        # Agent 1 (Parser)
        with st.spinner("Agent 1 (Parser) বিশ্লেষণ করছে..."):
            plan = safe_execute(analyze_command_cached, user_input)
            st.session_state["ai_plan"] = plan

        if not plan:
            st.error("AI Parser আপনার রিকোয়েস্ট বুঝতে পারেনি।")
            log_reasoning("Agent 1", "বৈধ প্ল্যান তৈরি হয়নি।")
            play_tts_response("আমি দুঃখিত, অনুগ্রহ করে আবার চেষ্টা করুন।", key="tts_parse_fail")
            st.session_state["processing"] = False
        else:
            log_reasoning("Agent 1", plan.reasoning)

            if plan.action == "CHECK_BALANCE":
                st.success("✅ ইনটেন্ট: ব্যালেন্স চেক")
                info = check_balance()
                st.info(info)
                play_tts_response(info, key="tts_balance")
                st.session_state["processing"] = False

            elif plan.action == "TRANSACT":
    if not plan.transactions:
        st.warning("লেনদেন ইনটেন্ট পাওয়া গেছে, কিন্তু কোনো বৈধ ঠিকানা/পরিমাণ পাইনি।")
        play_tts_response("ওয়ালেট ঠিকানা বা পরিমাণ পাইনি।", key="tts_no_txn")
        st.session_state["processing"] = False
    else:
        st.success(f"✅ ইনটেন্ট: লেনদেন (মোট {len(plan.transactions)})")
        df_tx = pd.DataFrame([t.model_dump() for t in plan.transactions])
        st.dataframe(df_tx, use_container_width=True)

        # --- Agent 2 (Auditor) ---
        audit_json = {"audit_result": "APPROVED", "audit_comment": "Bypass audit"}
        if st.session_state["enable_audit"]:
            with st.spinner("Agent 2 (Auditor) ঝুঁকি বিশ্লেষণ করছে..."):
                audit_str = safe_execute(analyze_audit_cached, str(plan))
            try:
                audit_json = json.loads(audit_str)
                log_reasoning("Agent 2", audit_json.get("audit_comment", "No comment"))
            except Exception as e:
                st.error(f"অডিট JSON পার্স করা যায়নি: {e}")
                log_reasoning("Agent 2", f"Invalid JSON: {audit_str}")
                st.session_state["processing"] = False

        st.session_state["audit_result"] = audit_json
        if audit_json["audit_result"] == "APPROVED":
            st.success(f"🛡️ অডিট: {audit_json['audit_result']} ({audit_json['audit_comment']})")
        elif audit_json["audit_result"] == "FLAGGED":
            st.warning(f"🛡️ অডিট: {audit_json['audit_result']} ({audit_json['audit_comment']})")
        else:
            st.error(f"🛡️ অডিট: {audit_json['audit_result']} ({audit_json['audit_comment']})")
            play_tts_response(f"লেনদেন বাতিল। কারণ: {audit_json['audit_comment']}", key="tts_reject")
            st.session_state["processing"] = False

        # --- 2FA PIN Verification ---
        st.markdown("### 3️⃣ নিরাপত্তা যাচাই (2FA)")
        pin = st.text_input("পিন লিখুন", type="password", key="pin_input",
                            disabled=st.session_state["processing"])

        if st.button("✅ অনুমোদন করুন ও পাঠান", key="execute_btn", type="primary", use_container_width=True,
                     disabled=st.session_state["processing"]):
            if audit_json["audit_result"] != "REJECTED":
                if pin == st.session_state["correct_pin"]:
                    st.success("পিন সঠিক। লেনদেন কার্যকর হচ্ছে...")
                    play_tts_response("পিন গৃহীত হয়েছে। পেমেন্ট প্রসেস করা হচ্ছে।", key="tts_pin_ok")
                    safe_execute(execute_transactions, plan.transactions)
                    total_sent = sum(t.amount for t in plan.transactions if 0 < t.amount <= 100)
                    st.session_state["mock_balance"] -= total_sent
                    st.session_state["ai_plan"] = None
                    st.session_state["audit_result"] = None
                    st.session_state["user_prompt"] = ""
                    st.session_state["processing"] = False
                else:
                    st.error("❌ ভুল পিন। লেনদেন বাতিল।")
                    play_tts_response("ভুল পিন। লেনদেন বাতিল।", key="tts_pin_fail")
                    log_transaction("N/A", 0, "failed", "Incorrect PIN")
                    st.session_state["processing"] = False
            else:
                st.error("❌ অডিট রিজেক্ট করেছে; পাঠানো যাবে না।")
                st.session_state["processing"] = False
else:
    st.warning(f"⚠️ অজানা ইনটেন্ট: {plan.action}")
    st.info(plan.reasoning)
    play_tts_response("আমি নিশ্চিত নই কী করতে হবে।", key="tts_unknown")
    st.session_state["processing"] = False

st.divider()
st.caption("Built with ❤️ by Team Believer")

# --- Tab 2: Dashboard & History ---
with tab2:
    st.subheader("📊 মূল মেট্রিক্স")
    df_hist = pd.DataFrame(st.session_state["transactions"])
    total_txn = len(df_hist)
    success_count = df_hist['status'].value_counts().get('success', 0) if total_txn else 0
    time_saved_minutes = total_txn * 1.5

    c1, c2, c3 = st.columns(3)
    c1.metric("মোট লেনদেন", total_txn)
    c2.metric("সফল লেনদেন", success_count)
    c3.metric("সময় সাশ্রয় (মিনিট)", f"{time_saved_minutes:.1f}")

    st.divider()
    st.subheader("📈 লেনদেনের ইতিহাস")
    if total_txn > 0:
        st.dataframe(df_hist.sort_values(by="timestamp", ascending=False), use_container_width=True)

        try:
            st.subheader("লেনদেনের স্ট্যাটাস ব্রেকডাউন")
            status_counts = df_hist['status'].value_counts()
            fig, ax = plt.subplots()
            status_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"চার্ট তৈরি হয়নি: {e}")
    else:
        st.info("এখনও কোনো লেনদেন রেকর্ড করা হয়নি।")

    st.divider()
    st.subheader("🧠 AI এজেন্ট অ্যানালাইসিস লগ")
    if st.session_state["reasoning_log"]:
        log_df = pd.DataFrame(st.session_state["reasoning_log"])
        st.dataframe(log_df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("শেষ কমান্ডের জন্য কোনো AI বিশ্লেষণ লগ নেই।")
