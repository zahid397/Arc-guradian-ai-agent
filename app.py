import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# LangChain ভার্সন সামঞ্জস্যপূর্ণ করার ফিক্স
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
import base64 # QR কোডের জন্য
import traceback # গ্লোবাল এক্সেপশন UI-এর জন্য
import os # ফাইল পাথ চেকের জন্য

# --- 🔄 আপডেটেড ইম্পোর্ট: ফ্রি ভয়েস-টু-টেক্সট ---
from streamlit_speech_to_text import st_speech_to_text
# from streamlit_mic_recorder import mic_recorder # <-- সরানো হয়েছে
# import openai # <-- LLM-এর জন্য প্রয়োজন নেই, langchain এটি সামলাবে

# অটো-রিফ্রেশ
from streamlit_autorefresh import st_autorefresh

# QR Code
import qrcode
from PIL import Image

# --- ElevenLabs এবং Lottie সরানো হয়েছে ---

# ---------------- CONFIG ----------------
st.set_option('client.showErrorDetails', False)

st.set_page_config(
    page_title="Arc Guardian AI Agent | Team Believer",
    page_icon="assets/favicon.png", # অ্যাসেট পাথ
    layout="wide"
)

# ------------------------------------------------------------
# 🔐 SECRETS & API KEYS
# ------------------------------------------------------------
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key")
ARC_API_KEY = st.secrets.get("arc", {}).get("api_key")
# ELEVENLABS_API_KEY সরানো হয়েছে

# ------------------------------------------------------------
# 🎨 UI POLISH (CSS INJECTION)
# ------------------------------------------------------------
st.markdown("""
    <style>
    /* ... (আপনার CSS কোড অপরিবর্তিত) ... */
    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# 🤖 MODEL SETUP
# ------------------------------------------------------------
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

# --- ElevenLabs ক্লায়েন্ট সরানো হয়েছে ---

try:
    llm = get_llm()
    # client = openai.OpenAI(api_key=OPENAI_API_KEY) # <-- Whisper-এর জন্য আর প্রয়োজন নেই
except Exception as e:
    st.error(f"API Key setup error: {e}")
    st.stop()

# ------------------------------------------------------------
# 🔊 TTS HELPER FUNCTION ( সরানো হয়েছে )
# ------------------------------------------------------------
def play_tts_response(text, key="tts_playback", voice_override=None):
    """ভয়েস আউটপুট ক্লাউডের জন্য ডিসেবল করা হয়েছে।"""
    pass # কোনো কিছু না করে সাইলেন্টলি স্কিপ

# ============================================================
# 🧠 ARC GUARDIAN — PART B: AGENTS SETUP
# ============================================================

# --- Pydantic Models ---
class Transaction(BaseModel):
    receiver: str = Field(description="Wallet address, must start with 0x")
    amount: float = Field(description="Amount of USDC to send")
    currency: str = Field(default="USDC")

class AIPlan(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for parsing the request.")
    transactions: List[Transaction] = Field(description="List of parsed transactions.")
    action: str = Field(description="Recognized intent: TRANSACT, CHECK_BALANCE, UNKNOWN")

# --- Agent 1: Parser Agent ---
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

# --- Agent 2: Audit Agent ---
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
# ⚙️ ARC GUARDIAN — PART C: SESSION STATE
# ============================================================
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
if "processing" not in st.session_state:
    st.session_state["processing"] = False

# ============================================================
# ⚙️ ARC GUARDIAN — PART D: HELPER FUNCTIONS
# ============================================================

def safe_execute(func, *args, **kwargs):
    """Global error handler wrapper."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"⚠️ Unexpected Runtime Error: {e}")
        st.code(traceback.format_exc()) # Shows traceback

# --- 🔄 সরানো হয়েছে: transcribe_audio ফাংশন ---
# @st.cache_data(show_spinner=False)
# def transcribe_audio(audio_bytes):
#     ... (এই কোডটি আর প্রয়োজন নেই)

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

# --- Asset Loading ---
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
                st.balloons() # Lottie-এর বদলে বেলুন ফলব্যাক
            else:
                # Real API Call
                # ... (আপনার কোড অপরিবর্তিত) ...
                pass # আপনার আসল API কল লজিক এখানে থাকবে

# ============================================================
# ⚙️ ARC GUARDIAN — PART E: SIDEBAR UI
# ============================================================
with st.sidebar:
    try:
        st.image("assets/team_logo.png", width=150)
    except FileNotFoundError:
        st.warning("assets/team_logo.png not found.")
    
    st.markdown("---") # অ্যানিমেশন সরানো হয়েছে

    st.header("🧭 Control Center")
    
    st.markdown("[🎥 Watch Demo](http.googleusercontent.com/youtube/com/2)")
    st.info("API keys loaded from `.streamlit/secrets.toml`")
    
    if not OPENAI_API_KEY: st.error("OpenAI API Key not found.")
    if not ARC_API_KEY: st.warning("Arc API Key not found.")
    else: st.success("API keys loaded successfully.")
    
    st.toggle("🧪 Simulation Mode", value=st.session_state["simulation_mode"], key="simulation_mode", 
              help="If on, no real API calls will be made.")
    
    st.divider()
    
    st.subheader("🤖 Agent Controls")
    st.toggle("🛡️ Enable Audit Agent", value=st.session_state["enable_audit"], key="enable_audit",
              help="If disabled, transactions will be approved automatically (DANGEROUS).")

    # --- ভয়েস ল্যাঙ্গুয়েজ সেকশন সরানো হয়েছে ---
    
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

# ============================================================
# ⚙️ ARC GUARDIAN — PART F: MAIN APP UI
# ============================================================
st.title("💰 Arc Guardian AI Agent")
st.caption("Built by Zahid Hasan | Team Believer 🧠 AI x FinTech Hackathon 2025")
st.markdown("<div style='background:linear-gradient(90deg,#00bcd4,#673ab7);padding:6px;border-radius:8px;text-align:center;color:white;'>💸 Arc Guardian | Secure AI Payments</div>", unsafe_allow_html=True)

st.markdown(f"<p style='color: #00e5ff; text-align: center; font-weight: bold;'>🧠 Mode: {'Audit On (Secure)' if st.session_state['enable_audit'] else 'Audit Off (Fast Mode)'}</p>", unsafe_allow_html=True)

# --- Global Data Calculation ---
df = pd.DataFrame(st.session_state["transactions"])
total_txn = len(df)
success_count = 0
time_saved = 0.0
if total_txn > 0:
    success_count = df['status'].value_counts().get('success', 0)
    time_saved = total_txn * 1.5

# --- Main Tabs ---
tab1, tab2 = st.tabs(["🤖 New Transaction", "📊 Dashboard & History"])

# --- Tab 1: New Transaction ---
with tab1:
    
    # --- ভয়েস ডেমো বাটন সরানো হয়েছে ---
    
    st.markdown("---") 

    with st.container(border=True):
        st.subheader("1. Enter Your Command")
        
        # --- 🔄 আপডেটেড UI: টেক্সটবক্স এবং ফ্রি ভয়েস বাটন ---
        
        st.info("🎙️ কথা বলতে মাইক্রোফোন বাটনটি চাপুন (Chrome/Edge ব্রাউজার প্রয়োজন)। এটি বিনামূল্যে।")
        
        # টেক্সট এরিয়া, যেখানে কমান্ড টাইপ করা যাবে বা ভয়েস থেকে আসবে
        st.text_area(
            "Type command or use microphone:",
            height=100,
            label_visibility="collapsed",
            key="user_prompt", # এই key-টিই বাকি লজিক ব্যবহার করে
            disabled=st.session_state["processing"]
        )

        # ফ্রি ভয়েস-টু-টেক্সট বাটন
        speech_text = st_speech_to_text(
            start_prompt="🎙️ কথা বলুন...",
            stop_prompt="⏹️ প্রসেসিং...",
            language="en-US", # ইংরেজি (US)
            key="speech_input_free",
            disabled=st.session_state["processing"],
            use_container_width=True
        )
        
        # লজিক: যদি ভয়েস থেকে টেক্সট আসে, তাহলে উপরের টেক্সট এরিয়া আপডেট করুন
        if speech_text:
            st.session_state["user_prompt"] = speech_text
            st.rerun() # টেক্সটবক্সে লেখাটি দেখানোর জন্য রি-রান

        # --- mic_recorder এবং if audio: ব্লকটি সরানো হয়েছে ---

        if st.button("Analyze Command 🧠", use_container_width=True, disabled=st.session_state["processing"]):
            st.session_state["processing"] = True
            
            def run_analysis():
                user_input = st.session_state["user_prompt"]
                if not user_input:
                    st.warning("Please enter a command or use the microphone.")
                    st.session_state["processing"] = False
                    return
                if not OPENAI_API_KEY:
                    st.error("OpenAI API key is not configured.")
                    st.session_state["processing"] = False
                    return

                with st.spinner("🧠 Agent 1 (Parser) is analyzing..."):
                    ai_plan = analyze_command_cached(user_input)
                
                if ai_plan:
                    st.session_state["ai_plan"] = ai_plan
                    log_reasoning("Parser", ai_plan.reasoning)
                    
                    if ai_plan.action == "TRANSACT":
                        if st.session_state["enable_audit"]:
                            with st.spinner("🛡️ Agent 2 (Auditor) is reviewing the plan..."):
                                plan_str = ai_plan.model_dump_json()
                                audit_response_str = analyze_audit_cached(plan_str)
                                
                                # --- ফিক্স ২: JSONDecodeError হ্যান্ডেলিং ---
                                try:
                                    audit_result = json.loads(audit_response_str)
                                except Exception: 
                                    st.warning("Audit Agent response invalid, forcing fallback → APPROVED")
                                    audit_result = {"audit_result": "APPROVED", "audit_comment": "Auto-approved (invalid JSON)"}
                                
                                st.session_state["audit_result"] = audit_result
                                log_reasoning("Auditor", audit_result.get("audit_comment", "No comment."))
                                # --- ফিক্স শেষ ---
                        else:
                            st.session_state["audit_result"] = {
                                "audit_result": "APPROVED",
                                "audit_comment": "Audit Agent was skipped by user."
                            }
                            log_reasoning("Auditor", "Audit Agent skipped by user.")
                    else:
                        st.session_state["audit_result"] = None
                else:
                    st.session_state["ai_plan"] = None
                    log_transaction("N/A", 0, "failed", "AI Parsing Error")
                
                st.session_state["processing"] = False
                st.rerun() 

            safe_execute(run_analysis) # Use the safe wrapper

    # --- Step 2: Review & Confirm Plan ---
    if st.session_state["ai_plan"]:
        plan = st.session_state["ai_plan"]
        audit = st.session_state.get("audit_result")
        
        with st.container(border=True):
            if plan.action == "CHECK_BALANCE":
                balance_text = check_balance()
                st.success(f"🤖 AI recognized 'Check Balance': {balance_text}")
                st.session_state["ai_plan"] = None
                st.session_state["audit_result"] = None

            elif plan.action == "TRANSACT":
                st.subheader("2. Review and Confirm Plan")
                
                if audit:
                    audit_status = audit.get("audit_result", "REJECTED")
                    audit_comment = audit.get("audit_comment", "No comment.")
                    
                    if audit_status == "APPROVED":
                        st.success(f"**Audit Status:** ✅ **APPROVED**\n\n*Auditor's Note: {audit_comment}*")
                    elif audit_status == "FLAGGED":
                        st.warning(f"**Audit Status:** ⚠️ **FLAGGED (Execution Halted)**\n\n*Auditor's Note: {audit_comment}*")
                    elif audit_status == "REJECTED":
                        st.error(f"**Audit Status:** 🚫 **REJECTED (Execution Halted)**\n\n*Auditor's Note: {audit_comment}*")
                else:
                    st.error("🛡️ Audit Agent: Could not review the plan. Execution halted.")
                    audit_status = "REJECTED"

                st.dataframe(pd.DataFrame([t.model_dump() for t in plan.transactions]))
                
                with st.expander("💡 Parser Agent Explanation"):
                    st.info(plan.reasoning)
                
                if audit_status == "APPROVED":
                    st.divider()
                    
                    user_pin = st.text_input("Enter 2FA PIN to Confirm:", type="password", key="pin_confirm", disabled=st.session_state["processing"])
                    
                    if st.button("Confirm & Execute Transactions ✅", use_container_width=True, type="primary", disabled=st.session_state["processing"]):
                        st.session_state["processing"] = True
                        
                        def run_confirmation():
                            if user_pin != st.session_state["correct_pin"]:
                                st.error("❌ Invalid PIN. Transactions aborted.")
                                st.session_state["processing"] = False
                            else:
                                st.success("✅ PIN Accepted. Executing transactions...")
                                execute_transactions(plan.transactions)
                                st.session_state["ai_plan"] = None
                                st.session_state["audit_result"] = None
                                st.session_state["processing"] = False
                                st.rerun() 
                        
                        safe_execute(run_confirmation) # Use the safe wrapper

            elif plan.action == "UNKNOWN":
                st.error(f"🤖 AI could not process this request. Reason: {plan.reasoning}")
                st.session_state["ai_plan"] = None
                st.session_state["audit_result"] = None

# --- Tab 2: Dashboard & History ---
with tab2:
    # ... (আপনার ড্যাশবোর্ডের সমস্ত কোড অপরিবর্তিত) ...
    st.subheader("📊 Transaction Dashboard & History")
    
    if total_txn > 0:
        total_amount = df[df['status'] == 'success']['amount'].sum()
        st.success(f"💸 Total USDC Sent: {total_amount:.2f} | Successful Transactions: {success_count}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Successful Txn", success_count)
        col2.metric("⚠️ Failed Txn", total_txn - success_count)
        col3.metric("⏱️ Time Saved (Est.)", f"{time_saved:.1f} mins")
        
        # ... (বাকি ড্যাশবোর্ড কোড) ...
    else:
        st.info("No transactions yet. Make your first transaction in the 'New Transaction' tab.")


# ============================================================
# ⚙️ ARC GUARDIAN — PART G: FOOTER & CREDITS
# ============================================================
# ... (আপনার ফুটারের সমস্ত কোড অপরিবর্তিত) ...
st.markdown("---")
st.caption("Powered by Arc + OpenAI | Built by Zahid Hasan 🚀")
st.caption("© 2025 Team Believer")
