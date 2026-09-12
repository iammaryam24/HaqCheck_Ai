import os
import re
import html
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

# Optional image/OCR dependencies
try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
except ImportError:
    cv2 = np = pytesseract = Image = None


# ============================================================
# HAQCHECK_AI — FINAL STREAMLIT APPLICATION
# ============================================================

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
MAX_SOURCES = 5
REQUEST_TIMEOUT = 15


# ------------------------------------------------------------
# CONFIG / HELPERS
# ------------------------------------------------------------

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Create a .env file in the "
            "HaqCheck_AI folder and add GROQ_API_KEY=your_key."
        )
    return Groq(api_key=api_key)


def source_level(url):
    domain = urlparse(url).netloc.lower().replace("www.", "")

    high = [
        "reuters.com", "apnews.com", "abcnews.com", "bbc.com", "bbc.co.uk",
        "dawn.com", "aljazeera.com", "who.int", "un.org", "gov.pk",
        ".gov", ".gov.uk", ".gov.au", ".edu"
    ]
    medium = [
        "tribune.com.pk", "thenews.com.pk", "geo.tv", "arynews.tv",
        "urdupoint.com", "express.com.pk"
    ]

    if any(domain == x or domain.endswith("." + x) for x in high):
        return "High"
    if any(domain == x or domain.endswith("." + x) for x in medium):
        return "Medium"
    return "Low"


def credibility_rank(level):
    return {"High": 0, "Medium": 1, "Low": 2}.get(level, 2)


def safe_text(value):
    return html.escape(str(value or ""))


def parse_confidence(value):
    if isinstance(value, (int, float)):
        return max(0, min(100, int(value)))
    match = re.search(r"(\d{1,3})", str(value or ""))
    return max(0, min(100, int(match.group(1)))) if match else 0


def normalize_verdict(value):
    text = str(value or "Unverified").strip()
    upper = text.upper()
    if "MISLEADING" in upper:
        return "Misleading"
    if "LIKELY FALSE" in upper or upper == "FALSE" or upper.startswith("FALSE"):
        return "Likely False"
    if "LIKELY TRUE" in upper or upper == "TRUE" or upper.startswith("TRUE"):
        return "Likely True"
    if "UNVERIFIED" in upper or "INSUFFICIENT" in upper:
        return "Unverified"
    return text[:80] if text else "Unverified"


# ------------------------------------------------------------
# AI CLAIM ANALYSIS
# ------------------------------------------------------------

def analyze_claim(claim):
    client = get_groq_client()

    prompt = f"""
You are the Claim Analysis Agent of HaqCheck_AI.

Analyze this claim before verification:

CLAIM:
{claim}

Identify concisely:
1. Main factual claim
2. Key entities
3. Important dates or numbers
4. Claim type
5. What specifically needs to be verified
6. Useful search keywords

Do NOT decide whether the claim is true or false.
Do NOT invent facts or sources.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a careful fact-checking claim analysis agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_completion_tokens=700,
    )
    return response.choices[0].message.content.strip()


# ------------------------------------------------------------
# WEB RESEARCH
# ------------------------------------------------------------

def web_search(query, max_results=5):
    results = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                url = (item.get("href") or "").strip()
                if not url:
                    continue
                results.append({
                    "title": item.get("title", "Untitled source"),
                    "url": url,
                    "snippet": item.get("body", ""),
                    "credibility": source_level(url),
                })
    except Exception as exc:
        st.session_state["last_search_error"] = str(exc)
    return results


def multi_query_research(claim, per_query=4):
    queries = [
        claim,
        f"fact check {claim}",
        f"{claim} official source",
        f"news {claim}",
    ]

    unique = {}
    for query in queries:
        for item in web_search(query, max_results=per_query):
            url = item.get("url", "")
            if url and url not in unique:
                unique[url] = item

    sources = list(unique.values())
    sources.sort(key=lambda x: credibility_rank(x.get("credibility", "Low")))
    return sources[:MAX_SOURCES]


# ------------------------------------------------------------
# SOURCE CONTENT EXTRACTION
# ------------------------------------------------------------

def fetch_article(url, max_chars=6000):
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        return text[:max_chars]
    except Exception:
        return ""


def enrich_sources(sources):
    enriched = []
    for source in sources:
        content = fetch_article(source["url"])
        enriched.append({**source, "content": content})
    return enriched


# ------------------------------------------------------------
# FINAL EVIDENCE ANALYSIS
# ------------------------------------------------------------

def analyze_evidence(claim, sources):
    client = get_groq_client()

    evidence_parts = []
    for i, source in enumerate(sources, 1):
        evidence_parts.append(
            f"""
SOURCE {i}
Title: {source.get('title', '')}
URL: {source.get('url', '')}
Credibility: {source.get('credibility', 'Low')}
Search snippet: {source.get('snippet', '')[:1800]}
Article content: {source.get('content', '')[:5000]}
--------------------------------------------------
"""
        )

    evidence_text = "\n".join(evidence_parts)

    if not sources:
        evidence_text = "NO RELIABLE WEB SOURCES WERE RETRIEVED."

    prompt = f"""
You are the Final Evidence Analyst for HaqCheck_AI.

CLAIM:
{claim}

RESEARCH EVIDENCE:
{evidence_text}

Evaluate ONLY the evidence supplied above.

Return EXACTLY these fields:
VERDICT: Likely True / Likely False / Misleading / Unverified
CONFIDENCE: 0-100
SUMMARY: one concise paragraph
CONFIRMED FACTS: bullet points
UNCERTAIN OR CONTRADICTING CLAIMS: bullet points
REASONING: concise evidence-based reasoning
SOURCE QUALITY: High / Medium / Low

RULES:
- Never invent facts, sources, URLs, or quotations.
- Do not treat an allegation as a proven fact.
- Do not treat OCR output as automatically correct.
- Prefer multiple independent high-credibility sources.
- If only part of a claim is supported, use Misleading when appropriate.
- If reliable evidence is insufficient, use Unverified.
- Confidence must reflect evidence quality, not certainty of the language model.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a rigorous evidence-based fact-checking analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_completion_tokens=1200,
    )
    return response.choices[0].message.content.strip()


def parse_final_verdict(text):
    text = text or ""

    def field(name, next_fields):
        pattern = rf"{re.escape(name)}\s*:\s*(.*?)(?=\n(?:{'|'.join(map(re.escape, next_fields))})\s*:|\Z)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    fields = [
        "VERDICT", "CONFIDENCE", "SUMMARY", "CONFIRMED FACTS",
        "UNCERTAIN OR CONTRADICTING CLAIMS", "REASONING", "SOURCE QUALITY"
    ]

    verdict = normalize_verdict(field("VERDICT", fields[1:]))
    confidence = parse_confidence(field("CONFIDENCE", fields[2:]))
    summary = field("SUMMARY", fields[3:])
    confirmed = field("CONFIRMED FACTS", fields[4:])
    uncertain = field("UNCERTAIN OR CONTRADICTING CLAIMS", fields[5:])
    reasoning = field("REASONING", fields[6:])
    source_quality = field("SOURCE QUALITY", []) or "Low"

    if not summary:
        summary = reasoning or "The available evidence was insufficient to produce a stronger conclusion."
    if not reasoning:
        reasoning = summary

    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "confirmed_facts": confirmed,
        "uncertain_claims": uncertain,
        "reasoning": reasoning,
        "source_quality": source_quality,
    }


# ------------------------------------------------------------
# TEXT VERIFICATION PIPELINE
# ------------------------------------------------------------

def verify_claim(claim):
    claim = claim.strip()
    if not claim:
        raise ValueError("Please provide a claim.")

    analysis = analyze_claim(claim)
    sources = multi_query_research(claim, per_query=4)
    enriched = enrich_sources(sources)
    final_text = analyze_evidence(claim, enriched)
    verdict = parse_final_verdict(final_text)

    return {
        "status": "success",
        "input_type": "text",
        "claim": claim,
        "claim_analysis": analysis,
        "sources": enriched,
        "final_text": final_text,
        "verdict": verdict,
    }


def safe_verify_claim(claim):
    try:
        return verify_claim(claim)
    except Exception as exc:
        return {
            "status": "error",
            "input_type": "text",
            "claim": claim,
            "claim_analysis": "",
            "sources": [],
            "final_text": "",
            "verdict": {
                "verdict": "Unverified",
                "confidence": 0,
                "summary": "The claim could not be reliably verified because the verification pipeline encountered an error.",
                "confirmed_facts": "",
                "uncertain_claims": "",
                "reasoning": str(exc),
                "source_quality": "Low",
            },
            "error": str(exc),
        }


# ------------------------------------------------------------
# IMAGE OCR
# ------------------------------------------------------------

def configure_tesseract():
    if pytesseract is None:
        raise RuntimeError("OCR packages are not installed. Install pytesseract, Pillow, opencv-python and numpy.")

    # Windows default installation locations.
    candidates = [
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path

    # If tesseract is already on PATH, this succeeds during OCR.
    return "PATH"


def improved_ocr(image):
    configure_tesseract()

    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    outputs = []
    for psm in (6, 11, 12):
        try:
            text = pytesseract.image_to_string(
                processed,
                lang="eng+urd",
                config=f"--psm {psm}",
            ).strip()
        except Exception:
            # Fall back to English if Urdu language data is not installed.
            text = pytesseract.image_to_string(
                processed,
                lang="eng",
                config=f"--psm {psm}",
            ).strip()
        if text and text not in outputs:
            outputs.append(text)

    return "\n\n--- OCR PASS ---\n\n".join(outputs)


def extract_claim_from_ocr(ocr_text):
    if not ocr_text.strip():
        return {
            "claim": "",
            "key_facts": "",
            "context": "",
            "status": "failed",
        }

    client = get_groq_client()
    prompt = f"""
You are HaqCheck_AI's Image Claim Extraction Agent.

The following text was extracted from an Urdu/English image using OCR and may contain errors.

OCR TEXT:
{ocr_text[:8000]}

Extract ONLY the main factual claim that can be verified online.
Correct obvious OCR mistakes when the intended meaning is clear.
Ignore usernames, follower counts, buttons, emojis, advertisements and UI decoration.
Preserve important names, numbers, dates, locations and events.
Do not verify the claim yet.

Return exactly:
CLAIM: <one concise factual claim>
KEY FACTS: <important names, numbers, dates, locations>
CONTEXT: <one short sentence>
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You extract factual claims from noisy OCR text without verifying them."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_completion_tokens=500,
        )
        result = response.choices[0].message.content.strip()
    except Exception:
        result = ""

    claim_match = re.search(r"CLAIM\s*:\s*(.*?)(?=\nKEY FACTS\s*:|\Z)", result, re.I | re.S)
    facts_match = re.search(r"KEY FACTS\s*:\s*(.*?)(?=\nCONTEXT\s*:|\Z)", result, re.I | re.S)
    context_match = re.search(r"CONTEXT\s*:\s*(.*)\Z", result, re.I | re.S)

    claim = claim_match.group(1).strip() if claim_match else ""
    facts = facts_match.group(1).strip() if facts_match else ""
    context = context_match.group(1).strip() if context_match else ""

    # Fallback if the model response is malformed.
    if len(claim) < 15:
        lines = [x.strip() for x in ocr_text.splitlines() if len(x.strip()) > 25]
        claim = " ".join(lines[:5]).strip()

    return {
        "claim": claim,
        "key_facts": facts,
        "context": context,
        "status": "success" if claim else "failed",
    }


def verify_image(image):
    ocr_text = improved_ocr(image)
    extraction = extract_claim_from_ocr(ocr_text)
    claim = extraction["claim"].strip()

    if not claim:
        return {
            "status": "failed",
            "input_type": "image",
            "ocr_text": ocr_text,
            "extraction": extraction,
            "verdict": {
                "verdict": "Unverified",
                "confidence": 0,
                "summary": "No meaningful factual claim could be extracted from the image.",
                "confirmed_facts": "",
                "uncertain_claims": "",
                "reasoning": "OCR did not produce enough reliable text to perform web verification.",
                "source_quality": "Low",
            },
            "sources": [],
        }

    result = verify_claim(claim)
    result.update({
        "input_type": "image",
        "ocr_text": ocr_text,
        "extraction": extraction,
    })
    return result


def safe_verify_image(image):
    try:
        return verify_image(image)
    except Exception as exc:
        return {
            "status": "error",
            "input_type": "image",
            "ocr_text": "",
            "extraction": {"claim": "", "key_facts": "", "context": ""},
            "sources": [],
            "verdict": {
                "verdict": "Unverified",
                "confidence": 0,
                "summary": "Image verification could not be completed.",
                "confirmed_facts": "",
                "uncertain_claims": "",
                "reasoning": str(exc),
                "source_quality": "Low",
            },
            "error": str(exc),
        }


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="HaqCheck_AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, rgba(79,70,229,.16), transparent 32%),
                    radial-gradient(circle at 90% 10%, rgba(37,99,235,.13), transparent 30%),
                    #070b14;
        color: #f8fafc;
    }
    .block-container { max-width: 1100px; padding-top: 1.2rem; }
    .hero { text-align:center; padding: 30px 10px 20px; }
    .logo { font-size: 44px; font-weight: 850; letter-spacing:-1.5px;
            background: linear-gradient(90deg,#60a5fa,#a78bfa); -webkit-background-clip:text;
            -webkit-text-fill-color:transparent; }
    .tagline { color:#94a3b8; font-size:16px; margin-top:7px; }
    .card { background:rgba(15,23,42,.78); border:1px solid rgba(148,163,184,.14);
            border-radius:18px; padding:22px; margin:12px 0; }
    .title { font-size:21px; font-weight:750; color:#f8fafc; }
    .muted { color:#94a3b8; }
    .verdict { border-radius:20px; padding:25px; margin-top:18px;
               background:linear-gradient(145deg,rgba(30,41,59,.96),rgba(15,23,42,.96));
               border:1px solid rgba(96,165,250,.22); }
    .verdict-name { font-size:32px; font-weight:850; }
    .confidence { color:#93c5fd; font-size:18px; margin-top:5px; }
    .source { background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.12);
              border-radius:15px; padding:17px; margin:10px 0; }
    .footer { text-align:center; color:#64748b; margin:55px 0 20px; font-size:13px; }
    div[data-testid="stFileUploader"] { background: rgba(15,23,42,.55); border-radius:15px; padding:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="logo">🔎 HaqCheck_AI</div>
        <div class="tagline">Verify Claims. Understand Context. Find the Truth.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
        <div class="title">🧠 AI-Powered Information Intelligence</div>
        <div class="muted" style="margin-top:7px;">
            Investigate claims using AI reasoning, web research, source credibility,
            webpage evidence and cautious cross-verification.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Verification mode",
    ["📝 Text Claim", "🖼️ Image", "🎥 Video"],
    horizontal=True,
)


# ------------------------------------------------------------
# TEXT MODE
# ------------------------------------------------------------

if mode == "📝 Text Claim":
    st.markdown('<div class="title">Enter a claim to verify</div>', unsafe_allow_html=True)
    claim = st.text_area(
        "Claim",
        placeholder="Paste a news claim, headline, social-media statement or allegation...",
        height=160,
        label_visibility="collapsed",
    )

    if st.button("🔎 Verify Claim", type="primary", use_container_width=True):
        if not claim.strip():
            st.warning("Please enter a claim before starting verification.")
        else:
            progress = st.progress(0)
            status = st.empty()
            try:
                status.info("🧠 Analyzing claim...")
                progress.progress(15)

                status.info("🌐 Searching relevant sources...")
                progress.progress(35)

                status.info("🔎 Evaluating source credibility and extracting evidence...")
                progress.progress(60)

                status.info("⚖️ Generating evidence-based assessment...")
                result = safe_verify_claim(claim)
                progress.progress(100)
                status.empty()

                st.session_state["last_result"] = result
                st.session_state["last_claim"] = claim
                st.rerun()
            except Exception as exc:
                progress.empty()
                status.empty()
                st.error(f"Verification failed: {exc}")


# ------------------------------------------------------------
# IMAGE MODE
# ------------------------------------------------------------

elif mode == "🖼️ Image":
    st.markdown('<div class="title">Upload a claim screenshot or image</div>', unsafe_allow_html=True)
    st.caption("HaqCheck_AI extracts Urdu/English text with OCR, reconstructs the factual claim, then verifies it on the web.")

    uploaded_image = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("🖼️ Verify Image Claim", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            try:
                status.info("📝 Extracting text with OCR...")
                progress.progress(25)
                result = safe_verify_image(image)
                progress.progress(100)
                status.empty()
                st.session_state["last_result"] = result
                st.session_state["last_claim"] = result.get("extraction", {}).get("claim", "")
                st.rerun()
            except Exception as exc:
                progress.empty()
                status.empty()
                st.error(f"Image verification failed: {exc}")


# ------------------------------------------------------------
# VIDEO MODE — ROADMAP
# ------------------------------------------------------------

else:
    st.info(
        "🎥 Video verification is planned for the next phase. "
        "The current production release supports Text Claim and Image verification."
    )
    st.file_uploader(
        "Video upload (planned)",
        type=["mp4", "mov", "avi", "mkv"],
        disabled=True,
    )


# ------------------------------------------------------------
# RESULT DISPLAY
# ------------------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    verdict = result.get("verdict", {})

    st.markdown('<div class="title" style="margin-top:28px;">📊 Verification Result</div>', unsafe_allow_html=True)

    verdict_name = safe_text(verdict.get("verdict", "Unverified"))
    confidence = parse_confidence(verdict.get("confidence", 0))
    source_quality = safe_text(verdict.get("source_quality", "Low"))

    st.markdown(
        f"""
        <div class="verdict">
            <div class="verdict-name">{verdict_name}</div>
            <div class="confidence">Confidence: {confidence}% &nbsp; • &nbsp; Source quality: {source_quality}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get("input_type") == "image":
        extraction = result.get("extraction", {})
        ocr_text = result.get("ocr_text", "")
        if extraction.get("claim"):
            st.markdown('<div class="card"><div class="title">📌 Extracted Claim</div></div>', unsafe_allow_html=True)
            st.write(extraction["claim"])
        with st.expander("📝 OCR Text"):
            st.text(ocr_text or "No OCR text detected.")

    st.markdown('<div class="card"><div class="title">🧾 Summary</div></div>', unsafe_allow_html=True)
    st.write(verdict.get("summary", ""))

    if verdict.get("confirmed_facts"):
        st.markdown('<div class="card"><div class="title">✅ Confirmed Facts</div></div>', unsafe_allow_html=True)
        st.write(verdict["confirmed_facts"])

    if verdict.get("uncertain_claims"):
        st.markdown('<div class="card"><div class="title">⚠️ Uncertain / Contradicting Claims</div></div>', unsafe_allow_html=True)
        st.write(verdict["uncertain_claims"])

    st.markdown('<div class="card"><div class="title">🧠 Reasoning</div></div>', unsafe_allow_html=True)
    st.write(verdict.get("reasoning", ""))

    sources = result.get("sources", [])
    st.markdown('<div class="title" style="margin-top:24px;">🌐 Evidence Sources</div>', unsafe_allow_html=True)

    if not sources:
        st.warning("No web sources were retrieved. The result should remain Unverified.")
    else:
        for i, source in enumerate(sources, 1):
            title = safe_text(source.get("title", "Untitled source"))
            url = source.get("url", "")
            credibility = safe_text(source.get("credibility", "Low"))
            snippet = safe_text(source.get("snippet", ""))
            st.markdown(
                f"""
                <div class="source">
                    <strong>{i}. {title}</strong><br>
                    <span class="muted">Credibility: {credibility}</span><br>
                    <a href="{safe_text(url)}" target="_blank">{safe_text(url)}</a>
                    <div style="margin-top:8px;">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result.get("error"):
        st.warning("The system encountered an error, so this result is intentionally cautious and should not be treated as a definitive fact-check.")


st.markdown(
    '<div class="footer">HaqCheck_AI • AI-powered claim verification & information intelligence</div>',
    unsafe_allow_html=True,
)
