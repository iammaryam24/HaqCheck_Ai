import os
import re
import html
import json
import tempfile
from datetime import datetime
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
MAX_HISTORY = 10


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
        "nasa.gov", "noaa.gov", "nih.gov", "cdc.gov", "fda.gov",
        "usgs.gov", "nps.gov", "state.gov", ".edu"
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
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# ------------------------------------------------------------
# IMAGE OCR
# ------------------------------------------------------------

def configure_tesseract():
    if pytesseract is None:
        raise RuntimeError("OCR packages are not installed. Install pytesseract, Pillow, opencv-python and numpy.")

    candidates = [
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
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
        return {"claim": "", "key_facts": "", "context": "", "status": "failed"}

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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# ------------------------------------------------------------
# VIDEO VERIFICATION — OCR + SPEECH-TO-TEXT
# ------------------------------------------------------------

VIDEO_STT_MODEL = "whisper-large-v3"
VIDEO_STT_MAX_BYTES = 25 * 1024 * 1024


def extract_video_frames(video_path, max_frames=6):
    if cv2 is None or Image is None:
        raise RuntimeError(
            "Video processing requires opencv-python, numpy and Pillow. "
            "Please install the required packages."
        )

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened or decoded.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0

    if total_frames > 0:
        frame_indexes = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    else:
        frame_indexes = np.arange(max_frames)

    frames = []
    for index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append({
            "index": int(index),
            "time": (float(index) / fps) if fps > 0 else 0.0,
            "image": Image.fromarray(frame_rgb),
        })

    capture.release()
    return frames, duration


def transcribe_video_audio(video_bytes, file_name):
    if not video_bytes:
        return ""

    if len(video_bytes) > VIDEO_STT_MAX_BYTES:
        raise ValueError(
            "This video is larger than 25 MB. Please upload a shorter/compressed video "
            "for speech-to-text verification."
        )

    client = get_groq_client()
    safe_name = os.path.basename(file_name or "video.mp4")

    transcription = client.audio.transcriptions.create(
        file=(safe_name, video_bytes),
        model=VIDEO_STT_MODEL,
        response_format="json",
        temperature=0.0,
    )

    return (getattr(transcription, "text", "") or "").strip()


def extract_video_claim(ocr_text, speech_text):
    combined = []
    if ocr_text.strip():
        combined.append("VISIBLE VIDEO TEXT (OCR):\n" + ocr_text[:10000])
    if speech_text.strip():
        combined.append("SPOKEN AUDIO TRANSCRIPT:\n" + speech_text[:12000])

    if not combined:
        return {"claim": "", "key_facts": "", "context": "", "status": "failed"}

    client = get_groq_client()
    prompt = f"""
You are HaqCheck_AI's Video Claim Extraction Agent.

A video was analyzed using two channels:
1. OCR from visible video frames.
2. Speech-to-text transcription from the video's audio.

The extracted text may contain transcription/OCR errors.

{chr(10).join(combined)}

Extract ONLY the main factual claim that can be verified online.
Prefer a clear factual statement from the spoken transcript when one exists.
Use visible OCR text to supplement or correct the claim only when the intended meaning is clear.
Do not verify the claim yet.
Do not invent facts, names, dates or numbers.
Ignore greetings, opinions, calls to action, usernames, advertisements and filler speech.

Return exactly:
CLAIM: <one concise factual claim>
KEY FACTS: <important names, numbers, dates, locations>
CONTEXT: <one short sentence>
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You extract factual claims from noisy video OCR and speech transcripts without verifying them.",
                },
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

    return {
        "claim": claim,
        "key_facts": facts,
        "context": context,
        "status": "success" if claim else "failed",
    }


def verify_video(video_bytes, file_name, max_frames=6):
    suffix = os.path.splitext(file_name or "video.mp4")[1].lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        suffix = ".mp4"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        frames, duration = extract_video_frames(temp_path, max_frames=max_frames)
        if not frames:
            raise ValueError("No readable frames could be extracted from this video.")

        frame_ocr = []
        for number, frame_data in enumerate(frames, 1):
            try:
                text = improved_ocr(frame_data["image"]).strip()
            except Exception as exc:
                text = f"[OCR failed for frame {number}: {exc}]"

            if text and not text.startswith("[OCR failed"):
                frame_ocr.append(
                    f"FRAME {number} (timestamp {frame_data['time']:.1f}s):\n{text}"
                )

        ocr_text = "\n\n--- VIDEO FRAME ---\n\n".join(frame_ocr).strip()
        speech_text = transcribe_video_audio(video_bytes, file_name)
        extraction = extract_video_claim(ocr_text, speech_text)
        claim = extraction.get("claim", "").strip()

        base_result = {
            "input_type": "video",
            "file_name": file_name,
            "duration": round(duration, 2),
            "frames_analyzed": len(frames),
            "ocr_text": ocr_text,
            "speech_text": speech_text,
            "extraction": extraction,
        }

        if not claim:
            return {
                **base_result,
                "status": "failed",
                "sources": [],
                "verdict": {
                    "verdict": "Unverified",
                    "confidence": 0,
                    "summary": "No reliable factual claim could be extracted from the video's visible text or speech.",
                    "confirmed_facts": "",
                    "uncertain_claims": "",
                    "reasoning": "The video was processed through both frame OCR and speech-to-text, but neither channel provided enough information to identify a single verifiable factual claim.",
                    "source_quality": "Low",
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        result = verify_claim(claim)
        result.update(base_result)
        result["status"] = "success"
        return result

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def safe_verify_video(video_bytes, file_name):
    try:
        return verify_video(video_bytes, file_name)
    except Exception as exc:
        return {
            "status": "error",
            "input_type": "video",
            "file_name": file_name,
            "sources": [],
            "ocr_text": "",
            "speech_text": "",
            "extraction": {"claim": "", "key_facts": "", "context": "", "status": "failed"},
            "verdict": {
                "verdict": "Unverified",
                "confidence": 0,
                "summary": "Video verification could not be completed.",
                "confirmed_facts": "",
                "uncertain_claims": "",
                "reasoning": str(exc),
                "source_quality": "Low",
            },
            "error": str(exc),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


# ------------------------------------------------------------
# HISTORY & EXPORT HELPERS
# ------------------------------------------------------------

def add_to_history(result):
    if "history" not in st.session_state:
        st.session_state["history"] = []
    entry = {
        "claim": result.get("claim") or result.get("extraction", {}).get("claim", ""),
        "verdict": result.get("verdict", {}).get("verdict", "Unverified"),
        "confidence": result.get("verdict", {}).get("confidence", 0),
        "input_type": result.get("input_type", "text"),
        "timestamp": result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "result": result,
    }
    st.session_state["history"].insert(0, entry)
    st.session_state["history"] = st.session_state["history"][:MAX_HISTORY]


def result_to_markdown(result):
    verdict = result.get("verdict", {})
    lines = [
        f"# HaqCheck_AI Verification Report",
        f"",
        f"**Timestamp:** {result.get('timestamp', 'N/A')}  ",
        f"**Input Type:** {result.get('input_type', 'text').title()}  ",
        f"**Claim:** {result.get('claim') or result.get('extraction', {}).get('claim', '')}",
        f"",
        f"## Verdict: {verdict.get('verdict', 'Unverified')}",
        f"**Confidence:** {verdict.get('confidence', 0)}%  ",
        f"**Source Quality:** {verdict.get('source_quality', 'Low')}",
        f"",
        f"## Summary",
        verdict.get("summary", ""),
        f"",
        f"## Confirmed Facts",
        verdict.get("confirmed_facts", "None"),
        f"",
        f"## Uncertain / Contradicting Claims",
        verdict.get("uncertain_claims", "None"),
        f"",
        f"## Reasoning",
        verdict.get("reasoning", ""),
        f"",
        f"## Evidence Sources",
    ]
    for i, src in enumerate(result.get("sources", []), 1):
        lines.append(f"{i}. [{src.get('title', 'Untitled')}]({src.get('url', '')}) — *{src.get('credibility', 'Low')} credibility*")
    return "\n".join(lines)


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="HaqCheck_AI — Verify Claims",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- PREMIUM CSS ----------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

    <style>
    :root {
        --bg-primary: #060912;
        --bg-secondary: #0b1120;
        --bg-tertiary: #111a2e;
        --border-soft: rgba(148, 163, 184, 0.12);
        --border-glow: rgba(96, 165, 250, 0.35);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #60a5fa;
        --accent-purple: #a78bfa;
        --accent-cyan: #22d3ee;
        --success: #34d399;
        --danger: #f87171;
        --warning: #fbbf24;
        --info: #38bdf8;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* ---------- ANIMATED BACKGROUND ---------- */
    .stApp {
        background-color: var(--bg-primary);
        background-image:
            radial-gradient(circle at 15% 0%, rgba(96, 165, 250, 0.15), transparent 40%),
            radial-gradient(circle at 85% 10%, rgba(167, 139, 250, 0.13), transparent 42%),
            radial-gradient(circle at 50% 100%, rgba(34, 211, 238, 0.08), transparent 45%);
        color: var(--text-primary);
        position: relative;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: -20%;
        left: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(96,165,250,0.18), transparent 70%);
        border-radius: 50%;
        filter: blur(60px);
        animation: floatOrb 18s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::after {
        content: "";
        position: fixed;
        bottom: -25%;
        right: -10%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(167,139,250,0.14), transparent 70%);
        border-radius: 50%;
        filter: blur(80px);
        animation: floatOrb 22s ease-in-out infinite reverse;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes floatOrb {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(60px, -40px) scale(1.15); }
    }

    /* ---------- LAYOUT ---------- */
    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 3rem;
        position: relative;
        z-index: 1;
    }

    /* ---------- SCROLLBAR ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
        border-radius: 8px;
        border: 2px solid var(--bg-secondary);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-blue));
    }

    /* ---------- HERO ---------- */
    .hero {
        text-align: center;
        padding: 42px 20px 26px;
        position: relative;
    }

    .logo {
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -2px;
        background: linear-gradient(120deg, #60a5fa 0%, #a78bfa 45%, #22d3ee 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 6s ease-in-out infinite;
        display: inline-block;
    }

    @keyframes shimmer {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .tagline {
        color: var(--text-secondary);
        font-size: 16px;
        margin-top: 10px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 14px;
        padding: 5px 14px;
        background: rgba(96, 165, 250, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 999px;
        font-size: 12px;
        color: var(--accent-blue);
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ---------- CARDS ---------- */
    .card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.85), rgba(11, 17, 32, 0.9));
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 22px 24px;
        margin: 12px 0;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(96,165,250,0.5), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }

    .card:hover {
        border-color: rgba(96, 165, 250, 0.28);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px -12px rgba(96, 165, 250, 0.25);
    }

    .card:hover::before { opacity: 1; }

    .title {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .muted { color: var(--text-secondary); }

    /* ---------- VERDICT PANEL ---------- */
    .verdict {
        border-radius: 22px;
        padding: 28px 30px;
        margin: 20px 0 18px;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .verdict.true {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.15), rgba(6, 78, 59, 0.25));
        border: 1px solid rgba(52, 211, 153, 0.4);
        box-shadow: 0 0 60px -20px rgba(52, 211, 153, 0.4);
    }
    .verdict.false {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.15), rgba(127, 29, 29, 0.25));
        border: 1px solid rgba(248, 113, 113, 0.4);
        box-shadow: 0 0 60px -20px rgba(248, 113, 113, 0.4);
    }
    .verdict.misleading {
        background: linear-gradient(145deg, rgba(251, 191, 36, 0.15), rgba(120, 53, 15, 0.25));
        border: 1px solid rgba(251, 191, 36, 0.4);
        box-shadow: 0 0 60px -20px rgba(251, 191, 36, 0.4);
    }
    .verdict.unverified {
        background: linear-gradient(145deg, rgba(148, 163, 184, 0.12), rgba(30, 41, 59, 0.3));
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 0 60px -20px rgba(148, 163, 184, 0.3);
    }

    .verdict-name {
        font-size: 36px;
        font-weight: 900;
        letter-spacing: -0.8px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .verdict.true .verdict-name { color: var(--success); }
    .verdict.false .verdict-name { color: var(--danger); }
    .verdict.misleading .verdict-name { color: var(--warning); }
    .verdict.unverified .verdict-name { color: #cbd5e1; }

    .confidence {
        color: #cbd5e1;
        font-size: 15px;
        margin-top: 10px;
        font-weight: 500;
    }

    .confidence strong {
        color: var(--accent-cyan);
        font-weight: 700;
    }

    /* ---------- CONFIDENCE BAR ---------- */
    .confidence-bar-wrap {
        margin-top: 16px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
        border: 1px solid var(--border-soft);
    }

    .confidence-bar {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
        position: relative;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 0 12px rgba(96, 165, 250, 0.6);
    }

    .confidence-bar::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent);
        animation: shimmerBar 2.5s infinite;
    }

    @keyframes shimmerBar {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    /* ---------- BADGES ---------- */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }

    .badge-high {
        background: rgba(52, 211, 153, 0.15);
        color: var(--success);
        border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .badge-medium {
        background: rgba(251, 191, 36, 0.15);
        color: var(--warning);
        border: 1px solid rgba(251, 191, 36, 0.35);
    }
    .badge-low {
        background: rgba(248, 113, 113, 0.15);
        color: var(--danger);
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    /* ---------- SOURCE CARDS ---------- */
    .source {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.7), rgba(11, 17, 32, 0.8));
        border: 1px solid var(--border-soft);
        border-radius: 16px;
        padding: 18px 20px;
        margin: 10px 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .source::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));
        opacity: 0;
        transition: opacity 0.3s;
    }

    .source:hover {
        border-color: rgba(96, 165, 250, 0.35);
        transform: translateX(4px);
        box-shadow: 0 8px 30px -10px rgba(96, 165, 250, 0.3);
    }

    .source:hover::before { opacity: 1; }

    .source a {
        color: var(--accent-blue);
        text-decoration: none;
        font-weight: 500;
        word-break: break-all;
        transition: color 0.2s;
    }
    .source a:hover { color: var(--accent-cyan); text-decoration: underline; }

    .source-snippet {
        margin-top: 10px;
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.6;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px -4px rgba(96, 165, 250, 0.5) !important;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px -4px rgba(96, 165, 250, 0.7) !important;
        filter: brightness(1.1);
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ---------- INPUTS ---------- */
    .stTextArea textarea, .stTextInput input {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.25s !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.15) !important;
    }

    /* ---------- RADIO ---------- */
    div[role="radiogroup"] {
        gap: 12px;
    }

    div[role="radiogroup"] label {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        transition: all 0.25s !important;
        cursor: pointer;
    }

    div[role="radiogroup"] label:hover {
        border-color: rgba(96, 165, 250, 0.4) !important;
        background: rgba(30, 41, 59, 0.7) !important;
    }

    /* ---------- FILE UPLOADER ---------- */
    div[data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.55) !important;
        border-radius: 16px !important;
        padding: 12px !important;
        border: 1px dashed rgba(96, 165, 250, 0.3) !important;
        transition: all 0.3s;
    }

    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(96, 165, 250, 0.6) !important;
        background: rgba(15, 23, 42, 0.75) !important;
    }

    /* ---------- PROGRESS ---------- */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important;
        border-radius: 999px;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 17, 32, 0.98), rgba(6, 9, 18, 0.98)) !important;
        border-right: 1px solid var(--border-soft);
    }

    .history-item {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--border-soft);
        border-radius: 12px;
        padding: 12px 14px;
        margin: 8px 0;
        transition: all 0.25s;
        cursor: pointer;
    }

    .history-item:hover {
        border-color: rgba(96, 165, 250, 0.4);
        background: rgba(30, 41, 59, 0.7);
        transform: translateX(3px);
    }

    .history-claim {
        font-size: 13px;
        color: var(--text-primary);
        font-weight: 500;
        margin-bottom: 6px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    .history-meta {
        font-size: 11px;
        color: var(--text-muted);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* ---------- FOOTER ---------- */
    .footer {
        text-align: center;
        color: var(--text-muted);
        margin: 60px 0 20px;
        font-size: 13px;
        padding-top: 24px;
        border-top: 1px solid var(--border-soft);
        position: relative;
        z-index: 1;
    }

    /* ---------- SECTION DIVIDER ---------- */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(96,165,250,0.3), transparent);
        margin: 24px 0;
    }

    /* ---------- EXAMPLE CHIPS ---------- */
    .chip {
        display: inline-block;
        padding: 8px 14px;
        background: rgba(96, 165, 250, 0.08);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 999px;
        color: var(--accent-blue);
        font-size: 13px;
        margin: 4px 6px 4px 0;
        transition: all 0.25s;
        cursor: pointer;
        text-decoration: none;
    }

    .chip:hover {
        background: rgba(96, 165, 250, 0.18);
        border-color: rgba(96, 165, 250, 0.5);
        transform: translateY(-1px);
        color: var(--accent-cyan);
    }

    /* ---------- RESPONSIVE ---------- */
    @media (max-width: 768px) {
        .logo { font-size: 36px; }
        .verdict-name { font-size: 26px; }
        .hero { padding: 24px 10px 16px; }
        .card { padding: 16px 18px; }
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    }

    /* Hide Streamlit branding */
    #MainMenu, footer[data-testid="stFooter"] { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- SESSION STATE INIT ----------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "pending_claim" not in st.session_state:
    st.session_state["pending_claim"] = ""


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 18px;">
            <div style="font-size: 24px; font-weight: 900; background: linear-gradient(120deg,#60a5fa,#a78bfa);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🔎 HaqCheck_AI
            </div>
            <div style="font-size: 12px; color:#64748b; margin-top:4px;">
                AI Fact-Checking Suite
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Checks", len(st.session_state["history"]))
    with col2:
        if st.session_state["history"]:
            avg_conf = int(sum(h["confidence"] for h in st.session_state["history"]) / len(st.session_state["history"]))
            st.metric("Avg Conf.", f"{avg_conf}%")
        else:
            st.metric("Avg Conf.", "—")

    st.markdown("---")

    st.markdown("### 🕒 History")
    if not st.session_state["history"]:
        st.caption("No verifications yet. Start by checking a claim.")
    else:
        verdict_icons = {
            "Likely True": "✅",
            "Likely False": "❌",
            "Misleading": "⚠️",
            "Unverified": "❓",
        }
        for idx, item in enumerate(st.session_state["history"][:8]):
            icon = verdict_icons.get(item["verdict"], "❓")
            claim_preview = (item["claim"][:70] + "...") if len(item["claim"]) > 70 else item["claim"]
            st.markdown(
                f"""
                <div class="history-item">
                    <div class="history-claim">{icon} {safe_text(claim_preview)}</div>
                    <div class="history-meta">
                        <span>{item['confidence']}% conf.</span>
                        <span>{item['timestamp'].split(' ')[1] if ' ' in item['timestamp'] else ''}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    st.markdown("---")
    st.caption("💡 **Tip:** Add `GROQ_API_KEY` in a `.env` file for best results.")


# ---------- HERO ----------
st.markdown(
    """
    <div class="hero">
        <div class="logo">🔎 HaqCheck_AI</div>
        <div class="tagline">Verify Claims. Understand Context. Find the Truth.</div>
        <div class="hero-badge">⚡ AI-Powered Verification Engine</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
        <div class="title">🧠 AI-Powered Information Intelligence</div>
        <div class="muted" style="margin-top:8px; line-height:1.6;">
            Investigate claims using AI reasoning, multi-source web research,
            source credibility scoring, and cautious evidence-based cross-verification.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- MODE SELECTOR ----------
mode = st.radio(
    "Verification mode",
    ["📝 Text Claim", "🖼️ Image", "🎥 Video"],
    horizontal=True,
    label_visibility="collapsed",
)


# ---------- TEXT MODE ----------
if mode == "📝 Text Claim":
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">✍️ Enter a claim to verify</div>', unsafe_allow_html=True)

    # Example chips
    st.markdown(
        """
        <div style="margin: 10px 0 12px;">
            <span style="font-size:12px; color:#64748b; margin-right:8px;">Try an example:</span>
            <a class="chip" href="#" onclick="return false;">Pakistan inflation 2024</a>
            <a class="chip" href="#" onclick="return false;">COVID vaccine efficacy</a>
            <a class="chip" href="#" onclick="return false;">Climate change data</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    claim = st.text_area(
        "Claim",
        value=st.session_state.get("pending_claim", ""),
        placeholder="Paste a news claim, headline, social-media statement or allegation...",
        height=170,
        label_visibility="collapsed",
    )

    if st.button("🔎 Verify Claim", type="primary", use_container_width=True):
        if not claim.strip():
            st.warning("⚠️ Please enter a claim before starting verification.")
        else:
            st.session_state["pending_claim"] = ""
            progress = st.progress(0)
            status = st.empty()
            try:
                status.info("🧠 Analyzing claim structure...")
                progress.progress(15)

                status.info("🌐 Searching relevant sources across the web...")
                progress.progress(35)

                status.info("🔎 Extracting evidence & scoring source credibility...")
                progress.progress(60)

                status.info("⚖️ Generating evidence-based assessment...")
                result = safe_verify_claim(claim)
                progress.progress(100)
                status.empty()

                st.session_state["last_result"] = result
                add_to_history(result)
                st.rerun()
            except Exception as exc:
                progress.empty()
                status.empty()
                st.error(f"Verification failed: {exc}")


# ---------- IMAGE MODE ----------
elif mode == "🖼️ Image":
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">🖼️ Upload a claim screenshot or image</div>', unsafe_allow_html=True)
    st.caption("Extracts Urdu/English text with OCR, reconstructs the factual claim, then verifies it against web sources.")

    uploaded_image = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_image is not None and Image is not None:
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
                add_to_history(result)
                st.rerun()
            except Exception as exc:
                progress.empty()
                status.empty()
                st.error(f"Image verification failed: {exc}")


# ---------- VIDEO MODE ----------
else:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">🎥 Upload a video to verify</div>', unsafe_allow_html=True)
    st.caption(
        "Combines representative-frame OCR with Groq Whisper speech-to-text, "
        "then extracts and verifies a factual claim against web evidence."
    )

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        st.info(
            "🎙️ Video verification uses both visible text (OCR) and spoken audio (Whisper). "
            "For best results, use a clear video under 25 MB."
        )

        if st.button("🎥 Verify Video Claim", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            try:
                status.info("🎞️ Extracting representative video frames...")
                progress.progress(20)

                status.info("📝 Reading visible text from video frames with OCR...")
                progress.progress(35)

                status.info("🎙️ Transcribing spoken audio with Groq Whisper...")
                progress.progress(55)

                video_bytes = uploaded_video.getvalue()
                result = safe_verify_video(video_bytes, uploaded_video.name)

                progress.progress(80)
                status.info("🌐 Verifying the extracted claim against web evidence...")

                progress.progress(100)
                status.empty()

                st.session_state["last_result"] = result
                add_to_history(result)
                st.rerun()
            except Exception as exc:
                progress.empty()
                status.empty()
                st.error(f"Video verification failed: {exc}")


# ---------- RESULT DISPLAY ----------
if st.session_state.get("last_result"):
    result = st.session_state["last_result"]
    verdict = result.get("verdict", {})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">📊 Verification Result</div>', unsafe_allow_html=True)

    verdict_name = verdict.get("verdict", "Unverified")
    confidence = parse_confidence(verdict.get("confidence", 0))
    source_quality = verdict.get("source_quality", "Low")

    # Verdict-specific class + icon
    verdict_class = {
        "Likely True": "true",
        "Likely False": "false",
        "Misleading": "misleading",
        "Unverified": "unverified",
    }.get(verdict_name, "unverified")

    verdict_icon = {
        "Likely True": "✅",
        "Likely False": "❌",
        "Misleading": "⚠️",
        "Unverified": "❓",
    }.get(verdict_name, "❓")

    st.markdown(
        f"""
        <div class="verdict {verdict_class}">
            <div class="verdict-name">{verdict_icon} {safe_text(verdict_name)}</div>
            <div class="confidence">
                Confidence: <strong>{confidence}%</strong> &nbsp;•&nbsp;
                Source Quality: <strong>{safe_text(source_quality)}</strong>
                &nbsp;•&nbsp;
                <span style="color:#64748b; font-size:13px;">{safe_text(result.get('timestamp',''))}</span>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-bar" style="width: {confidence}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input-specific extras
    if result.get("input_type") in {"image", "video"}:
        extraction = result.get("extraction", {})
        ocr_text = result.get("ocr_text", "")
        if extraction.get("claim"):
            st.markdown(
                f"""
                <div class="card">
                    <div class="title">📌 Extracted Claim</div>
                    <div style="margin-top:10px; color:#cbd5e1; line-height:1.6;">{safe_text(extraction['claim'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if result.get("input_type") == "video":
            duration = result.get("duration", 0)
            frames_analyzed = result.get("frames_analyzed", 0)
            st.caption(f"🎞️ Duration: {duration}s • Frames analyzed: {frames_analyzed}")

        with st.expander("📝 Extracted OCR Text"):
            st.text(ocr_text or "No readable text detected.")

        if result.get("input_type") == "video":
            with st.expander("🎙️ Speech-to-Text Transcript"):
                st.text(result.get("speech_text", "") or "No spoken audio was detected or transcribed.")

    # Summary
    st.markdown(
        f"""
        <div class="card">
            <div class="title">🧾 Summary</div>
            <div style="margin-top:12px; color:#e2e8f0; line-height:1.7;">{safe_text(verdict.get('summary',''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confirmed facts
    if verdict.get("confirmed_facts"):
        st.markdown(
            f"""
            <div class="card">
                <div class="title">✅ Confirmed Facts</div>
                <div style="margin-top:12px; color:#d1fae5; line-height:1.7; white-space:pre-wrap;">{safe_text(verdict['confirmed_facts'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Uncertain claims
    if verdict.get("uncertain_claims"):
        st.markdown(
            f"""
            <div class="card">
                <div class="title">⚠️ Uncertain / Contradicting Claims</div>
                <div style="margin-top:12px; color:#fef3c7; line-height:1.7; white-space:pre-wrap;">{safe_text(verdict['uncertain_claims'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Reasoning
    st.markdown(
        f"""
        <div class="card">
            <div class="title">🧠 Reasoning</div>
            <div style="margin-top:12px; color:#cbd5e1; line-height:1.7; white-space:pre-wrap;">{safe_text(verdict.get('reasoning',''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sources
    sources = result.get("sources", [])
    st.markdown('<div class="title" style="margin-top:28px;">🌐 Evidence Sources</div>', unsafe_allow_html=True)

    if not sources:
        st.warning("No web sources were retrieved. The result should remain Unverified.")
    else:
        st.caption(f"Retrieved {len(sources)} source(s), sorted by credibility.")
        for i, source in enumerate(sources, 1):
            title = safe_text(source.get("title", "Untitled source"))
            url = source.get("url", "")
            cred = source.get("credibility", "Low")
            badge_class = {
                "High": "badge-high",
                "Medium": "badge-medium",
                "Low": "badge-low",
            }.get(cred, "badge-low")
            snippet = safe_text(source.get("snippet", ""))
            st.markdown(
                f"""
                <div class="source">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
                        <strong style="color:#f1f5f9; font-size:15px;">{i}. {title}</strong>
                        <span class="badge {badge_class}">{cred}</span>
                    </div>
                    <div style="margin-top:8px;">
                        <a href="{safe_text(url)}" target="_blank">{safe_text(url)}</a>
                    </div>
                    <div class="source-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Error note
    if result.get("error"):
        st.warning("⚠️ The system encountered an error. This result is intentionally cautious and should not be treated as definitive.")

    # ---------- EXPORT BUTTONS ----------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">📥 Export Report</div>', unsafe_allow_html=True)

    md_report = result_to_markdown(result)
    json_report = json.dumps(result, indent=2, default=str)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "📄 Download Markdown",
            data=md_report,
            file_name=f"haqcheck_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "🧾 Download JSON",
            data=json_report,
            file_name=f"haqcheck_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_c:
        if st.button("🗑️ Clear Result", use_container_width=True):
            st.session_state["last_result"] = None
            st.rerun()


st.markdown(
    """
    <div class="footer">
        🔎 <strong>HaqCheck_AI</strong> • AI-powered claim verification & information intelligence<br>
        <span style="font-size:11px;">Always cross-check critical information with primary sources.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
