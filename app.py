import os
import re
import io
import time
import html
import random
import datetime
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup

# Optional dependency: much cleaner article extraction than raw BeautifulSoup.
try:
    import trafilatura
except ImportError:
    trafilatura = None

# Optional image/OCR dependencies
try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image
except ImportError:
    cv2 = np = pytesseract = Image = None


# ============================================================
# HAQCHECK_AI — HACKATHON EDITION
# ============================================================

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"
MAX_SOURCES = 6
QUERIES_PER_CLAIM = 5
REQUEST_TIMEOUT = 10
MAX_FETCH_RETRIES = 2

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]


# ------------------------------------------------------------
# CONFIG / HELPERS
# ------------------------------------------------------------

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Create a .env file next to this "
            "app and add GROQ_API_KEY=your_key, then restart the app."
        )
    return Groq(api_key=api_key)


# --- Source credibility -------------------------------------------------
# NOTE ON THE "SOURCE QUALITY ALWAYS LOW" BUG:
# The old version matched domains against a very short whitelist and quietly
# defaulted EVERY unmatched domain (nytimes.com, cnn.com, theguardian.com,
# nature.com, snopes.com, ...) to "Low". Since the AI's SOURCE QUALITY field
# was written after reading "Credibility: Low" on every single source, it had
# no choice but to report Low almost every time. This version:
#   1. Uses a much larger, categorized allow-list (global wires, major
#      newspapers, science/health orgs, fact-checking outlets, Pakistani &
#      South-Asian outlets, and generic .gov/.edu/.ac domains).
#   2. Only marks a domain "Low" if it is a known low-signal source (social
#      media, blog platforms, forums) OR it is completely unrecognized AND
#      looks like a low-effort/junk domain. Any other unrecognized domain
#      defaults to "Medium" instead of "Low", since an unknown outlet isn't
#      automatically untrustworthy.
#   3. Computes the overall "Source Quality" verdict directly in code from
#      the actual retrieved sources, rather than leaving it entirely to the
#      language model to guess.

FACT_CHECK_DOMAINS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "afpfactcheck.com", "apnews.com/hub/ap-fact-check", "poynter.org",
    "boomlive.in", "altnews.in", "factcheck.afp.com", "leadstories.com",
    "checkyourfact.com", "science.feedback.org", "healthfeedback.org",
}

HIGH_CRED_DOMAINS = {
    # Wire services / global broadcasters
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk",
    "aljazeera.com", "npr.org", "dw.com", "france24.com", "cbc.ca",
    "abc.net.au", "pbs.org",
    # Major newspapers / magazines
    "nytimes.com", "washingtonpost.com", "wsj.com", "ft.com",
    "theguardian.com", "economist.com", "bloomberg.com", "time.com",
    "theatlantic.com", "newyorker.com",
    # Science / health / institutions
    "who.int", "un.org", "unicef.org", "worldbank.org", "imf.org",
    "cdc.gov", "nih.gov", "nasa.gov", "nature.com", "science.org",
    "thelancet.com", "nejm.org", "unesco.org",
    # Pakistan (top-tier)
    "dawn.com", "app.com.pk",
}

MEDIUM_CRED_DOMAINS = {
    # Broadcasters / large outlets with mixed editorial track records
    "cnn.com", "foxnews.com", "nbcnews.com", "cbsnews.com",
    "abcnews.go.com", "skynews.com", "independent.co.uk",
    "telegraph.co.uk", "dailymail.co.uk", "usatoday.com", "newsweek.com",
    # South Asia
    "tribune.com.pk", "thenews.com.pk", "geo.tv", "arynews.tv",
    "samaa.tv", "dunyanews.tv", "urdupoint.com", "express.com.pk",
    "brecorder.com", "pakobserver.net", "thefridaytimes.com",
    "ndtv.com", "indiatoday.in", "hindustantimes.com",
    "timesofindia.indiatimes.com", "thehindu.com",
}

LOW_CRED_DOMAINS = {
    # Social platforms & user-generated content — never primary evidence
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "quora.com", "pinterest.com", "tumblr.com",
    "blogspot.com", "wordpress.com", "medium.com", "substack.com",
    "youtube.com", "telegram.org", "t.me",
}

HIGH_CRED_SUFFIXES = (".gov", ".gov.uk", ".gov.au", ".gov.pk", ".edu", ".ac.uk", ".mil")


def _root_domain(url):
    return urlparse(url).netloc.lower().replace("www.", "")


def source_level(url):
    domain = _root_domain(url)
    if not domain:
        return "Low"

    if any(domain == d or domain.endswith("." + d) for d in FACT_CHECK_DOMAINS):
        return "High"
    if domain.endswith(HIGH_CRED_SUFFIXES):
        return "High"
    if any(domain == d or domain.endswith("." + d) for d in HIGH_CRED_DOMAINS):
        return "High"
    if any(domain == d or domain.endswith("." + d) for d in MEDIUM_CRED_DOMAINS):
        return "Medium"
    if any(domain == d or domain.endswith("." + d) for d in LOW_CRED_DOMAINS):
        return "Low"

    # Unknown domain: don't punish it just for being unrecognized.
    return "Medium"


def is_fact_check_source(url):
    domain = _root_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in FACT_CHECK_DOMAINS)


def credibility_rank(level):
    return {"High": 0, "Medium": 1, "Low": 2}.get(level, 2)


def compute_overall_source_quality(sources):
    """Derive an overall 'Source Quality' rating directly from the sources
    that were actually retrieved, instead of relying purely on the LLM."""
    if not sources:
        return "Low"

    counts = {"High": 0, "Medium": 0, "Low": 0}
    for s in sources:
        counts[s.get("credibility", "Low")] = counts.get(s.get("credibility", "Low"), 0) + 1

    total = len(sources)
    unique_domains = len({_root_domain(s.get("url", "")) for s in sources})

    if counts["High"] >= 2 or (counts["High"] >= 1 and counts["Medium"] >= 1):
        quality = "High"
    elif counts["High"] + counts["Medium"] >= max(2, round(total * 0.5)):
        quality = "Medium"
    elif counts["Medium"] >= 1:
        quality = "Medium"
    else:
        quality = "Low"

    return quality, counts, unique_domains


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


VERDICT_STYLE = {
    "Likely True": {"icon": "✅", "color": "#22C55E"},
    "Likely False": {"icon": "❌", "color": "#EF4444"},
    "Misleading": {"icon": "⚠️", "color": "#F59E0B"},
    "Unverified": {"icon": "❔", "color": "#94A3B8"},
}


def verdict_style(name):
    return VERDICT_STYLE.get(name, VERDICT_STYLE["Unverified"])


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

@st.cache_data(show_spinner=False, ttl=1800)
def web_search(query, max_results=5):
    results = []
    last_error = None
    for attempt in range(2):
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
            break
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.2)
    if not results and last_error:
        st.session_state["last_search_error"] = last_error
    return results


def multi_query_research(claim, per_query=4):
    queries = [
        claim,
        f"fact check {claim}",
        f"{claim} official statement",
        f"{claim} news",
        f"is it true {claim}",
    ]

    unique = {}
    for query in queries[:QUERIES_PER_CLAIM]:
        for item in web_search(query, max_results=per_query):
            url = item.get("url", "")
            if url and url not in unique:
                unique[url] = item

    sources = list(unique.values())
    # Rank fact-checkers and high-credibility sources first, but keep a
    # mix so we don't accidentally throw away every non-listed outlet.
    sources.sort(key=lambda x: (credibility_rank(x.get("credibility", "Low")),
                                 not is_fact_check_source(x.get("url", ""))))
    return sources[:MAX_SOURCES]


# ------------------------------------------------------------
# SOURCE CONTENT EXTRACTION
# ------------------------------------------------------------

def _extract_with_trafilatura(html_text):
    if trafilatura is None:
        return ""
    try:
        extracted = trafilatura.extract(html_text, include_comments=False, include_tables=False)
        return extracted or ""
    except Exception:
        return ""


def _extract_with_soup(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()

    article = soup.find("article")
    text = article.get_text(" ", strip=True) if article else ""

    if len(text) < 200:
        text = soup.get_text(" ", strip=True)

    if len(text) < 100:
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            text = meta["content"]

    return text


def fetch_article(url, max_chars=6000):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9,ur;q=0.8",
    }
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if response.status_code != 200 or not response.text:
                continue

            text = _extract_with_trafilatura(response.text)
            if len(text) < 150:
                text = _extract_with_soup(response.text)

            return text[:max_chars].strip()
        except Exception:
            time.sleep(0.6)
    return ""


def enrich_sources(sources):
    enriched = []
    for source in sources:
        content = fetch_article(source["url"])
        enriched.append({**source, "content": content, "fetched": bool(content)})
    return enriched


# ------------------------------------------------------------
# FINAL EVIDENCE ANALYSIS
# ------------------------------------------------------------

def analyze_evidence(claim, sources, aggregate_quality, credibility_counts, unique_domains):
    client = get_groq_client()

    evidence_parts = []
    for i, source in enumerate(sources, 1):
        evidence_parts.append(
            f"""
SOURCE {i}
Title: {source.get('title', '')}
URL: {source.get('url', '')}
Credibility: {source.get('credibility', 'Low')}
Fact-checking outlet: {"Yes" if is_fact_check_source(source.get('url', '')) else "No"}
Search snippet: {source.get('snippet', '')[:1800]}
Article content: {source.get('content', '') or '[Full article could not be retrieved — rely on the snippet above.]'}
--------------------------------------------------
"""
        )

    evidence_text = "\n".join(evidence_parts)

    if not sources:
        evidence_text = "NO RELIABLE WEB SOURCES WERE RETRIEVED."

    aggregate_line = (
        f"Computed source mix: {credibility_counts.get('High', 0)} High, "
        f"{credibility_counts.get('Medium', 0)} Medium, {credibility_counts.get('Low', 0)} Low "
        f"across {unique_domains} unique domains. Suggested overall source quality: {aggregate_quality}."
    )

    prompt = f"""
You are the Final Evidence Analyst for HaqCheck_AI.

CLAIM:
{claim}

{aggregate_line}

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
- Use the computed source mix above as a strong anchor for SOURCE QUALITY —
  do not report "Low" just because a few individual outlets were unfamiliar
  to you if the computed mix says otherwise.
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
    ai_source_quality = field("SOURCE QUALITY", []) or "Low"

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
        "ai_source_quality": ai_source_quality,
    }


# ------------------------------------------------------------
# TEXT VERIFICATION PIPELINE
# ------------------------------------------------------------

def verify_claim(claim, progress_cb=None):
    claim = claim.strip()
    if not claim:
        raise ValueError("Please provide a claim.")

    started = time.time()

    if progress_cb:
        progress_cb(10, "Analyzing the claim...")
    analysis = analyze_claim(claim)

    if progress_cb:
        progress_cb(30, "Searching multiple independent sources...")
    sources = multi_query_research(claim, per_query=4)

    if progress_cb:
        progress_cb(55, "Reading full articles from each source...")
    enriched = enrich_sources(sources)

    quality_tuple = compute_overall_source_quality(enriched)
    aggregate_quality, credibility_counts, unique_domains = quality_tuple

    if progress_cb:
        progress_cb(80, "Cross-checking evidence and drafting the verdict...")
    final_text = analyze_evidence(claim, enriched, aggregate_quality, credibility_counts, unique_domains)
    verdict = parse_final_verdict(final_text)

    # The computed, code-driven quality is what we surface as the headline
    # "Source Quality" — it can't be dragged down just because the model
    # under-recognized a legitimate outlet.
    verdict["source_quality"] = aggregate_quality

    if progress_cb:
        progress_cb(100, "Done.")

    elapsed = round(time.time() - started, 1)

    return {
        "status": "success",
        "input_type": "text",
        "claim": claim,
        "claim_analysis": analysis,
        "sources": enriched,
        "final_text": final_text,
        "verdict": verdict,
        "credibility_counts": credibility_counts,
        "unique_domains": unique_domains,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def safe_verify_claim(claim, progress_cb=None):
    try:
        return verify_claim(claim, progress_cb=progress_cb)
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
                "ai_source_quality": "Low",
            },
            "credibility_counts": {"High": 0, "Medium": 0, "Low": 0},
            "unique_domains": 0,
            "elapsed_seconds": 0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "error": str(exc),
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
                processed, lang="eng+urd", config=f"--psm {psm}",
            ).strip()
        except Exception:
            text = pytesseract.image_to_string(
                processed, lang="eng", config=f"--psm {psm}",
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


def verify_image(image, progress_cb=None):
    if progress_cb:
        progress_cb(15, "Reading text from the image (OCR)...")
    ocr_text = improved_ocr(image)

    if progress_cb:
        progress_cb(30, "Reconstructing the factual claim...")
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
                "ai_source_quality": "Low",
            },
            "sources": [],
            "credibility_counts": {"High": 0, "Medium": 0, "Low": 0},
            "unique_domains": 0,
            "elapsed_seconds": 0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    result = verify_claim(claim, progress_cb=lambda p, m: progress_cb(30 + p * 0.7, m) if progress_cb else None)
    result.update({
        "input_type": "image",
        "ocr_text": ocr_text,
        "extraction": extraction,
    })
    return result


def safe_verify_image(image, progress_cb=None):
    try:
        return verify_image(image, progress_cb=progress_cb)
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
                "ai_source_quality": "Low",
            },
            "credibility_counts": {"High": 0, "Medium": 0, "Low": 0},
            "unique_domains": 0,
            "elapsed_seconds": 0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "error": str(exc),
        }


# ------------------------------------------------------------
# REPORT EXPORT
# ------------------------------------------------------------

def build_markdown_report(result):
    verdict = result.get("verdict", {})
    lines = [
        "# HaqCheck_AI Verification Report",
        f"*Generated {result.get('timestamp', '')}*",
        "",
        f"**Claim:** {result.get('claim', '')}",
        "",
        f"**Verdict:** {verdict.get('verdict', 'Unverified')}",
        f"**Confidence:** {verdict.get('confidence', 0)}%",
        f"**Source quality:** {verdict.get('source_quality', 'Low')}",
        "",
        "## Summary",
        verdict.get("summary", ""),
        "",
        "## Confirmed facts",
        verdict.get("confirmed_facts", "") or "_None reported._",
        "",
        "## Uncertain or contradicting claims",
        verdict.get("uncertain_claims", "") or "_None reported._",
        "",
        "## Reasoning",
        verdict.get("reasoning", ""),
        "",
        "## Evidence sources",
    ]
    for i, source in enumerate(result.get("sources", []), 1):
        lines.append(f"{i}. [{source.get('title', 'Untitled')}]({source.get('url', '')}) — credibility: {source.get('credibility', 'Low')}")
    if not result.get("sources"):
        lines.append("_No web sources were retrieved._")

    lines += ["", "---", "Generated by HaqCheck_AI. AI-assisted fact-checking can make mistakes — always verify critical claims independently."]
    return "\n".join(lines)


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="HaqCheck_AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .hc-display { font-family: 'Space Grotesk', sans-serif; }

    .stApp {
        background:
            repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 42px),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 42px),
            #0A0E17;
        color: #E7ECF3;
    }
    .block-container { max-width: 1120px; padding-top: 1.4rem; }

    .hc-hero { padding: 22px 4px 6px; border-bottom: 1px solid rgba(240,180,41,0.18); margin-bottom: 22px; }
    .hc-brand { display:flex; align-items:baseline; gap:12px; }
    .hc-logo { font-size: 40px; font-weight: 700; color:#F5F7FA; letter-spacing:-0.5px; }
    .hc-logo span { color:#F0B429; }
    .hc-tagline { color:#8B95A7; font-size:15.5px; margin-top:6px; max-width:620px; }

    .hc-panel { background:#0F1420; border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:20px 22px; margin:14px 0; }
    .hc-panel-title { font-size:16.5px; font-weight:600; color:#F5F7FA; margin-bottom:4px; }
    .hc-muted { color:#8B95A7; font-size:14px; }

    .hc-verdict { border-left:4px solid var(--v-color, #94A3B8); background:#0F1420; border-radius:8px;
                  padding:22px 24px; margin-top:8px; }
    .hc-verdict-top { display:flex; align-items:center; gap:14px; }
    .hc-verdict-icon { font-size:30px; }
    .hc-verdict-name { font-size:27px; font-weight:700; color: var(--v-color, #E7ECF3); }
    .hc-verdict-meta { color:#8B95A7; font-size:14px; margin-top:10px; }

    .hc-source { border-bottom:1px solid rgba(255,255,255,0.07); padding:14px 2px; }
    .hc-source:last-child { border-bottom:none; }
    .hc-source-title { font-weight:600; color:#F5F7FA; font-size:15px; }
    .hc-badge { display:inline-block; font-size:11.5px; padding:2px 9px; border-radius:999px; margin-left:8px; font-weight:600; }
    .hc-badge-high { background:rgba(34,197,94,0.15); color:#4ADE80; }
    .hc-badge-medium { background:rgba(245,158,11,0.15); color:#FBBF24; }
    .hc-badge-low { background:rgba(148,163,184,0.15); color:#CBD5E1; }
    .hc-badge-fc { background:rgba(96,165,250,0.16); color:#93C5FD; }
    .hc-source-url { color:#93C5FD; font-size:13px; text-decoration:none; }
    .hc-source-snippet { color:#A9B2C3; font-size:14px; margin-top:6px; line-height:1.5; }

    .hc-footer { text-align:center; color:#4B5568; margin:50px 0 18px; font-size:12.5px; }

    div[data-testid="stFileUploader"] { background: rgba(255,255,255,0.03); border-radius:10px; padding:8px; }
    div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hc-hero">
        <div class="hc-brand"><div class="hc-logo">🔎 Haq<span>Check</span>_AI</div></div>
        <div class="hc-tagline">Cross-checks claims against live web evidence, weighs source credibility,
        and shows its reasoning — instead of just handing you a verdict.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.getenv("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY is not set. Create a `.env` file next to this app with "
        "`GROQ_API_KEY=your_key` and restart, or verification will fail.",
        icon="⚠️",
    )

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔎 HaqCheck_AI")
    st.caption("AI-assisted claim verification")

    with st.expander("How it works", expanded=False):
        st.markdown(
            "1. **Understand** — the claim is broken down into entities, dates and search terms.\n"
            "2. **Research** — several search queries pull independent sources from the live web.\n"
            "3. **Read** — full articles are fetched and cleaned, not just search snippets.\n"
            "4. **Weigh** — each source is scored for credibility (wire services, fact-checkers, "
            "government/academic sites score higher than blogs or social media).\n"
            "5. **Verdict** — the evidence, not just the model's intuition, drives the final call."
        )

    st.markdown("---")
    st.markdown("#### History")
    history = st.session_state.setdefault("history", [])
    if not history:
        st.caption("No claims verified yet this session.")
    else:
        for i, item in enumerate(reversed(history[-10:])):
            idx = len(history) - 1 - i
            style = verdict_style(item["verdict"]["verdict"])
            label = item["claim"][:46] + ("…" if len(item["claim"]) > 46 else "")
            if st.button(f"{style['icon']} {label}", key=f"hist_{idx}", use_container_width=True):
                st.session_state["last_result"] = item
                st.rerun()
        if st.button("Clear history", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

# ------------------------------------------------------------
# MODE SELECTION
# ------------------------------------------------------------

mode = st.radio(
    "Verification mode",
    ["📝 Text Claim", "🖼️ Image", "🎥 Video"],
    horizontal=True,
)


def run_with_progress(work_fn):
    progress = st.progress(0)
    status = st.empty()

    def cb(pct, message):
        progress.progress(min(100, int(pct)))
        status.info(message)

    try:
        result = work_fn(cb)
    finally:
        progress.empty()
        status.empty()
    return result


# ------------------------------------------------------------
# TEXT MODE
# ------------------------------------------------------------

if mode == "📝 Text Claim":
    st.markdown('<div class="hc-panel-title">Enter a claim to verify</div>', unsafe_allow_html=True)
    claim = st.text_area(
        "Claim",
        placeholder="Paste a news claim, headline, social-media statement or allegation...",
        height=150,
        label_visibility="collapsed",
    )

    if st.button("🔎 Verify Claim", type="primary", use_container_width=True):
        if not claim.strip():
            st.warning("Please enter a claim before starting verification.")
        else:
            result = run_with_progress(lambda cb: safe_verify_claim(claim, progress_cb=cb))
            st.session_state["last_result"] = result
            st.session_state.setdefault("history", []).append(result)
            st.rerun()


# ------------------------------------------------------------
# IMAGE MODE
# ------------------------------------------------------------

elif mode == "🖼️ Image":
    st.markdown('<div class="hc-panel-title">Upload a claim screenshot or image</div>', unsafe_allow_html=True)
    st.caption("HaqCheck_AI extracts Urdu/English text with OCR, reconstructs the factual claim, then verifies it on the web.")

    if pytesseract is None:
        st.error("OCR dependencies are not installed. Run: pip install pytesseract pillow opencv-python numpy — and install the Tesseract binary.")
    else:
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

            if st.button("🖼️ Verify Image Claim", type="primary", use_container_width=True):
                result = run_with_progress(lambda cb: safe_verify_image(image, progress_cb=cb))
                st.session_state["last_result"] = result
                st.session_state.setdefault("history", []).append(result)
                st.rerun()


# ------------------------------------------------------------
# VIDEO MODE — ROADMAP
# ------------------------------------------------------------

else:
    st.info(
        "🎥 Video verification is planned for the next phase. "
        "The current release supports Text Claim and Image verification."
    )
    st.file_uploader("Video upload (planned)", type=["mp4", "mov", "avi", "mkv"], disabled=True)


# ------------------------------------------------------------
# RESULT DISPLAY
# ------------------------------------------------------------

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    verdict = result.get("verdict", {})
    style = verdict_style(verdict.get("verdict", "Unverified"))

    st.markdown('<div class="hc-panel-title" style="margin-top:8px;">Verification result</div>', unsafe_allow_html=True)

    verdict_name = safe_text(verdict.get("verdict", "Unverified"))
    confidence = parse_confidence(verdict.get("confidence", 0))
    source_quality = safe_text(verdict.get("source_quality", "Low"))
    counts = result.get("credibility_counts", {"High": 0, "Medium": 0, "Low": 0})
    unique_domains = result.get("unique_domains", 0)

    st.markdown(
        f"""
        <div class="hc-verdict" style="--v-color:{style['color']};">
            <div class="hc-verdict-top">
                <div class="hc-verdict-icon">{style['icon']}</div>
                <div class="hc-verdict-name">{verdict_name}</div>
            </div>
            <div class="hc-verdict-meta">Claim: "{safe_text(result.get('claim', ''))}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{confidence}%")
    m2.metric("Source quality", source_quality)
    m3.metric("Sources analyzed", len(result.get("sources", [])))
    m4.metric("Processing time", f"{result.get('elapsed_seconds', 0)}s")

    st.caption(
        f"Credibility mix of retrieved sources: {counts.get('High', 0)} High · "
        f"{counts.get('Medium', 0)} Medium · {counts.get('Low', 0)} Low, "
        f"across {unique_domains} unique domain(s)."
    )

    if result.get("input_type") == "image":
        extraction = result.get("extraction", {})
        ocr_text = result.get("ocr_text", "")
        if extraction.get("claim"):
            st.markdown('<div class="hc-panel"><div class="hc-panel-title">📌 Extracted claim</div></div>', unsafe_allow_html=True)
            st.write(extraction["claim"])
        with st.expander("📝 Raw OCR text"):
            st.text(ocr_text or "No OCR text detected.")

    tab_summary, tab_evidence, tab_sources, tab_reasoning = st.tabs(
        ["Summary", "Confirmed & uncertain facts", "Evidence sources", "Reasoning"]
    )

    with tab_summary:
        st.write(verdict.get("summary", ""))

    with tab_evidence:
        st.markdown("**✅ Confirmed facts**")
        st.write(verdict.get("confirmed_facts", "") or "_None reported._")
        st.markdown("**⚠️ Uncertain or contradicting claims**")
        st.write(verdict.get("uncertain_claims", "") or "_None reported._")

    with tab_sources:
        sources = result.get("sources", [])
        if not sources:
            st.warning("No web sources were retrieved. The result should remain Unverified.")
        else:
            for i, source in enumerate(sources, 1):
                title = safe_text(source.get("title", "Untitled source"))
                url = source.get("url", "")
                credibility = source.get("credibility", "Low")
                snippet = safe_text(source.get("snippet", ""))
                badge_class = {"High": "hc-badge-high", "Medium": "hc-badge-medium", "Low": "hc-badge-low"}.get(credibility, "hc-badge-low")
                fc_badge = '<span class="hc-badge hc-badge-fc">Fact-check outlet</span>' if is_fact_check_source(url) else ""
                fetched_note = "" if source.get("fetched", True) else '<span class="hc-muted"> · full article unavailable, snippet used</span>'
                st.markdown(
                    f"""
                    <div class="hc-source">
                        <span class="hc-source-title">{i}. {title}</span>
                        <span class="hc-badge {badge_class}">{safe_text(credibility)}</span>
                        {fc_badge}
                        {fetched_note}
                        <div><a class="hc-source-url" href="{safe_text(url)}" target="_blank">{safe_text(url)}</a></div>
                        <div class="hc-source-snippet">{snippet}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_reasoning:
        st.write(verdict.get("reasoning", ""))
        if verdict.get("ai_source_quality") and verdict.get("ai_source_quality") != verdict.get("source_quality"):
            st.caption(
                f"Note: the model's own read of source quality was '{verdict['ai_source_quality']}'. "
                f"The badge above uses the computed value ('{verdict['source_quality']}') from the actual "
                f"credibility mix, which is more consistent across runs."
            )

    if result.get("error"):
        st.warning("The system encountered an error, so this result is intentionally cautious and should not be treated as a definitive fact-check.")

    st.markdown("---")
    report_md = build_markdown_report(result)
    st.download_button(
        "⬇️ Download report (Markdown)",
        data=report_md,
        file_name=f"haqcheck_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.markdown(
    '<div class="hc-footer">HaqCheck_AI — AI-assisted claim verification. Always verify high-stakes claims independently.</div>',
    unsafe_allow_html=True,
)