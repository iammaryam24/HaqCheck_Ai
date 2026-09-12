# 🔎 HaqCheck_AI

### Verify Claims. Understand Context. Find the Truth.

**HaqCheck_AI** is an AI-powered claim verification and information intelligence application built with Streamlit. It helps users investigate textual and image-based claims by combining AI reasoning, web research, source credibility assessment, webpage evidence extraction, and cautious cross-verification.

---

## 🚀 Features

### 📝 Text Claim Verification

Users can enter a news claim, headline, social-media statement, or allegation and receive an evidence-based assessment.

The system:

* Analyzes the claim
* Identifies key entities, dates, numbers, and claim type
* Searches multiple web queries
* Retrieves relevant sources
* Extracts webpage content
* Evaluates source credibility
* Uses AI to generate an evidence-based verdict

### 🖼️ Image Claim Verification

Users can upload screenshots or images containing factual claims.

The application:

* Extracts text using OCR
* Supports English and attempts Urdu text extraction
* Identifies the main factual claim
* Sends the extracted claim through the web verification pipeline
* Displays the final verification result and evidence

### 🌐 Source Credibility Assessment

Sources are classified into:

* **High**
* **Medium**
* **Low**

The system gives preference to recognized news organizations, government sources, educational domains, and international organizations.

### ⚖️ Evidence-Based Verdicts

HaqCheck_AI can return:

* **Likely True**
* **Likely False**
* **Misleading**
* **Unverified**

The system also provides:

* Confidence score
* Summary
* Confirmed facts
* Uncertain or contradicting claims
* Reasoning
* Source quality
* Retrieved evidence sources

---

## 🧠 How It Works

```text
User Claim
    ↓
Claim Analysis
    ↓
Web Search
    ↓
Source Collection
    ↓
Webpage Content Extraction
    ↓
Source Credibility Assessment
    ↓
AI Evidence Analysis
    ↓
Verdict + Confidence
    ↓
Evidence Sources
```

For image input:

```text
Image
  ↓
OCR
  ↓
Claim Extraction
  ↓
Claim Verification Pipeline
  ↓
Evidence-Based Verdict
```

---

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Groq API**
* **Large Language Model (LLM)**
* **DuckDuckGo Search**
* **Requests**
* **BeautifulSoup**
* **OpenCV**
* **NumPy**
* **Pillow**
* **Tesseract OCR**
* **python-dotenv**

---

## 📁 Project Structure

```text
HaqCheck_AI/
│
├── app.py
├── requirements.txt
├── README.md
└── .env
```

> The `.env` file contains the API key and should never be uploaded publicly to GitHub.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/HaqCheck_AI.git
cd HaqCheck_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit the `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
.venv/
__pycache__/
```

---

## ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔍 Example Workflow

### Text Verification

1. Open HaqCheck_AI.
2. Select **Text Claim**.
3. Enter a factual claim.
4. Click **Verify Claim**.
5. The system analyzes and researches the claim.
6. Review the verdict, confidence, reasoning, and evidence sources.

### Image Verification

1. Select **Image**.
2. Upload a screenshot containing a claim.
3. Click **Verify Image Claim**.
4. OCR extracts the text.
5. AI identifies the main factual claim.
6. The claim is researched and verified.
7. Review the extracted claim and evidence.

---

## 📊 Verification Philosophy

HaqCheck_AI is designed to avoid treating language-model output as unquestionable truth.

The system follows several principles:

* Do not invent facts or sources.
* Do not treat allegations as proven facts.
* Do not automatically trust OCR output.
* Prefer multiple independent credible sources.
* Use **Misleading** when only part of a claim is supported.
* Use **Unverified** when reliable evidence is insufficient.
* Confidence should reflect evidence quality rather than language-model certainty.

---

## 🔐 Security

API credentials should be stored in environment variables and never committed to the repository.

For deployment platforms, configure secrets through the platform's secret-management system rather than exposing them in source code.

---

## 🚧 Current Limitations

* Web results depend on search-engine availability.
* Some webpages may block automated content extraction.
* OCR accuracy depends on image quality and text clarity.
* Video verification is currently planned for a future phase.
* The current production interface supports **Text Claim** and **Image** verification.

---

## 🔮 Future Improvements

Potential future enhancements include:

* Full video claim verification
* Improved multilingual OCR
* More advanced source ranking
* Fact-checking database integration
* Claim history and analytics
* User accounts
* Improved misinformation detection
* More structured evidence comparison
* Advanced multilingual support

---

## 🎯 Project Goal

The goal of HaqCheck_AI is to provide users with a practical AI-assisted tool for investigating online claims, understanding evidence, and making more informed judgments about potentially misleading information.

---

## 👩‍💻 Project

**HaqCheck_AI — AI-Powered Claim Verification & Information Intelligence**

Built with Python, Streamlit, AI reasoning, web research, and OCR.
