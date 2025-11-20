#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReStart MVP v4
- Upload PDF OR paste text
- Detect simple skills per sector (rule-based)
- Detect main language of CV
- Allow counselor to download a short text report
- NEW: AI (LLM) analysis: summary, skills, suggestions, job ideas & search queries
"""

import re
import streamlit as st
import pdfplumber
from langdetect import detect, LangDetectException

from openai import OpenAI  # NEW: LLM client

# ----------------- Streamlit & OpenAI setup -----------------
st.set_page_config(page_title="ReStart – Skill Extractor", page_icon="🟣")

st.title("ReStart – Skill Extractor (MVP v4)")
st.write("Upload a refugee CV (or paste text) to get a rough idea of skills and job sectors.")

# OpenAI client – reads key from Streamlit secrets
# Make sure you set OPENAI_API_KEY in Streamlit → Settings → Secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def analyze_cv_with_llm(cv_text: str) -> str:
    """
    Use a large language model to:
    - Summarize the CV
    - Extract hard & soft skills
    - Suggest improvements
    - Propose job titles
    - Propose search queries for job platforms
    Returns markdown text.
    """
    # Avoid sending extremely long texts
    cv_text_short = cv_text[:8000]

    prompt = f"""
You are helping a job coach working with a refugee or newcomer in Europe.

You receive the FULL CV text of this person (might be messy, with mixed languages or bad formatting).

CV TEXT:
\"\"\"{cv_text_short}\"\"\"

Please analyse it and return the following sections in clear markdown:

### 1. Short professional summary
5–7 sentences that describe:
- background & profile
- main experience areas
- key strengths

### 2. Hard skills (bullet list)
Technical skills, tools, software, languages, certifications, machinery, etc.

### 3. Soft skills & strengths (bullet list)
Interpersonal skills, character strengths, work style.

### 4. Suggestions to improve the CV (max 8 bullets)
Focus on what the candidate or counselor can change:
layout, clarity, tailoring to jobs in Europe, missing info, structure, etc.

### 5. Example job titles in Europe
Give 5–7 realistic job titles in English that could fit this profile
(for The Netherlands or similar labour markets).
Example: "Warehouse operative", "Cleaner", "Junior data analyst", etc.

### 6. Example job search queries
Give 3–5 concrete search strings they can use on platforms like LinkedIn or Indeed.
Example:
- "warehouse worker English Amsterdam"
- "cleaner part-time Utrecht"
- "junior data analyst English Rotterdam"

Write everything in clear English, but keep important non-English terms (e.g. diploma names, job titles) where useful.
    """

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert career coach and CV analyst helping refugees find work in Europe.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    content = completion.choices[0].message.content
    return content


# ----------------- File upload & text input -----------------
uploaded_file = st.file_uploader("Upload CV file (PDF only)", type=["pdf"])
cv_text = ""

if uploaded_file is not None:
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                cv_text += page.extract_text() or ""
        st.success("PDF text extracted successfully! 🎉")
    except Exception as e:
        st.error(f"Could not extract text from PDF. Error: {e}")

st.subheader("Or paste CV text below:")
paste_text = st.text_area("Paste text here:", height=200)

if paste_text.strip():
    cv_text = paste_text

st.caption("Example content: work experience, jobs, tasks, skills, etc.")

# ----------------- Skills dictionary (rule-based) -----------------
SKILLS = {
    "logistics": [
        "warehouse",
        "forklift",
        "picking",
        "packing",
        "inventory",
        "logistics",
        "order picker",
        "stockroom",
    ],
    "hospitality": [
        "cleaning",
        "housekeeping",
        "kitchen",
        "dishwasher",
        "hotel",
        "restaurant",
        "room service",
    ],
    "retail": [
        "cashier",
        "shop",
        "store",
        "customer service",
        "sales",
        "retail",
        "cash register",
    ],
    "tech": [
        "python",
        "java",
        "it support",
        "helpdesk",
        "software",
        "computer",
        "technical support",
    ],
}

# ----------------- Rule-based analysis button -----------------
if st.button("Analyze CV (rule-based)"):
    if not cv_text.strip():
        st.warning("Please upload a CV or paste some text first.")
    else:
        st.success("CV received. Here is a first rough analysis 👇")

        text_lower = cv_text.lower()
        sector_matches = {sector: [] for sector in SKILLS}

        for sector, keywords in SKILLS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    sector_matches[sector].append(kw)

        # ---------- Language detection ----------
        st.subheader("Detected language (rough):")
        try:
            lang_code = detect(cv_text)
            st.write(f"Main language detected: **{lang_code}**")
        except LangDetectException:
            lang_code = None
            st.write("Could not reliably detect language (text too short or noisy).")

        # ---------- Show detected skills ----------
        st.subheader("Detected skills by sector (very rough):")
        any_match = False
        for sector, kws in sector_matches.items():
            if kws:
                any_match = True
                st.write(f"**{sector.capitalize()}** → {', '.join(sorted(set(kws)))}")

        if not any_match:
            st.write("No sector matches yet. This is only an MVP — we’ll improve it!")

        # ---------- Suggested jobs ----------
        st.subheader("Suggested job directions (rule-based):")

        suggestions = []
        if sector_matches["logistics"]:
            suggestions.append(
                "- 📦 Logistics & Warehouse: warehouse assistant, order picker, stockroom worker"
            )
        if sector_matches["hospitality"]:
            suggestions.append(
                "- 🧹 Hospitality & Cleaning: cleaner, kitchen assistant, hotel staff"
            )
        if sector_matches["retail"]:
            suggestions.append(
                "- 🛒 Retail & Customer Service: shop assistant, cashier, store support"
            )
        if sector_matches["tech"]:
            suggestions.append(
                "- 💻 Tech & Digital: IT support, helpdesk, junior IT roles"
            )

        if suggestions:
            for s in suggestions:
                st.write(s)
        else:
            st.write("No job suggestions yet – once the model improves, this will be richer.")

        # ---------- Simple counselor report ----------
        st.subheader("Downloadable counselor summary (rule-based)")

        report_lines = []
        report_lines.append("ReStart – Candidate Summary")
        report_lines.append("--------------------------------")
        if lang_code is not None:
            report_lines.append(f"Detected language: {lang_code}")
        else:
            report_lines.append("Detected language: unknown")

        report_lines.append("\nDetected skills by sector:")
        if any_match:
            for sector, kws in sector_matches.items():
                if kws:
                    report_lines.append(
                        f"- {sector.capitalize()}: {', '.join(sorted(set(kws)))}"
                    )
        else:
            report_lines.append("- No obvious skills detected (MVP limitations).")

        report_lines.append("\nSuggested job directions:")
        if suggestions:
            for s in suggestions:
                report_lines.append(s)
        else:
            report_lines.append("- None yet.")

        report_text = "\n".join(report_lines)

        st.download_button(
            label="⬇️ Download candidate summary (text)",
            data=report_text,
            file_name="restart_candidate_summary.txt",
            mime="text/plain",
        )

# ----------------- AI (LLM) analysis section -----------------
st.markdown("---")
st.subheader("🔮 AI-powered CV analysis (LLM)")

if not cv_text.strip():
    st.caption("Upload or paste a CV first to enable AI analysis.")
else:
    if st.button("Run AI analysis (summary, skills, job ideas)", key="ai_button"):
        with st.spinner("Asking the AI model to analyse this CV..."):
            try:
                ai_result = analyze_cv_with_llm(cv_text)
                st.success("AI analysis ready ✅")
                st.markdown(ai_result)
            except Exception as e:
                st.error(f"AI analysis failed. Please try again or contact the team. Error: {e}")

st.info(
    "MVP v4: PDF upload, basic language detection, simple report download + AI-powered analysis. "
    "Next: better NLP + ESCO mapping."
)
