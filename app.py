import os
import re

import streamlit as st
import pymupdf
import pandas as pd
import plotly.express as px

from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. BLACK + PURPLE DARK THEME
# ==================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background:
        radial-gradient(
            circle at top right,
            #241044 0%,
            #10091c 38%,
            #08080e 100%
        );
    color: #f5f3ff;
}

/* Main layout */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Main headings */

h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

p, li, label {
    color: #d8d3e8;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #170d28,
        #0b0912
    );
    border-right: 1px solid #392052;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] p {
    color: #ffffff !important;
}

/* Sidebar navigation */

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #171121;
    border: 1px solid #30203f;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 7px;
}

/* Main title */

.hero-title {
    font-size: 46px;
    font-weight: 900;
    line-height: 1.2;
    color: #ffffff;
    margin-bottom: 12px;
}

.hero-title span {
    color: #bd8cff;
}

.hero-subtitle {
    font-size: 17px;
    color: #b9b0cc;
    margin-bottom: 25px;
}

/* Hero banner */

.hero-card {
    background: linear-gradient(
        135deg,
        #25113c,
        #171025,
        #10101b
    );
    border: 1px solid #65369b;
    border-radius: 22px;
    padding: 35px;
    margin-bottom: 25px;
    box-shadow: 0 0 35px rgba(139, 92, 246, 0.12);
}

/* Feature cards */

.feature-card {
    background: linear-gradient(
        145deg,
        #1d162b,
        #11101a
    );
    border: 1px solid #352345;
    border-radius: 18px;
    padding: 23px;
    min-height: 175px;
    margin-bottom: 15px;
    transition: 0.2s ease;
}

.feature-card:hover {
    border-color: #a855f7;
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.15);
}

.feature-icon {
    font-size: 32px;
    margin-bottom: 10px;
}

.feature-title {
    color: #ffffff;
    font-size: 19px;
    font-weight: 800;
    margin-bottom: 8px;
}

.feature-description {
    color: #b9b0cc;
    font-size: 14px;
    line-height: 1.6;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #21152f,
        #12101b
    );
    border: 1px solid #4b2b67;
    border-radius: 17px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
}

[data-testid="stMetricLabel"] {
    color: #c4b5fd !important;
}

[data-testid="stMetricValue"] {
    color: #d8b4fe !important;
    font-weight: 800;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        100deg,
        #7c3aed,
        #a855f7
    );
    color: #ffffff;
    border: 1px solid #a855f7;
    border-radius: 11px;
    padding: 11px 18px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        100deg,
        #6d28d9,
        #9333ea
    );
    color: #ffffff;
    border-color: #c084fc;
    box-shadow: 0 0 18px rgba(168, 85, 247, 0.3);
}

/* Download button */

.stDownloadButton > button {
    background: linear-gradient(
        100deg,
        #6d28d9,
        #9333ea
    );
    color: #ffffff;
    border: none;
    border-radius: 11px;
    font-weight: 700;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background: #15111f;
    border: 1px dashed #8b5cf6;
    border-radius: 15px;
    padding: 16px;
}

/* Text area */

.stTextArea textarea {
    background: #15111f !important;
    color: #ffffff !important;
    border: 1px solid #49305f !important;
    border-radius: 12px !important;
}

.stTextArea textarea:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 1px #a855f7 !important;
}

/* Text input */

.stTextInput input {
    background: #15111f !important;
    color: #ffffff !important;
    border: 1px solid #49305f !important;
    border-radius: 10px !important;
}

/* Select boxes */

[data-baseweb="select"] > div {
    background: #15111f;
    color: #ffffff;
    border-color: #49305f;
}

/* Dividers */

hr {
    border-color: #352345;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Expander */

[data-testid="stExpander"] {
    background: #15111f;
    border: 1px solid #352345;
    border-radius: 12px;
}

/* Progress bar */

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #c084fc
    );
}

/* Links */

a {
    color: #c084fc !important;
}

/* Footer */

.custom-footer {
    color: #8e849f;
    text-align: center;
    padding: 25px 0 5px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. GEMINI API CONFIGURATION
# ==================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        client = None


# ==================================================
# 4. SKILLS DATABASE
# ==================================================

SKILLS = [
    "python", "java", "c++", "c", "dart", "flutter",
    "html", "css", "javascript", "sql", "mysql",
    "firebase", "machine learning", "deep learning",
    "data analysis", "pandas", "numpy", "power bi",
    "excel", "git", "github", "communication",
    "teamwork", "problem solving", "flask", "django",
    "streamlit", "react", "node.js", "mongodb",
    "data structures", "oop", "aws", "cloud computing",
    "artificial intelligence", "scikit-learn",
    "tensorflow", "pytorch", "rest api", "api",
    "docker", "kubernetes", "linux", "statistics",
    "data visualization", "project management"
]


# ==================================================
# 5. PDF TEXT EXTRACTION
# ==================================================

def extract_text(pdf_file):

    text = ""

    try:
        with pymupdf.open(
            stream=pdf_file.getvalue(),
            filetype="pdf"
        ) as doc:

            for page in doc:
                text += page.get_text() + "\n"

    except Exception as e:
        st.error(f"PDF reading error: {e}")
        return ""

    return text.strip()


# ==================================================
# 6. SKILLS DETECTION
# ==================================================

def find_skills(text):

    text = text.lower()
    found = []

    for skill in SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found))


# ==================================================
# 7. TEXT SIMILARITY
# ==================================================

def calculate_match(resume_text, job_text):

    if not resume_text.strip() or not job_text.strip():
        return 0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            [resume_text, job_text]
        )

        score = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(score * 100)

    except ValueError:
        return 0


# ==================================================
# 8. GEMINI AI ANALYSIS
# ==================================================

def analyze_resume_with_ai(resume_text, job_description):

    if client is None:
        return (
            "Gemini AI is not configured. "
            "Please check your .env file and API key."
        )

    resume_input = resume_text[:18000]
    job_input = job_description[:10000]

    prompt = f"""
You are an expert resume analyst and career advisor.

Analyze this resume against the job description.

RESUME:
{resume_input}

JOB DESCRIPTION:
{job_input}

Write a detailed, professional report using these sections:

1. RESUME SUMMARY
2. RELEVANT SKILLS
3. POSSIBLE SKILL GAPS
4. RELEVANT EXPERIENCE AND PROJECTS
5. RESUME STRENGTHS
6. AREAS FOR IMPROVEMENT
7. PERSONALIZED SUGGESTIONS
8. OVERALL SUITABILITY

Explain findings clearly and give practical examples of
how the candidate could improve the resume.

IMPORTANT:
- Never invent skills, projects or experience.
- Distinguish missing information from confirmed skill gaps.
- Do not guarantee a job or interview.
- Do not treat text similarity as hiring probability.
- Keep the response detailed but easy to understand.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return "Gemini AI returned an empty response."

    except Exception as e:

        return (
            "Gemini AI request failed. "
            "Check API key, internet, model availability "
            "and API quota. Technical details: "
            + str(e)
        )


# ==================================================
# 9. SESSION STATE
# ==================================================

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None


# ==================================================
# 10. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;padding:12px 0 22px 0;">
            <div style="font-size:45px;">📄</div>
            <h2 style="margin:5px 0;color:#d8b4fe;">
                ResumeAI
            </h2>
            <p style="color:#a99abf;font-size:13px;">
                Smart Career Assistant
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "🔍 Analyze Resume",
            "ℹ️ About Project"
        ],
        label_visibility="visible"
    )

    st.divider()

    if client is not None:
        st.success("🟢 Gemini AI Connected")
    else:
        st.warning("🟡 Gemini AI Not Configured")

    st.markdown(
        """
        <div style="padding-top:20px;color:#8e849f;font-size:12px;">
            <b>AI RESUME ANALYZER</b><br>
            Built with Python & Gemini AI
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 11. HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero-card">
            <div style="font-size:14px;color:#c084fc;
                        font-weight:700;letter-spacing:2px;">
                AI-POWERED CAREER TOOL
            </div>

            <div class="hero-title">
                Your Career.<br>
                <span>Your Next Opportunity.</span>
            </div>

            <div class="hero-subtitle">
                Analyze your resume, discover skill gaps,
                and get personalized career recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if client is not None:
        st.success("Gemini AI is configured and ready to analyze resumes.")
    else:
        st.warning(
            "Gemini AI is not configured. Check your .env file "
            "if you want to use AI analysis."
        )

    st.markdown("## ✨ Everything you need to improve your resume")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <div class="feature-title">Resume Analysis</div>
                <div class="feature-description">
                    Upload your resume PDF and extract
                    important skills and information.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Job Matching</div>
                <div class="feature-description">
                    Compare resume content with a job
                    description and identify matching skills.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">Gemini AI Feedback</div>
                <div class="feature-description">
                    Get personalized suggestions for
                    improving your resume.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Visual Insights</div>
                <div class="feature-description">
                    Explore your detected skills through
                    interactive charts.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💡</div>
                <div class="feature-title">Skill Gap Insights</div>
                <div class="feature-description">
                    Discover which listed job skills were
                    not detected in your resume.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col6:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📥</div>
                <div class="feature-title">Download Report</div>
                <div class="feature-description">
                    Download your analysis and review
                    the results whenever you need them.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown("## 🚀 Ready to get started?")

    st.write(
        "Choose **Analyze Resume** from the sidebar to begin."
    )

    if st.button("🔍 Go to Resume Analysis", type="primary"):
        st.session_state.analysis_data = None
        st.session_state.ai_analysis = None
        st.rerun()

    st.markdown(
        """
        <div class="custom-footer">
            AI Resume Analyzer · Python · Streamlit · Gemini AI
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 12. ANALYZE RESUME PAGE
# ==================================================

elif page == "🔍 Analyze Resume":

    st.markdown(
        """
        <div class="hero-title">
            Resume <span>Analysis</span>
        </div>
        <div class="hero-subtitle">
            Upload your resume and compare it with your target job.
        </div>
        """,
        unsafe_allow_html=True
    )

    upload_col, description_col = st.columns([1, 1.4])

    with upload_col:

        st.markdown("### 📄 Upload Resume")

        uploaded_file = st.file_uploader(
            "Choose your resume PDF",
            type=["pdf"]
        )

        if uploaded_file:
            st.success(f"Selected: {uploaded_file.name}")

    with description_col:

        st.markdown("### 💼 Target Job")

        job_description = st.text_area(
            "Job Description",
            placeholder=(
                "Paste the complete job description here...\n\n"
                "Example: Python Developer Fresher, SQL, "
                "Flask, Django, REST APIs..."
            ),
            height=210
        )

    st.write("")

    if st.button(
        "🚀 Analyze My Resume",
        type="primary",
        width="stretch"
    ):

        if uploaded_file is None:
            st.error("Please upload your resume PDF first.")

        elif not job_description.strip():
            st.error("Please enter the target job description.")

        else:

            with st.spinner("Extracting resume information..."):

                resume_text = extract_text(uploaded_file)

                if not resume_text:

                    st.error(
                        "Could not extract text from this PDF. "
                        "Try a text-based PDF. Scanned PDFs may "
                        "require OCR."
                    )

                else:

                    resume_skills = find_skills(resume_text)
                    job_skills = find_skills(job_description)

                    matched_skills = [
                        skill for skill in job_skills
                        if skill in resume_skills
                    ]

                    missing_skills = [
                        skill for skill in job_skills
                        if skill not in resume_skills
                    ]

                    match_score = calculate_match(
                        resume_text,
                        job_description
                    )

                    st.session_state.analysis_data = {
                        "resume_text": resume_text,
                        "job_description": job_description,
                        "resume_skills": resume_skills,
                        "job_skills": job_skills,
                        "matched_skills": matched_skills,
                        "missing_skills": missing_skills,
                        "match_score": match_score,
                        "filename": uploaded_file.name
                    }

                    st.session_state.ai_analysis = None

                    st.success("Resume analysis completed!")

    data = st.session_state.analysis_data

    if data is not None:

        resume_text = data["resume_text"]
        job_description = data["job_description"]
        resume_skills = data["resume_skills"]
        job_skills = data["job_skills"]
        matched_skills = data["matched_skills"]
        missing_skills = data["missing_skills"]
        match_score = data["match_score"]
        filename = data["filename"]

        st.divider()

        st.markdown("## 📊 Your Analysis Dashboard")

        st.caption(f"Resume file: {filename}")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Text Similarity",
            f"{match_score}%"
        )

        c2.metric(
            "Resume Skills Detected",
            len(resume_skills)
        )

        c3.metric(
            "Job Skills Matched",
            f"{len(matched_skills)} / {len(job_skills)}"
        )

        st.progress(match_score / 100)

        st.caption(
            "Text similarity estimates textual overlap. "
            "It is not a validated ATS score or a hiring probability."
        )

        # ------------------------------------------
        # SKILLS MATCHING
        # ------------------------------------------

        st.divider()

        st.markdown("## 🎯 Skills Matching")

        skills_col1, skills_col2 = st.columns(2)

        with skills_col1:

            st.markdown("### ✅ Matched Skills")

            if matched_skills:

                for skill in matched_skills:
                    st.markdown(
                        f"""
                        <div style="
                            background:#14251f;
                            border:1px solid #22664b;
                            border-radius:9px;
                            padding:9px 13px;
                            margin:6px 0;
                            color:#86efac;">
                            ✓ {skill.title()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:
                st.info(
                    "No job-description skills from the "
                    "predefined list were detected in your resume."
                )

        with skills_col2:

            st.markdown("### 🔎 Skills Not Detected")

            if missing_skills:

                for skill in missing_skills:
                    st.markdown(
                        f"""
                        <div style="
                            background:#29151c;
                            border:1px solid #743044;
                            border-radius:9px;
                            padding:9px 13px;
                            margin:6px 0;
                            color:#fda4af;">
                            • {skill.title()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:
                st.success(
                    "All detected job-description skills "
                    "were found in your resume."
                )

        # ------------------------------------------
        # SKILLS CHART
        # ------------------------------------------

        st.divider()

        st.markdown("## 📈 Skills Visualization")

        if resume_skills:

            chart_data = pd.DataFrame({
                "Skill": resume_skills,
                "Count": [1] * len(resume_skills)
            })

            fig = px.bar(
                chart_data,
                x="Skill",
                y="Count",
                title="Skills Detected in Your Resume",
                color="Count",
                color_continuous_scale=[
                    "#4c1d95",
                    "#a855f7",
                    "#e9d5ff"
                ],
                template="plotly_dark"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis_title="Detected Skills",
                yaxis_title="",
                xaxis_tickangle=-40,
                coloraxis_showscale=False
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        else:

            st.info(
                "No skills from the predefined list "
                "were detected in the resume."
            )

        # ------------------------------------------
        # GEMINI AI ANALYSIS
        # ------------------------------------------

        st.divider()

        st.markdown("## 🤖 Gemini AI Career Insights")

        st.write(
            "Generate personalized feedback based on your resume "
            "and the job description."
        )

        if client is None:

            st.warning(
                "Gemini AI is not configured. "
                "Check the GEMINI_API_KEY setting in your .env file."
            )

        else:

            if st.button(
                "✨ Generate Detailed AI Analysis",
                type="primary",
                width="stretch"
            ):

                with st.spinner(
                    "Gemini AI is reviewing your resume..."
                ):

                    st.session_state.ai_analysis = (
                        analyze_resume_with_ai(
                            resume_text,
                            job_description
                        )
                    )

            if st.session_state.ai_analysis:

                st.markdown("### 📋 Personalized AI Report")

                st.markdown(
                    st.session_state.ai_analysis
                )

        # ------------------------------------------
        # IMPROVEMENT SUGGESTIONS
        # ------------------------------------------

        st.divider()

        st.markdown("## 💡 Resume Improvement Suggestions")

        suggestions = []

        if missing_skills:

            suggestions.append(
                "Review these job-related skills and include "
                "them only if you genuinely have the relevant "
                "knowledge or experience: "
                + ", ".join(missing_skills)
            )

        if len(resume_skills) < 5:

            suggestions.append(
                "Consider adding relevant technical and soft "
                "skills that are supported by your real projects "
                "and experience."
            )

        if len(resume_text.split()) < 100:

            suggestions.append(
                "The extracted resume text appears short. "
                "Check whether your education, projects, skills "
                "and experience are adequately described."
            )

        suggestions.append(
            "Use clear section headings and concise bullet points."
        )

        suggestions.append(
            "Describe genuine project achievements with measurable "
            "results wherever possible."
        )

        for index, suggestion in enumerate(suggestions, start=1):

            st.markdown(
                f"""
                <div class="feature-card"
                     style="min-height:auto;padding:15px;">
                    <span style="color:#c084fc;font-weight:800;">
                        {index:02d}
                    </span>
                    <span style="color:#e9e4f5;">
                        &nbsp; {suggestion}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ------------------------------------------
        # DOWNLOAD REPORT
        # ------------------------------------------

        st.divider()

        st.markdown("## 📥 Download Your Report")

        ai_report = (
            st.session_state.ai_analysis
            or "AI analysis has not been generated."
        )

        report = f"""
AI RESUME ANALYZER - REPORT

Resume File:
{filename}

TEXT SIMILARITY:
{match_score}%

DETECTED RESUME SKILLS:
{', '.join(resume_skills) or 'None detected'}

MATCHED JOB SKILLS:
{', '.join(matched_skills) or 'None detected'}

JOB SKILLS NOT DETECTED:
{', '.join(missing_skills) or 'None detected'}

IMPROVEMENT SUGGESTIONS:
{chr(10).join('- ' + item for item in suggestions)}

GEMINI AI ANALYSIS:
{ai_report}

IMPORTANT:
This report is an automated analysis.
Text similarity is not a validated ATS score or a hiring probability.
Verify AI-generated suggestions before using them.
"""

        st.download_button(
            "📥 Download Analysis Report (.txt)",
            data=report,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
            width="stretch"
        )


# ==================================================
# 13. ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    st.markdown(
        """
        <div class="hero-title">
            About <span>ResumeAI</span>
        </div>

        <div class="hero-subtitle">
            An AI-powered resume analysis and career assistance tool.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 🛠️ Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    tech1.metric("Language", "Python")
    tech2.metric("Web Framework", "Streamlit")
    tech3.metric("AI Engine", "Gemini")

    st.divider()

    st.markdown("## 📦 Project Modules")

    st.markdown(
        """
        <div class="feature-card" style="min-height:auto;">
            <div class="feature-title">01 · PDF Text Extraction</div>
            <div class="feature-description">
                Extracts readable text from uploaded PDF resumes.
            </div>
        </div>

        <div class="feature-card" style="min-height:auto;">
            <div class="feature-title">02 · Skills Detection</div>
            <div class="feature-description">
                Detects skills from a predefined technical
                and professional skills list.
            </div>
        </div>

        <div class="feature-card" style="min-height:auto;">
            <div class="feature-title">03 · Job Matching</div>
            <div class="feature-description">
                Compares resume text with the target job description.
            </div>
        </div>

        <div class="feature-card" style="min-height:auto;">
            <div class="feature-title">04 · Gemini AI Analysis</div>
            <div class="feature-description">
                Generates personalized feedback and improvement
                suggestions using the Gemini API.
            </div>
        </div>

        <div class="feature-card" style="min-height:auto;">
            <div class="feature-title">05 · Report Generation</div>
            <div class="feature-description">
                Creates a downloadable text report of the analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("## ⚠️ Important Limitations")

    st.info(
        "Results depend on the text extracted from the PDF and "
        "the predefined skills database. Scanned PDFs may require "
        "OCR. Text similarity is not a validated ATS score or a "
        "hiring probability. AI-generated feedback may contain "
        "errors and should be reviewed."
    )

    st.markdown(
        """
        <div class="custom-footer">
            ResumeAI · Built with Python, Streamlit and Gemini AI
        </div>
        """,
        unsafe_allow_html=True
    )