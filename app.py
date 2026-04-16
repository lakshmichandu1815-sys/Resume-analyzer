import streamlit as st
import pdfplumber
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

nlp = spacy.load("en_core_web_sm")

# ---------- UI ----------
st.set_page_config(page_title="Ultimate Resume Analyzer", layout="wide")

st.markdown("""
<style>
body {background-color: #0e1117;}
.block {
    background-color: #161b22;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Ultimate Resume Analyzer")

# ---------- Upload ----------
col1, col2 = st.columns(2)

with col1:
    resumes = st.file_uploader("📂 Upload Resumes", type=["pdf"], accept_multiple_files=True)

with col2:
    jd_text = st.text_area("🧾 Paste Job Description")

# ---------- Functions ----------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_email(text):
    match = re.findall(r"\S+@\S+", text)
    return match[0] if match else "Not found"

def extract_name(text):
    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Unknown"

def extract_skills(text):
    skills_db = [
        "python","java","sql","machine learning","deep learning","nlp",
        "data science","aws","docker","excel","power bi","html","css","javascript"
    ]
    text = text.lower()
    return [s for s in skills_db if s in text]

def similarity(resume, jd):
    tfidf = TfidfVectorizer().fit_transform([resume, jd])
    return cosine_similarity(tfidf)[0][1]

def generate_feedback(score, skills_count):
    if score < 50:
        return "Add relevant skills and improve keyword matching."
    elif score < 75:
        return "Good resume. Add projects and measurable achievements."
    else:
        return "Excellent resume. ATS optimized."

def create_pdf(text, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = [Paragraph(text, styles["Normal"])]
    doc.build(content)

# ---------- Processing ----------
if resumes and jd_text:

    results = []

    for file in resumes:
        text = extract_text(file)
        name = extract_name(text)
        email = extract_email(text)
        skills = extract_skills(text)
        score = int(similarity(text, jd_text) * 100)

        feedback = generate_feedback(score, len(skills))

        results.append({
            "Name": name,
            "Email": email,
            "Skills": ", ".join(skills),
            "Score": score,
            "Feedback": feedback
        })

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

    # ---------- Cards UI ----------
    st.subheader("🏆 Candidate Results")

    for _, row in df.iterrows():
        st.markdown(f"""
        <div class="block">
        <h3>{row['Name']} ({row['Score']}/100)</h3>
        <p><b>Email:</b> {row['Email']}</p>
        <p><b>Skills:</b> {row['Skills']}</p>
        <p><b>Feedback:</b> {row['Feedback']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Chart ----------
    st.subheader("📊 Ranking Chart")

    fig, ax = plt.subplots()
    ax.bar(df["Name"], df["Score"])
    st.pyplot(fig)

    # ---------- Download Report ----------
    st.subheader("📄 Download Report")

    report_text = df.to_string(index=False)
    create_pdf(report_text)

    with open("report.pdf", "rb") as f:
        st.download_button("Download PDF Report", f, file_name="Resume_Report.pdf")