import os
import pdfplumber
import pytesseract
import spacy
import json
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Mount static directory for favicon
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load NLP model for entity extraction
nlp = spacy.load("en_core_web_md")

# Load BERT model for job matching
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load job descriptions
with open("jobs.json", "r") as f:
    job_data = json.load(f)

# Define Tesseract OCR path (for Windows users, update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to AI-Powered Resume Analyzer!"}

# Serve the favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join("static", "favicon.ico"))

# Function to extract text from PDFs (includes OCR for scanned resumes)
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
            else:
                # Apply OCR if no text is found
                image = page.to_image().convert("RGB")
                ocr_text = pytesseract.image_to_string(image)
                extracted_text += ocr_text + "\n"
    return extracted_text.strip()

# Skill keywords for extraction
SKILL_SET = {
    "Python", "Java", "C++", "Machine Learning", "SQL", "React", "Node.js", "Django",
    "JavaScript", "C#", "HTML", "CSS", "Kotlin", "Swift", "R", "TensorFlow", "PyTorch",
    "AWS", "Azure", "Docker", "Kubernetes", "Linux", "Flask", "FastAPI", "MongoDB", "PostgreSQL"
}

# Function to extract structured details from resume text
def extract_resume_details(text):
    doc = nlp(text)
    
    # Extract skills using NLP entity recognition
    extracted_skills = {ent.text for ent in doc.ents if ent.label_ == "SKILL"}
    
    # Extract skills using keyword matching
    found_skills = {word for word in text.split() if word in SKILL_SET}
    
    # Combine and clean skills
    final_skills = list(extracted_skills.union(found_skills))
    
    # Extract education
    education = [ent.text for ent in doc.ents if ent.label_ == "EDUCATION"]
    
    # Extract company experience
    experience = list(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    
    # Extract email and phone
    email = next((token.text for token in doc if token.like_email), None)
    phone = next((token.text for token in doc if token.like_num and len(token.text) >= 10), None)
    
    return {
        "skills": final_skills,
        "education": education,
        "experience": experience,
        "email": email,
        "phone": phone
    }

# Upload resume and extract details
@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        content = await file.read()
        text = content.decode("utf-8")
    
    # Extract structured details
    parsed_data = extract_resume_details(text)
    
    return {
        "filename": file.filename,
        "skills_extracted": parsed_data["skills"],
        "education": parsed_data["education"],
        "experience": parsed_data["experience"],
        "email": parsed_data["email"],
        "phone": parsed_data["phone"]
    }

# TF-IDF similarity function
def tfidf_similarity(resume_text, job_descriptions):
    vectorizer = TfidfVectorizer()
    texts = [resume_text] + job_descriptions
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = (tfidf_matrix[0] * tfidf_matrix[1:].T).toarray()
    return cosine_similarities.flatten()

# BERT similarity function
def bert_similarity(resume_text, job_descriptions):
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    job_embeddings = bert_model.encode(job_descriptions, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings)
    return similarities[0].tolist()

# Match jobs with the resume
@app.post("/match_jobs/")
async def match_jobs(resume: dict):
    resume_text = " ".join(resume["skills"])  # Use skills for matching
    job_descriptions = [job["description"] for job in job_data]

    tfidf_scores = tfidf_similarity(resume_text, job_descriptions)
    bert_scores = bert_similarity(resume_text, job_descriptions)

    job_scores = []
    for i, job in enumerate(job_data):
        avg_score = (tfidf_scores[i] + bert_scores[i]) / 2
        job_scores.append({"job": job["title"], "score": avg_score})

    job_scores.sort(key=lambda x: x["score"], reverse=True)
    return {"matched_jobs": job_scores}

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
