import spacy
import re

# Load the spaCy NLP model
nlp = spacy.load("en_core_web_md")

# Regex patterns for extracting contact info
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_REGEX = r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"


# Predefined skills list
SKILLS_LIST = {
    "Python", "Java", "C++", "SQL", "JavaScript", "Machine Learning",
    "Deep Learning", "NLP", "Django", "Flask", "React", "Node.js",
    "TensorFlow", "Keras", "Data Science", "AWS", "Docker", "Kubernetes",
    "FastAPI", "MongoDB", "PostgreSQL", "Redis"
}

# List of common degrees
DEGREE_KEYWORDS = ["Bachelor", "Master", "B.Sc", "M.Sc", "B.Tech", "M.Tech", "PhD", "MBA"]

# List of common job titles
JOB_TITLES = ["Software Engineer", "Data Scientist", "Project Manager", "Backend Developer", "Frontend Developer", "AI Engineer"]

def extract_resume_details(text):
    """Extract skills, education, experience, and contact details from resume text."""
    doc = nlp(text)

    # Extract skills by matching predefined skills list
    skills_found = {skill for skill in SKILLS_LIST if skill.lower() in text.lower()}

    # Extract education details
    education = [sent.text for sent in doc.sents if any(deg in sent.text for deg in DEGREE_KEYWORDS)]

    # Extract job titles (Experience)
    experience = [sent.text for sent in doc.sents if any(title in sent.text for title in JOB_TITLES)]

    # Extract email and phone number using regex
    email = re.findall(EMAIL_REGEX, text)
    phone = re.findall(PHONE_REGEX, text)

    return {
        "skills": list(skills_found),
        "education": education,
        "experience": experience,
        "email": email[0] if email else None,
        "phone": phone[0] if phone else None

    }
