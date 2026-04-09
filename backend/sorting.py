import os
import re
import tempfile
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pdf2image import convert_from_path
import pytesseract

# Set path to tesseract (IMPORTANT for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def extract_text_with_ocr(pdf_path):
    text = ""

    try:
        images = convert_from_path(pdf_path, poppler_path=r"C:/poppler/Library/bin")
        for img in images:
            text += pytesseract.image_to_string(img)

    except Exception as e:
        print("OCR failed:", e)

    return text


# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use a temporary directory to avoid triggering file watchers/reloaders
RESUME_FOLDER = os.path.join(tempfile.gettempdir(), "resume-screening-uploads")
print("Using folder:", RESUME_FOLDER)
print("Files:", os.listdir(RESUME_FOLDER) if os.path.exists(RESUME_FOLDER) else "No folder")

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

ABBREVIATIONS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure"
}

# A broad library of technical skills for dynamic extraction
SKILLS_DB = [
    # Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "ruby", "php", "go", "rust", "swift", "kotlin", "scala",
    # Data & ML
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "sql", "nosql", "postgresql", "mongodb", "mysql", "redis", "spark", "hadoop", "tableau", "power bi",
    # Web & DevOps
    "react", "angular", "vue", "next.js", "node.js", "express", "flask", "django", "spring boot", "dot net", "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "jenkins", "ansible", "ci/cd", "git", "github", "gitlab",
    # Domains
    "project management", "agile", "scrum", "product management", "ui/ux", "graphic design", "figma", "cybersecurity", "blockchain", "fintech"
]

# ---------------- TEXT EXTRACTION ---------------- #

def extract_text_from_pdf(path):
    text_chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
    return "\n".join(text_chunks)


def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception as e:
        print(f"Error reading DOCX {path}: {e}")
        return ""


# ---------------- PREPROCESSING ---------------- #

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def expand_abbreviations(text):
    words = text.split()
    expanded_words = [
        ABBREVIATIONS.get(word, word)
        for word in words
    ]
    return " ".join(expanded_words)


def preprocess_text(text):
    text = clean_text(text)
    text = expand_abbreviations(text)
    return text


# ---------------- FEATURE EXTRACTION ---------------- #

def extract_skills(text, skills_db):
    """Extract skills present in the text from a predefined list."""
    text = text.lower()
    found_skills = set()
    for skill in skills_db:
        # Use word boundaries to avoid matching 'c' in 'cat'
        if re.search(rf'\b{re.escape(skill)}\b', text):
            found_skills.add(skill)
    return found_skills


def extract_experience_years(text):
    """Find mentions of years of experience in text."""
    # Look for patterns like '5+ years', '3 years of experience', 'Exp: 2 yrs'
    patterns = [
        r'(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?',
        r'experience[:\s]+(\d+)'
    ]
    all_years = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for m in matches:
            all_years.append(int(m))
    
    return max(all_years) if all_years else 0


def parse_sections(text):
    """Split resume into common sections with weights."""
    sections = {
        "experience": "",
        "skills": "",
        "projects": "",
        "education": "",
        "other": ""
    }
    
    # Common section headers
    headers = {
        "experience": [r'experience', r'work history', r'employment'],
        "skills": [r'skills', r'technical skills', r'expertise'],
        "projects": [r'projects', r'personal projects'],
        "education": [r'education', r'academic background']
    }
    
    lines = text.split('\n')
    current_section = "other"
    
    for line in lines:
        clean_line = line.strip().lower()
        found_new = False
        for sec_name, sec_patterns in headers.items():
            for p in sec_patterns:
                if re.match(rf'^{p}$', clean_line) or (len(clean_line) < 20 and p in clean_line):
                    current_section = sec_name
                    found_new = True
                    break
            if found_new: break
        
        sections[current_section] += line + "\n"
        
    return sections


# ---------------- LOAD RESUMES ---------------- #

def load_resumes():
    resumes_data = []

    if not os.path.exists(RESUME_FOLDER):
        return resumes_data

    for filename in os.listdir(RESUME_FOLDER):
        file_path = os.path.join(RESUME_FOLDER, filename)
        text = ""

        try:
            if filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(file_path)

                # OCR fallback if no text found
                if not text.strip() or len(text.strip()) < 50:
                    print(f"OCR used for: {filename}")
                    text = extract_text_with_ocr(file_path)

            elif filename.lower().endswith(".docx"):
                text = preprocess_text(extract_text_with_ocr(file_path))
        except Exception:
            continue

        if text and text.strip():
            sections = parse_sections(text)
            # Create a weighted text block where experience and skills are more prominent
            weighted_text = (
                sections["experience"] * 2 + 
                sections["skills"] * 2 + 
                sections["projects"] * 2 + 
                sections["education"] + 
                sections["other"]
            )
            
            resumes_data.append({
                "filename": filename,
                "raw_text": text,
                "weighted_text": preprocess_text(weighted_text),
                "skills": extract_skills(text, SKILLS_DB),
                "years": extract_experience_years(text)
            })

    return resumes_data


# ---------------- MAIN RANKING FUNCTION ---------------- #

def rank_resumes(query):
    resumes_data = load_resumes()

    if not resumes_data:
        return []

    # 1. PREPARE QUERY
    processed_query = preprocess_text(query)
    query_skills = extract_skills(query, SKILLS_DB)
    required_years = extract_experience_years(query)

    # 2. SEMANTIC SIMILARITY (40%)
    resumes_text = [r["weighted_text"] for r in resumes_data]
    embeddings = model.encode(resumes_text + [processed_query])
    query_embedding = embeddings[-1]
    resume_embeddings = embeddings[:-1]
    semantic_scores = cosine_similarity([query_embedding], resume_embeddings)[0]

    # 3. DYNAMIC SKILL MATCH (30%)
    skill_scores = []
    if not query_skills:
        skill_scores = [0] * len(resumes_data)
    else:
        for r in resumes_data:
            match_count = len(query_skills.intersection(r["skills"]))
            score = match_count / len(query_skills)
            skill_scores.append(score)

    # 4. EXPERIENCE SCORE (20%)
    exp_scores = []
    for r in resumes_data:
        if required_years == 0:
            exp_scores.append(1.0)
        else:
            # Score is 1.0 if they meet req, less if not, slightly more if exceed
            ratio = r["years"] / required_years
            score = min(1.2, ratio) # Cap at 1.2 to not over-weight
            exp_scores.append(score)

    # 5. TF-IDF SCORE (10%)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(resumes_text + [processed_query])
    tfidf_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # 6. FINAL HYBRID SCORE
    final_results = []
    for i in range(len(resumes_data)):
        final_score = (
            0.4 * semantic_scores[i] +
            0.3 * skill_scores[i] +
            0.2 * exp_scores[i] +
            0.1 * tfidf_scores[i]
        )
        final_results.append((resumes_data[i]["filename"], final_score))

    # SORT
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results
