from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
import os
from google import genai
from nltk.corpus import wordnet
import json

app = Flask(__name__)
app.secret_key = "supersecretkey"  
Bootstrap(app)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# --- Simple in-memory user store ---
users = {}  # {'username': 'password'}


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# -------------------- FUNCTION: Distractors --------------------
from nltk.corpus import wordnet
import random

def get_distractors(word):
    """Generate up to 3 meaningful distractors for a given word using WordNet."""
    distractors = set()

    for syn in wordnet.synsets(word, pos=wordnet.NOUN):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                distractors.add(name)

        # Include hyponyms and hypernyms (related terms)
        for related in syn.hyponyms() + syn.hypernyms():
            for lemma in related.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    distractors.add(name)

    distractors = list(distractors)
    random.shuffle(distractors)
    return distractors[:3]


# -------------------- FUNCTION: Generate Improved MCQs --------------------
def generate_mcqs(text, num_questions=5):
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
    mcqs = []
    tried_sentences = set()

    while len(mcqs) < num_questions and len(tried_sentences) < len(sentences):
        sentence = random.choice(sentences)
        tried_sentences.add(sentence)
        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

        if not nouns:
            continue

        subject = random.choice(nouns)
        question_stem = sentence.replace(subject, "______", 1)

        distractors = get_distractors(subject)

        if len(distractors) < 3:
            backup_nouns = [token.text for token in doc if token.pos_ == "NOUN" and token.text != subject]
            random.shuffle(backup_nouns)
            distractors.extend(backup_nouns[:3 - len(distractors)])

        distractors = distractors[:3]

        if len(distractors) < 3:
            continue  # skip if still not enough distractors

        options = distractors + [subject]
        random.shuffle(options)
        correct_answer = chr(65 + options.index(subject))

        mcqs.append((question_stem, options, correct_answer))

    return mcqs



# -------------------- FUNCTION: Generate Summary --------------------
def generate_summary(text):
    """
    Generates a summary that is approximately one-third the length of the original text.
    Uses frequency-based extractive summarization with spaCy.
    """
    if not text:
        return "No text provided."

    doc = nlp(text)
    sentences = list(doc.sents)
    total_sentences = len(sentences)
    if total_sentences == 0:
        return "No valid sentences found."

    # Frequency calculation for words
    word_freq = {}
    for word in doc:
        if not word.is_stop and not word.is_punct:
            word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1

    max_freq = max(word_freq.values()) if word_freq else 1
    for word in word_freq:
        word_freq[word] /= max_freq

    # Sentence scoring
    sentence_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text.lower() in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word.text.lower()]

    # Sort sentences by score
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

    # Select top 1/4 of sentences
    summary_length = max(1, total_sentences // 4)
    summarized_sentences = ranked_sentences[:summary_length]

    # Preserve original order for readability
    summarized_sentences = sorted(summarized_sentences, key=lambda s: list(doc.sents).index(s))

    summary = " ".join([sent.text for sent in summarized_sentences])
    return summary


# -------------------- FUNCTION: Generate Flashcards --------------------
def generate_flashcards(text):
    if not text:
        return []
    doc = nlp(text)
    flashcards = []
    for sent in doc.sents:
        sent_doc = nlp(sent.text)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        if nouns:
            answer = nouns[0]
            question = sent.text.replace(answer, "______", 1)
            flashcards.append((question, answer))
    return flashcards


# -------------------- FUNCTION: Generate with Gemini AI --------------------
import json, re

def generate_with_ai(text, ai_type):
    """Generates summary, MCQs, or flashcards using Gemini AI."""
    if not text.strip():
        return {"summary": "No text provided.", "mcqs": [], "flashcards": []}

    # Build structured prompt
    if ai_type == "summary":
        prompt = f"Summarize this text in 3-5 sentences:\n\n{text}"
    elif ai_type == "mcq":
        prompt = f"""
        Create 5 multiple-choice questions from the following text.
        Each question must have 4 options (a–d) and specify:
        "Correct Answer: <letter>"

        Example format:
        Q1: What is AI?
        a) Animal Intelligence
        b) Artificial Intelligence
        c) Automated Input
        d) Analog Integration
        Correct Answer: b

        TEXT:
        {text}
        """
    elif ai_type == "flashcard":
        prompt = f"""
        Create 5 flashcards from this text.
        Each should be in the format:
        term - definition

        Example:
        AI - Artificial Intelligence is a field of computer science that builds intelligent systems.

        TEXT:
        {text}
        """
    else:
        prompt = f"""
        Read the following text and generate:
        1. A short summary (3-5 sentences)
        2. 5 MCQs (each with 4 options labeled a–d, and 'Correct Answer: <letter>')
        3. 5 flashcards (term - definition format)

        Respond strictly in valid JSON:
        {{
            "summary": "...",
            "mcqs": [
                {{
                    "question": "...",
                    "options": ["a) ...", "b) ...", "c) ...", "d) ..."],
                    "correct": "b"
                }}
            ],
            "flashcards": [
                {{
                    "term": "...",
                    "definition": "..."
                }}
            ]
        }}

        TEXT:
        {text}
        """

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        result_text = response.text.strip()

        # Try JSON parsing first
        try:
            data = json.loads(result_text)
            # Ensure all required keys exist
            return {
                "summary": data.get("summary", ""),
                "mcqs": data.get("mcqs", []),
                "flashcards": data.get("flashcards", [])
            }
        except json.JSONDecodeError:
            pass  # fallback below

        # Fallback for non-JSON responses
        if ai_type == "summary":
            return {"summary": result_text, "mcqs": [], "flashcards": []}
        elif ai_type == "mcq":
            return {"summary": "", "mcqs": parse_mcqs(result_text), "flashcards": []}
        elif ai_type == "flashcard":
            return {"summary": "", "mcqs": [], "flashcards": parse_flashcards(result_text)}
        else:
            return {"summary": "AI response not structured as JSON.", "mcqs": [], "flashcards": []}

    except Exception as e:
        print("Gemini generation error:", e)
        return {"summary": "AI generation failed.", "mcqs": [], "flashcards": []}


# -------------------- HELPERS: Parse MCQs and Flashcards --------------------
def parse_mcqs(text):
    """Extracts formatted MCQs from AI response."""
    mcqs = []
    blocks = re.split(r'\n\s*\n', text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        # Extract question
        question = lines[0]
        if not re.search(r'\?', question):
            continue  # skip if it's not a question

        # Extract options
        options = [l for l in lines if re.match(r'^[a-dA-D]\)', l)]
        correct = ""

        # Extract correct answer
        for l in lines:
            match = re.search(r'Correct Answer:\s*([a-dA-D])', l, re.IGNORECASE)
            if match:
                correct = match.group(1).lower()

        if question and options:
            mcqs.append({
                "question": question,
                "options": [opt.strip() for opt in options],
                "correct": correct
            })
    return mcqs


def parse_flashcards(text):
    """Parses flashcards in 'term - definition' format."""
    flashcards = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "-" not in line:
            continue
        parts = line.split("-", 1)
        term, definition = parts[0].strip(), parts[1].strip()
        if term and definition:
            flashcards.append({"term": term, "definition": definition})
    return flashcards




# -------------------- FUNCTION: Process PDF --------------------
import PyPDF2
import io
import re


def extract_text_from_pdf(file):
    """
    Extracts text from an uploaded PDF file.

    Args:
        pdf_file: An uploaded file object from Streamlit (st.file_uploader).

    Returns:
        str: The extracted text from the PDF, or None if an error occurs.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text


def read_text_file(file):
    """
    Reads text from an uploaded plain text file.

    Args:
        text_file: An uploaded file object from Streamlit (st.file_uploader).\

    Returns:
        str: The content of the text file, or None if an error occurs.
    """
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None


def preprocess_text(text):
    """
    Performs basic text preprocessing to clean the input for LLMs.
    - Removes extra whitespace, ensures consistent line breaks.
    - This helps in providing cleaner input to the models.

    Args:
        text (str): The raw input text.

    Returns:
        str: The preprocessed text.
    """
    if text is None:
        return ""
    

    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)

    # Remove leading/trailing whitespace from each line and then join them
    text = "\n".join([line.strip() for line in text.splitlines()])

    # Strip any leading/trailing whitespace from the entire text
    return text.strip()


# -------------------- ROUTES --------------------

# Login Page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password!", "danger")

    return render_template('login.html')

# Sign-up Page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash("Username already exists!", "danger")
        else:
            users[username] = password
            flash("Sign-up successful! Please log in.", "success")
            return redirect(url_for('login'))

    return render_template('signup.html')

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

# Main Text Processor Page
@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        text = ""
        feature = request.form.get('feature')
        ai_type = request.form.get('ai_type')  # for AI generation

        if 'files[]' in request.files and any(file.filename for file in request.files.getlist('files[]')):
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += extract_text_from_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += read_text_file(file)
        else:
            text = request.form.get('text', '')

        text = preprocess_text(text)
        if not text.strip():
            flash("Please provide text input or upload a file.", "warning")
            return redirect(url_for('index'))

        if feature == "mcq":
            num_questions = int(request.form['num_questions'])
            mcqs = generate_mcqs(text, num_questions)
            mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
            return render_template('mcqs.html', mcqs=mcqs_with_index)

        elif feature == "summary":
            summary = generate_summary(text)
            return render_template('summary.html', summary=summary)

        elif feature == "flashcard":
            flashcards = generate_flashcards(text)
            return render_template('flashcards.html', flashcards=flashcards)

        elif feature == "ai_generate":
            ai_data = generate_with_ai(text, ai_type)
            return render_template(
                'ai_results.html',
                summary=ai_data.get("summary", ""),
                mcqs=ai_data.get("mcqs", []),
                flashcards=ai_data.get("flashcards", [])
            )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
