from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
import pdfplumber

app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session
Bootstrap(app)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# --- Simple in-memory user store (for demo purposes) ---
users = {}  # {'username': 'password'}


from nltk.corpus import wordnet

def get_distractors(word):
    distractors = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            l = lemma.name().replace('_', ' ')
            if l.lower() != word.lower() and l not in distractors:
                distractors.append(l)
    return distractors[:3]


# -------------------- FUNCTION: Generate MCQs --------------------
def generate_mcqs(text, num_questions=5):
    if not text:
        return []

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    num_questions = min(num_questions, len(sentences))
    selected_sentences = random.sample(sentences, num_questions)
    mcqs = []

    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

        if len(nouns) < 2:
            continue

        noun_counts = Counter(nouns)
        subject = noun_counts.most_common(1)[0][0]

        # Only replace the first occurrence
        question_stem = sentence.replace(subject, "______", 1)

        answer_choices = [subject]
        distractors = list(set(nouns) - {subject})
        while len(distractors) < 3:
            distractors.append("[Distractor]")

        random.shuffle(distractors)
        for distractor in distractors[:3]:
            answer_choices.append(distractor)

        random.shuffle(answer_choices)
        correct_answer = chr(64 + answer_choices.index(subject) + 1)
        mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs

# -------------------- FUNCTION: Generate Summary --------------------
def generate_summary(text, num_sentences=3):
    if not text:
        return "No text provided."
    doc = nlp(text)
    sentence_scores = {}
    word_freq = {}
    for word in doc:
        if word.is_stop or word.is_punct:
            continue
        word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1
    max_freq = max(word_freq.values()) if word_freq else 1
    for word in word_freq:
        word_freq[word] /= max_freq
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word.text.lower()]
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = " ".join([sent.text for sent in summarized_sentences[:num_sentences]])
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

# -------------------- FUNCTION: Process PDF --------------------
def process_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

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

        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8')
        else:
            text = request.form['text']

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

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)