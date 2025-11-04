from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from flask_bootstrap import Bootstrap
import spacy, random, os, json, re, io, torch
from google import genai
from nltk.corpus import wordnet
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from difflib import SequenceMatcher
import PyPDF2
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = "supersecretkey"
Bootstrap(app)

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# In-memory user store
users = {}

# Load environment variables and Gemini client
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------- Distractors using WordNet --------------------
def get_distractors(word):
    distractors = set()
    for syn in wordnet.synsets(word, pos=wordnet.NOUN):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                distractors.add(name)
        for related in syn.hyponyms() + syn.hypernyms():
            for lemma in related.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    distractors.add(name)
    distractors = list(distractors)
    random.shuffle(distractors)
    return distractors[:3]


# -------------------- Improved LLM-based Q&A Generator --------------------
class QAGenerator:
    def __init__(self, qg_model_name="valhalla/t5-base-qg-hl", qa_model_name="distilbert-base-uncased-distilled-squad"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.qg_tokenizer = T5Tokenizer.from_pretrained(qg_model_name)
        self.qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name).to(self.device)
        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model_name,
            tokenizer=qa_model_name,
            device=0 if self.device == "cuda" else -1
        )

    def similar(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.7

    def chunk_text(self, text, max_sentences=4):
        sentences = re.split(r'(?<=[.!?]) +', text)
        for i in range(0, len(sentences), max_sentences):
            yield " ".join(sentences[i:i + max_sentences])

    def generate_qa_pairs(self, text, num_qa=5):
        if not text or len(text.strip()) < 50:
            return []

        all_qas = []
        seen_questions = set()

        for chunk in self.chunk_text(text):
            input_text = "generate questions: " + chunk
            inputs = self.qg_tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(self.device)

            num_beams = 10
            num_return_sequences = 5

            generated_ids = self.qg_model.generate(
                inputs["input_ids"],
                num_beams=num_beams,
                max_length=64,
                early_stopping=True,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=2
            )

            questions = [self.qg_tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_ids]

            for q in questions:
                q_clean = re.sub(r'\s+', ' ', q)
                if "?" not in q_clean:
                    continue
                if any(self.similar(q_clean, existing) for existing in seen_questions):
                    continue

                try:
                    answer = self.qa_pipeline(question=q_clean, context=chunk)
                    ans = answer["answer"].strip()
                    if ans and len(ans) > 3:
                        all_qas.append({"question": q_clean, "answer": ans})
                        seen_questions.add(q_clean)
                except Exception:
                    continue

                if len(all_qas) >= num_qa:
                    break

            if len(all_qas) >= num_qa:
                break

        return all_qas


qa_generator = QAGenerator()

# -------------------- Generate MCQs via LLM --------------------
def generate_mcqs(text, num_questions=5):
    qa_pairs = qa_generator.generate_qa_pairs(text, num_qa=num_questions)
    mcqs = []

    for qa in qa_pairs:
        question = qa["question"]
        answer = qa["answer"]

        # Try WordNet distractors first
        distractors = get_distractors(answer)

        # Fallback 1: Random words from text
        if len(distractors) < 3:
            words = list(set(re.findall(r'\b[A-Za-z]{4,}\b', text)))
            random.shuffle(words)
            distractors.extend([w for w in words if w.lower() != answer.lower()][:3 - len(distractors)])

        # Fallback 2: Add simple variations if still missing
        if len(distractors) < 3:
            distractors.extend([answer + str(i) for i in range(1, 4 - len(distractors))])

        distractors = distractors[:3]  # limit to 3

        options = distractors + [answer]
        random.shuffle(options)
        correct_answer = chr(65 + options.index(answer))
        mcqs.append((question, options, correct_answer))

    return mcqs



# -------------------- Generate Flashcards via LLM --------------------
def generate_flashcards(text):
    """
    Generate 8-10 unique flashcards using the LLM Q/A generator.
    Automatically handles chunking, randomness, and avoids repetition.
    """
    total_flashcards = 10  # fixed target
    all_flashcards = []
    seen = set()

    # Split text into chunks to improve question variety
    chunks = list(qa_generator.chunk_text(text, max_sentences=4))
    random.shuffle(chunks)

    for chunk in chunks:
        qa_pairs = qa_generator.generate_qa_pairs(chunk, num_qa=3)
        for qa in qa_pairs:
            q, a = qa["question"].strip(), qa["answer"].strip()
            key = (q.lower(), a.lower())
            if key not in seen and len(a) > 3:
                seen.add(key)
                all_flashcards.append((q, a))
            if len(all_flashcards) >= total_flashcards:
                break
        if len(all_flashcards) >= total_flashcards:
            break

    # Fallback: if not enough flashcards, generate more directly from the full text
    if len(all_flashcards) < total_flashcards:
        extra_needed = total_flashcards - len(all_flashcards)
        more_pairs = qa_generator.generate_qa_pairs(text, num_qa=extra_needed)
        for qa in more_pairs:
            q, a = qa["question"].strip(), qa["answer"].strip()
            key = (q.lower(), a.lower())
            if key not in seen and len(a) > 3:
                seen.add(key)
                all_flashcards.append((q, a))
            if len(all_flashcards) >= total_flashcards:
                break

    return all_flashcards


# -------------------- Summary (spaCy extractive) --------------------
def generate_summary(text):
    if not text:
        return "No text provided."
    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return "No valid sentences found."
    word_freq = {}
    for word in doc:
        if not word.is_stop and not word.is_punct:
            word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1
    max_freq = max(word_freq.values()) if word_freq else 1
    for word in word_freq:
        word_freq[word] /= max_freq
    sentence_scores = {}
    for sent in sentences:
        for word in sent:
            if word.text.lower() in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word.text.lower()]
    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary_len = max(1, len(sentences) // 4)
    summary = " ".join([sent.text for sent in sorted(ranked[:summary_len], key=lambda s: sentences.index(s))])
    return summary


# -------------------- Gemini AI Mode --------------------
def generate_with_ai(text, ai_type):
    if not text.strip():
        return {"summary": "No text provided.", "mcqs": [], "flashcards": []}
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
        2. 5 MCQs (each with 4 options labeled a-d, and 'Correct Answer: <letter>')
        3. 5 flashcards (term - definition format)

        Respond strictly in valid JSON:
        {{
            "summary": "...",
            "mcqs": [...],
            "flashcards": [...]
        }}

        TEXT:
        {text}
        """

    try:
        response = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        result_text = response.text.strip()
        try:
            data = json.loads(result_text)
            return {
                "summary": data.get("summary", ""),
                "mcqs": data.get("mcqs", []),
                "flashcards": data.get("flashcards", [])
            }
        except json.JSONDecodeError:
            pass

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


# -------------------- Parse AI Outputs --------------------
def parse_mcqs(text):
    mcqs = []
    blocks = re.split(r'\n\s*\n', text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        question = lines[0]
        if not re.search(r'\?', question):
            continue
        options = [l for l in lines if re.match(r'^[a-dA-D]\)', l)]
        correct = ""
        for l in lines:
            m = re.search(r'Correct Answer:\s*([a-dA-D])', l, re.I)
            if m:
                correct = m.group(1).lower()
        if question and options:
            mcqs.append({"question": question, "options": [opt.strip() for opt in options], "correct": correct})
    return mcqs


def parse_flashcards(text):
    flashcards = []
    for line in text.splitlines():
        if "-" not in line:
            continue
        term, definition = line.split("-", 1)
        flashcards.append({"term": term.strip(), "definition": definition.strip()})
    return flashcards


# -------------------- File Processing --------------------
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF extract error: {e}")
    return text


def read_text_file(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        print(f"Text read error: {e}")
        return None


def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return "\n".join([line.strip() for line in text.splitlines()]).strip()


# -------------------- Routes --------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        if u in users and users[u] == p:
            session['user'] = u
            return redirect(url_for('index'))
        flash("Invalid username or password!", "danger")
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        if u in users:
            flash("Username already exists!", "danger")
        else:
            users[u] = p
            flash("Sign-up successful! Please log in.", "success")
            return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))


@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        text = ""
        feature = request.form.get('feature')
        ai_type = request.form.get('ai_type')

        if 'files[]' in request.files and any(f.filename for f in request.files.getlist('files[]')):
            for file in request.files.getlist('files[]'):
                if file.filename.endswith('.pdf'):
                    text += extract_text_from_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += read_text_file(file)
        else:
            text = request.form.get('text', '')

        text = preprocess_text(text)
        if not text.strip():
            flash("Please provide text or upload a file.", "warning")
            return redirect(url_for('index'))

        if feature == "mcq":
            num_q = int(request.form['num_questions'])
            mcqs = generate_mcqs(text, num_q)
            return render_template('mcqs.html', mcqs=[(i + 1, m) for i, m in enumerate(mcqs)])
        elif feature == "summary":
            return render_template('summary.html', summary=generate_summary(text))
        elif feature == "flashcard":
            return render_template('flashcards.html', flashcards=generate_flashcards(text))
        elif feature == "ai_generate":
            ai_data = generate_with_ai(text, ai_type)
            return render_template('ai_results.html',
                                   summary=ai_data.get("summary", ""),
                                   mcqs=ai_data.get("mcqs", []),
                                   flashcards=ai_data.get("flashcards", []))
    return render_template('index.html')


@app.route("/export_summary", methods=["POST"])
def export_summary():
    summary_text = request.form.get("summary_text", "")
    if not summary_text.strip():
        flash("No summary available to export!", "warning")
        return redirect(url_for("index"))
    return Response(summary_text, mimetype="text/plain",
                    headers={"Content-Disposition": "attachment; filename=summary.txt"})


if __name__ == '__main__':
    app.run(debug=True)
