The personalizes Education Assets Generator is an educational web application built using Flask that allows users to:

Upload PDF or text files

Automatically generate:

📋 Multiple Choice Questions (MCQs)

🧾 Summaries

🃏 Flashcards

Or use AI-powered generation 

The app also supports user authentication (Signup/Login) and can be extended into a full-fledged learning platform.

Features

✅ User signup, login, and logout
✅ Text/PDF input processing
✅ Generate:

Summary

MCQs (with correct answer tracking)

Flashcards
✅ “Generate with AI” mode (for advanced question and summary generation)
✅ Modular, clean folder structure

Assets-generator/
├── app.py
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── mcqs.html
│   ├── summary.html
│   ├── flashcards.html
│   └── ai_results.html
├── static/
│   ├── css/  optional
│   └── js/   optional
├── requirements.txt
├── .env                # (store your API key here, not committed)
├── .gitignore
└── README.md
