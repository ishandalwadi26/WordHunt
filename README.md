# WordHeat

Single-player semantic word guessing game.

## Current V1
- Minimal UI with 3-color palette
- Home → Game → Result flow
- Countdown timer
- Dense guess history (repeated words allowed)
- Similarity feedback (mocked)
- Restrained game-board visual theme
- Two-column game UI: guess + nearest words on the left, history on the right
- Replay with a new hidden word
- Backend-ready project structure

## Run
### Frontend
Open `frontend/index.html` directly in a browser.

### Backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The current frontend prototype runs standalone and uses a local mock similarity engine. The next step is connecting the backend to `glove.840B.300d.txt`.

## Project structure
```text
wordheat/
├── README.md
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── backend/
    ├── main.py
    └── requirements.txt
```
