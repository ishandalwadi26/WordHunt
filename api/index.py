from pathlib import Path
import json
import random
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
VFILE = DATA / "vectors.npy"
IFILE = DATA / "word_index.json"

app = FastAPI(title="WordHeat API", version="5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectors = None
word_to_id = {}
id_to_word = []
games = {}


startup_error = None


@app.on_event("startup")
def load_vectors():
    global vectors, word_to_id, id_to_word, startup_error

    try:
        print(f"Loading vectors from: {VFILE}")
        print(f"Vector file exists: {VFILE.exists()}")
        print(f"Index file exists: {IFILE.exists()}")

        if not VFILE.exists():
            startup_error = f"vectors.npy not found: {VFILE}"
            print(startup_error)
            return

        if not IFILE.exists():
            startup_error = f"word_index.json not found: {IFILE}"
            print(startup_error)
            return

        vectors = np.load(VFILE, mmap_mode="r")
        print(f"Vectors shape: {vectors.shape}")
        print(f"Vectors dtype: {vectors.dtype}")

        word_to_id = json.loads(
            IFILE.read_text(encoding="utf-8")
        )

        id_to_word = [""] * len(word_to_id)

        for word, idx in word_to_id.items():
            id_to_word[int(idx)] = word

        print(
            f"WORDHEAT BACKEND 5.1: "
            f"loaded {len(id_to_word):,} vectors."
        )

    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        vectors = None
        print(f"STARTUP ERROR: {startup_error}")

class StartRequest(BaseModel):
    hidden_word: str | None = None


class GuessRequest(BaseModel):
    game_id: str
    word: str = Field(min_length=1, max_length=80)


class HintRequest(BaseModel):
    game_id: str


class EndRequest(BaseModel):
    game_id: str


def require_model():
    if vectors is None:
        raise HTTPException(
            status_code=503,
            detail="Vectors are not prepared. Run prepare_vectors.py first."
        )


def clean(word):
    return word.strip().lower()


def hidden_similarities(game):
    """
    Calculate semantic similarity of every vocabulary word to the hidden word.

    Because vectors.npy was normalized during preprocessing, matrix multiplication
    is cosine similarity.

    The returned array is used for BOTH:
      - normal guess scores
      - hint scores

    Therefore a hint can never claim a score that the same word gets when guessed.
    """
    hidden_id = game["hidden_id"]
    hidden_vector = np.asarray(vectors[hidden_id], dtype=np.float32)

    similarities = np.asarray(vectors @ hidden_vector)

    # The answer itself is never exposed as a hint.
    similarities[hidden_id] = -np.inf

    return similarities


def rank_score(similarities, word_id):
    """
    WordHeat score is based on semantic rank against the hidden word.

    0..100 means approximately how high this word ranks among the entire
    prepared vocabulary for the current hidden word.

    This fixes the old unfair behavior where hints used artificial percentages
    that did not match the score received when the player guessed that word.
    """
    value = float(similarities[word_id])

    if not np.isfinite(value):
        return 100

    valid = similarities[np.isfinite(similarities)]

    # Percentile rank: how much of the vocabulary this word is closer than.
    rank = np.searchsorted(np.sort(valid), value, side="right") - 1
    denominator = max(1, len(valid) - 1)

    return int(round((rank / denominator) * 100))


def rank_score_fast(similarities, word_id):
    """
    Fast equivalent of rank_score().

    Uses np.count_nonzero instead of sorting all 100K values. This is used
    during hint generation so a hint can be selected much faster.
    """
    value = float(similarities[word_id])

    if not np.isfinite(value):
        return 100

    valid = similarities[np.isfinite(similarities)]
    rank = np.count_nonzero(valid <= value) - 1
    denominator = max(1, len(valid) - 1)

    return int(round((rank / denominator) * 100))


def normal_cosine_score(cosine):
    """
    Used only for nearest-word information if needed.
    """
    return int(round(max(0.0, min(1.0, float(cosine))) * 100))


def nearest(word_id, limit=8, exclude=()):
    """
    Nearest words to the player's current guess.

    These percentages are cosine similarity percentages for this panel,
    not the hidden-word game score.
    """
    q = np.asarray(vectors[word_id], dtype=np.float32)
    sims = np.asarray(vectors @ q)

    for idx in exclude:
        sims[int(idx)] = -np.inf

    n = min(limit, len(sims))
    if n <= 0:
        return []

    ids = np.argpartition(sims, -n)[-n:]
    ids = ids[np.argsort(sims[ids])[::-1]]

    return [
        {
            "word": id_to_word[int(idx)],
            "score": normal_cosine_score(float(sims[idx]))
        }
        for idx in ids
    ]


def choose_hint(game):
    """
    Select one real semantic hint whose DISPLAYED SCORE is also its real
    WordHeat guess score.

    Hint progression:
      - no strong guess: approximately 20 -> 45 -> 70 -> 96
      - if the player is already at 45: approximately 50 -> 65 -> 81 -> 96
      - if already at 70: approximately 75 -> 82 -> 89 -> 96

    The important rule is that we choose the WORD by its actual percentile
    score first, then return that exact same score. There is no fake score.
    """

    hint_number = len(game["hints"]) + 1
    if hint_number > 4:
        raise HTTPException(400, "All 4 hints have already been used.")

    similarities = hidden_similarities(game)

    # Words already guessed or shown as hints cannot be returned.
    excluded_ids = set()
    for word in game["guessed_words"] | set(game["hints"]):
        idx = word_to_id.get(word)
        if idx is not None:
            excluded_ids.add(int(idx))

    for idx in excluded_ids:
        similarities[idx] = -np.inf

    valid_ids = np.flatnonzero(np.isfinite(similarities))
    if len(valid_ids) == 0:
        raise HTTPException(400, "No unused semantic hints are available.")

    # Calculate the player's current actual best score.
    best_guess = max(
        [item["score"] for item in game["history"]],
        default=0
    )

    # The target score is a TARGET, not a fake displayed score.
    if best_guess <= 20:
        targets = [20, 45, 70, 96]
        target_score = targets[hint_number - 1]
    else:
        if hint_number == 4:
            target_score = 96
        else:
            # Always move above the player's current best and spread toward 96.
            start = best_guess + 5
            progress = (hint_number - 1) / 3
            target_score = round(
                start + (96 - start) * progress
            )

    # Keep hints strictly above the player's current best.
    target_score = max(target_score, best_guess + 1)

    if hint_number == 4:
        target_score = 96

    target_score = min(96, target_score)

    # Fast percentile lookup.
    #
    # The old implementation sorted all ~100K words every time Hint was
    # clicked. That made the hint feel slow. argpartition finds the target
    # percentile without fully sorting the vocabulary.
    target_percentile = target_score / 100.0
    target_pos = int(round(
        target_percentile * (len(valid_ids) - 1)
    ))

    valid_values = similarities[valid_ids]

    # Get a small neighborhood around the desired percentile in O(n)
    # instead of sorting the complete vocabulary.
    radius = min(120, max(20, len(valid_ids) // 500))
    low = max(0, target_pos - radius)
    high = min(len(valid_ids) - 1, target_pos + radius)

    # Partition once around the target region.
    partitioned_positions = np.argpartition(
        valid_values,
        [low, high]
    )

    candidate_positions = partitioned_positions[low:high + 1]

    best_id = None
    best_distance = float("inf")

    for pos in candidate_positions:
        idx = int(valid_ids[int(pos)])
        actual = rank_score_fast(similarities, idx)

        # Must be strictly above current best.
        if actual <= best_guess:
            continue

        distance = abs(actual - target_score)

        if distance < best_distance:
            best_distance = distance
            best_id = idx

            if actual == target_score:
                break

    if best_id is None:
        # If no word can beat the current score, tell the player honestly.
        raise HTTPException(
            400,
            "There is no unused hint above your current score."
        )

    actual_score = rank_score(similarities, best_id)

    return {
        "word": id_to_word[best_id],
        "score": actual_score,
        "cosine": round(float(similarities[best_id]), 6),
        "hint_number": hint_number,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok" if vectors is not None else "error",
        "version": "5.1",
        "vectors_loaded": vectors is not None,
        "startup_error": startup_error,
        "vector_file_exists": VFILE.exists(),
        "index_file_exists": IFILE.exists(),
        "vocabulary_size": len(word_to_id),
        "dimensions": int(vectors.shape[1]) if vectors is not None else 0,
        "endpoints": [
            "/api/game/start",
            "/api/game/guess",
            "/api/game/hint",
            "/api/game/end",
        ],
    }

@app.post("/api/game/start")
def start(req: StartRequest):
    require_model()

    hidden = clean(req.hidden_word) if req.hidden_word else random.choice(id_to_word)

    if hidden not in word_to_id:
        raise HTTPException(
            400,
            "Hidden word is not in the prepared vocabulary."
        )

    gid = uuid.uuid4().hex

    games[gid] = {
        "hidden_word": hidden,
        "hidden_id": word_to_id[hidden],
        "history": [],
        "guessed_words": set(),
        "hints": [],
        "ended": False,
    }

    return {"game_id": gid, "backend_version": "5.1"}


@app.post("/api/game/guess")
def guess(req: GuessRequest):
    require_model()

    game = games.get(req.game_id)

    if not game:
        raise HTTPException(404, "Game not found.")
    if game["ended"]:
        raise HTTPException(400, "This game has already ended.")

    word = clean(req.word)

    if word not in word_to_id:
        raise HTTPException(
            400,
            "That word is not in the WordHeat vocabulary."
        )

    word_id = word_to_id[word]

    # IMPORTANT:
    # Use the exact same score function used to select hints.
    similarities = hidden_similarities(game)
    current_score = rank_score_fast(similarities, word_id)

    item = {
        "word": word,
        "score": current_score,
        "cosine": round(float(similarities[word_id]), 6),
    }

    # Repeated guesses remain allowed.
    game["history"].append(item)
    game["guessed_words"].add(word)

    return {
        "guess": word,
        "score": current_score,
        "cosine": item["cosine"],
        "won": word == game["hidden_word"],
        "nearest_words": nearest(
            word_id,
            8,
            (word_id, game["hidden_id"])
        ),
        "history": game["history"],
    }


@app.post("/api/game/hint")
def hint(req: HintRequest):
    require_model()

    game = games.get(req.game_id)

    if not game:
        raise HTTPException(404, "Game not found.")
    if game["ended"]:
        raise HTTPException(400, "This game has already ended.")

    if len(game["hints"]) >= 4:
        raise HTTPException(
            400,
            "All 4 hints have already been used."
        )

    item = choose_hint(game)

    game["hints"].append(item["word"])

    return {
        "hint": item,
        "hint_number": len(game["hints"]),
        "max_hints": 4,
    }


@app.post("/api/game/end")
def end_game(req: EndRequest):
    game = games.get(req.game_id)

    if not game:
        raise HTTPException(404, "Game not found.")

    game["ended"] = True

    return {
        "ended": True,
        "answer": game["hidden_word"],
        "history": game["history"],
    }


@app.get("/api/game/{game_id}")
def game_state(game_id: str):
    game = games.get(game_id)

    if not game:
        raise HTTPException(404, "Game not found.")

    return {
        "game_id": game_id,
        "history": game["history"],
        "hints_used": len(game["hints"]),
    }
