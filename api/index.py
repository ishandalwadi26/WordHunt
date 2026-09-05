
from pathlib import Path
import json
import os
import random
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from upstash_redis import Redis


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
VFILE = DATA / "vectors.npy"
IFILE = DATA / "word_index.json"


app = FastAPI(title="WordHeat API", version="6.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REDIS
# ============================================================

def create_redis():
    # Standard Upstash names
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

    # Vercel Upstash integration names
    if not url:
        url = os.getenv("KV_REST_API_URL")

    if not token:
        token = os.getenv("KV_REST_API_TOKEN")

    # Additional names used by some Upstash/Vercel integrations
    if not url:
        url = os.getenv("KV_URL")

    if not token:
        token = os.getenv("REDIS_TOKEN")

    if not url:
        url = os.getenv("REDIS_URL")

    if not url or not token:
        raise RuntimeError(
            "Redis environment variables were not found. "
            "Expected UPSTASH_REDIS_REST_URL / "
            "UPSTASH_REDIS_REST_TOKEN or KV_REST_API_URL / "
            "KV_REST_API_TOKEN."
        )

    return Redis(
        url=url,
        token=token,
    )


redis = None
redis_error = None

try:
    redis = create_redis()
except Exception as exc:
    redis_error = f"{type(exc).__name__}: {exc}"


GAME_TTL = 60 * 60 * 2


def game_key(game_id):
    return f"wordheat:game:{game_id}"


def save_game(game_id, game):
    if redis is None:
        raise HTTPException(
            status_code=503,
            detail="Redis is not connected."
        )

    redis.set(
        game_key(game_id),
        json.dumps(game),
        ex=GAME_TTL,
    )


def load_game(game_id):
    if redis is None:
        raise HTTPException(
            status_code=503,
            detail="Redis is not connected."
        )

    data = redis.get(
        game_key(game_id)
    )

    if data is None:
        return None

    if isinstance(data, bytes):
        data = data.decode("utf-8")

    if isinstance(data, str):
        return json.loads(data)

    return data


# ============================================================
# MODEL DATA
# ============================================================

vectors = None
word_to_id = {}
id_to_word = []

startup_error = None


@app.on_event("startup")
def load_vectors():
    global vectors
    global word_to_id
    global id_to_word
    global startup_error

    try:
        print(f"Loading vectors from: {VFILE}")
        print(f"Vector file exists: {VFILE.exists()}")
        print(f"Index file exists: {IFILE.exists()}")

        if not VFILE.exists():
            startup_error = (
                f"vectors.npy not found: {VFILE}"
            )
            print(startup_error)
            return

        if not IFILE.exists():
            startup_error = (
                f"word_index.json not found: {IFILE}"
            )
            print(startup_error)
            return

        vectors = np.load(
            VFILE,
            mmap_mode="r"
        )

        print(
            f"Vectors shape: {vectors.shape}"
        )
        print(
            f"Vectors dtype: {vectors.dtype}"
        )

        word_to_id = json.loads(
            IFILE.read_text(
                encoding="utf-8"
            )
        )

        id_to_word = [
            ""
        ] * len(word_to_id)

        for word, idx in word_to_id.items():
            id_to_word[int(idx)] = word

        print(
            f"WORDHEAT BACKEND 6.1: "
            f"loaded {len(id_to_word):,} vectors."
        )

    except Exception as exc:
        startup_error = (
            f"{type(exc).__name__}: {exc}"
        )

        vectors = None
        word_to_id = {}
        id_to_word = []

        print(
            f"STARTUP ERROR: {startup_error}"
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class StartRequest(BaseModel):
    hidden_word: str | None = None


class GuessRequest(BaseModel):
    game_id: str
    word: str = Field(
        min_length=1,
        max_length=80
    )


class HintRequest(BaseModel):
    game_id: str


class EndRequest(BaseModel):
    game_id: str


# ============================================================
# HELPERS
# ============================================================

def require_model():
    if vectors is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vectors are not prepared. "
                "Run prepare_vectors.py first."
            )
        )


def require_redis():
    if redis is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Redis is not connected."
                + (
                    f" {redis_error}"
                    if redis_error
                    else ""
                )
            )
        )


def clean(word):
    return word.strip().lower()


def get_game(game_id):
    game = load_game(game_id)

    if game is None:
        raise HTTPException(
            status_code=404,
            detail="Game not found."
        )

    return game


# ============================================================
# SIMILARITY
# ============================================================

def hidden_similarities(game):
    hidden_id = int(
        game["hidden_id"]
    )

    hidden_vector = np.asarray(
        vectors[hidden_id],
        dtype=np.float32
    )

    similarities = np.asarray(
        vectors @ hidden_vector
    )

    similarities[hidden_id] = -np.inf

    return similarities


def rank_score(similarities, word_id):
    value = float(
        similarities[word_id]
    )

    if not np.isfinite(value):
        return 100

    valid = similarities[
        np.isfinite(similarities)
    ]

    rank = (
        np.searchsorted(
            np.sort(valid),
            value,
            side="right"
        )
        - 1
    )

    denominator = max(
        1,
        len(valid) - 1
    )

    return int(
        round(
            (rank / denominator) * 100
        )
    )


def rank_score_fast(similarities, word_id):
    value = float(
        similarities[word_id]
    )

    if not np.isfinite(value):
        return 100

    valid = similarities[
        np.isfinite(similarities)
    ]

    rank = (
        np.count_nonzero(
            valid <= value
        )
        - 1
    )

    denominator = max(
        1,
        len(valid) - 1
    )

    return int(
        round(
            (rank / denominator) * 100
        )
    )


def normal_cosine_score(cosine):
    return int(
        round(
            max(
                0.0,
                min(
                    1.0,
                    float(cosine)
                )
            ) * 100
        )
    )


# ============================================================
# NEAREST WORDS
# ============================================================

def nearest(
    word_id,
    limit=8,
    exclude=()
):
    q = np.asarray(
        vectors[word_id],
        dtype=np.float32
    )

    sims = np.asarray(
        vectors @ q
    )

    for idx in exclude:
        sims[int(idx)] = -np.inf

    n = min(
        limit,
        len(sims)
    )

    if n <= 0:
        return []

    ids = np.argpartition(
        sims,
        -n
    )[-n:]

    ids = ids[
        np.argsort(
            sims[ids]
        )[::-1]
    ]

    return [
        {
            "word": id_to_word[int(idx)],
            "score": normal_cosine_score(
                float(sims[idx])
            )
        }
        for idx in ids
        if np.isfinite(sims[idx])
    ]


# ============================================================
# HINT SYSTEM
# ============================================================

def choose_hint(game):
    hint_number = (
        len(game["hints"]) + 1
    )

    if hint_number > 4:
        raise HTTPException(
            status_code=400,
            detail=(
                "All 4 hints have already been used."
            )
        )

    similarities = hidden_similarities(
        game
    )

    hidden_id = int(
        game["hidden_id"]
    )

    excluded_words = set(
        game.get(
            "guessed_words",
            []
        )
    )

    excluded_words.update(
        game.get(
            "hints",
            []
        )
    )

    excluded_ids = set()

    for word in excluded_words:
        idx = word_to_id.get(word)

        if idx is not None:
            excluded_ids.add(
                int(idx)
            )

    excluded_ids.add(
        hidden_id
    )

    for idx in excluded_ids:
        similarities[idx] = -np.inf

    valid_ids = np.flatnonzero(
        np.isfinite(similarities)
    )

    if len(valid_ids) == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "No unused semantic hints "
                "are available."
            )
        )

    best_guess = max(
        [
            item["score"]
            for item in game.get(
                "history",
                []
            )
        ],
        default=0
    )

    if best_guess <= 20:
        targets = [
            20,
            45,
            70,
            96
        ]

        target_score = targets[
            hint_number - 1
        ]

    else:
        if hint_number == 4:
            target_score = 96
        else:
            start = best_guess + 5

            progress = (
                (hint_number - 1)
                / 3
            )

            target_score = round(
                start
                + (96 - start)
                * progress
            )

    target_score = max(
        target_score,
        best_guess + 1
    )

    if hint_number == 4:
        target_score = 96

    target_score = min(
        96,
        target_score
    )

    target_percentile = (
        target_score / 100.0
    )

    target_pos = int(
        round(
            target_percentile
            * (len(valid_ids) - 1)
        )
    )

    valid_values = similarities[
        valid_ids
    ]

    radius = min(
        120,
        max(
            20,
            len(valid_ids) // 500
        )
    )

    low = max(
        0,
        target_pos - radius
    )

    high = min(
        len(valid_ids) - 1,
        target_pos + radius
    )

    partitioned_positions = np.argpartition(
        valid_values,
        [low, high]
    )

    candidate_positions = (
        partitioned_positions[
            low:high + 1
        ]
    )

    best_id = None
    best_distance = float("inf")

    for pos in candidate_positions:
        idx = int(
            valid_ids[int(pos)]
        )

        actual = rank_score_fast(
            similarities,
            idx
        )

        if actual <= best_guess:
            continue

        distance = abs(
            actual - target_score
        )

        if distance < best_distance:
            best_distance = distance
            best_id = idx

            if actual == target_score:
                break

    if best_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "There is no unused hint "
                "above your current score."
            )
        )

    actual_score = rank_score(
        similarities,
        best_id
    )

    return {
        "word": id_to_word[best_id],
        "score": actual_score,
        "cosine": round(
            float(
                similarities[best_id]
            ),
            6
        ),
        "hint_number": hint_number
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    redis_connected = False
    redis_health_error = redis_error

    if redis is not None:
        try:
            redis.set(
                "wordheat:health",
                "ok",
                ex=60
            )

            value = redis.get(
                "wordheat:health"
            )

            if isinstance(value, bytes):
                value = value.decode(
                    "utf-8"
                )

            redis_connected = (
                value == "ok"
            )

        except Exception as exc:
            redis_health_error = (
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "status": (
            "ok"
            if (
                vectors is not None
                and redis_connected
            )
            else "error"
        ),
        "version": "6.1",
        "vectors_loaded": (
            vectors is not None
        ),
        "startup_error": startup_error,
        "vector_file_exists": VFILE.exists(),
        "index_file_exists": IFILE.exists(),
        "vocabulary_size": len(
            word_to_id
        ),
        "dimensions": (
            int(vectors.shape[1])
            if vectors is not None
            else 0
        ),
        "redis_connected": redis_connected,
        "redis_error": redis_health_error,
        "endpoints": [
            "/api/game/start",
            "/api/game/guess",
            "/api/game/hint",
            "/api/game/end"
        ]
    }


# ============================================================
# START GAME
# ============================================================

@app.post("/api/game/start")
def start(req: StartRequest):
    require_model()
    require_redis()

    if not id_to_word:
        raise HTTPException(
            status_code=503,
            detail="Vocabulary is empty."
        )

    if req.hidden_word:
        hidden = clean(
            req.hidden_word
        )
    else:
        hidden = random.choice(
            id_to_word
        )

    if hidden not in word_to_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Hidden word is not in "
                "the prepared vocabulary."
            )
        )

    gid = uuid.uuid4().hex

    game = {
        "game_id": gid,
        "hidden_word": hidden,
        "hidden_id": int(
            word_to_id[hidden]
        ),
        "history": [],
        "guessed_words": [],
        "hints": [],
        "ended": False
    }

    save_game(
        gid,
        game
    )

    return {
        "game_id": gid,
        "backend_version": "6.1"
    }


# ============================================================
# GUESS
# ============================================================

@app.post("/api/game/guess")
def guess(req: GuessRequest):
    require_model()
    require_redis()

    game = get_game(
        req.game_id
    )

    if game["ended"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "This game has already ended."
            )
        )

    word = clean(
        req.word
    )

    if word not in word_to_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "That word is not in "
                "the WordHeat vocabulary."
            )
        )

    word_id = int(
        word_to_id[word]
    )

    similarities = hidden_similarities(
        game
    )

    current_score = rank_score_fast(
        similarities,
        word_id
    )

    item = {
        "word": word,
        "score": current_score,
        "cosine": round(
            float(
                similarities[word_id]
            ),
            6
        )
    }

    game.setdefault(
        "history",
        []
    ).append(
        item
    )

    game.setdefault(
        "guessed_words",
        []
    )

    if word not in game["guessed_words"]:
        game["guessed_words"].append(
            word
        )

    won = (
        word == game["hidden_word"]
    )

    if won:
        game["ended"] = True

    save_game(
        req.game_id,
        game
    )

    exclude_ids = [
        word_to_id[w]
        for w in game.get(
            "guessed_words",
            []
        )
        if w in word_to_id
    ]

    exclude_ids.append(
        game["hidden_id"]
    )

    nearest_words = nearest(
        word_id,
        8,
        exclude_ids
    )

    return {
        "guess": word,
        "score": current_score,
        "cosine": item["cosine"],
        "won": won,
        "nearest_words": nearest_words,
        "history": game["history"]
    }


# ============================================================
# HINT
# ============================================================

@app.post("/api/game/hint")
def hint(req: HintRequest):
    require_model()
    require_redis()

    game = get_game(
        req.game_id
    )

    if game["ended"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "This game has already ended."
            )
        )

    if len(
        game.get(
            "hints",
            []
        )
    ) >= 4:
        raise HTTPException(
            status_code=400,
            detail=(
                "All 4 hints have already been used."
            )
        )

    item = choose_hint(
        game
    )

    game.setdefault(
        "hints",
        []
    ).append(
        item["word"]
    )

    save_game(
        req.game_id,
        game
    )

    return {
        "hint": item,
        "hint_number": len(
            game["hints"]
        ),
        "max_hints": 4
    }


# ============================================================
# END GAME
# ============================================================

@app.post("/api/game/end")
def end_game(req: EndRequest):
    require_redis()

    game = get_game(
        req.game_id
    )

    game["ended"] = True

    save_game(
        req.game_id,
        game
    )

    return {
        "ended": True,
        "answer": game["hidden_word"],
        "history": game["history"]
    }


# ============================================================
# GET GAME STATE
# ============================================================

@app.get("/api/game/{game_id}")
def game_state(game_id: str):
    require_redis()

    game = get_game(
        game_id
    )

    return {
        "game_id": game_id,
        "history": game.get(
            "history",
            []
        ),
        "hints_used": len(
            game.get(
                "hints",
                []
            )
        ),
        "ended": game.get(
            "ended",
            False
        )
    }

