"""Rubric-prompted LLM grader. The backend is swappable: anything with generate(prompt) -> (text, info) works.

The rubrics come from Kaggle's Essay_Set_Descriptions.zip (unzipped into data/), read at runtime and never committed.
In prompts 1, 7 and 8 the final score is the sum of two raters, so the LLM gives one rater's score, which we double.
"""
import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path

from aes.data import DATA_DIR, ROOT

# Scale the LLM is asked to use (one rater), and the multiplier back to the final score.
LLM_SCALE = {1: (1, 6, 2), 2: (1, 6, 1), 3: (0, 3, 1), 4: (0, 3, 1), 5: (0, 4, 1), 6: (0, 4, 1), 7: (0, 15, 2), 8: (0, 30, 2)}
CACHE_DIR = ROOT / "llm_cache"


def load_rubrics(folder=None) -> dict:
    """{prompt: full text of 'Essay Set #k--ReadMeFirst.docx'} (task prompt, source text, rubric)."""
    folder = Path(folder or DATA_DIR / "Essay_Set_Descriptions")
    out = {}
    for p in range(1, 9):
        xml = zipfile.ZipFile(next(folder.glob(f"Essay Set #{p}--*.docx"))).read("word/document.xml").decode("utf-8")
        paras = (re.sub(r"<[^>]+>", "", m) for m in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S))
        out[p] = "\n".join(t for t in paras if t.strip())
    return out


def build_prompt(prompt_id: int, rubric: str, essay: str, examples=()) -> str:
    lo, hi, _ = LLM_SCALE[prompt_id]
    note = " Score domain 1 (Writing Applications) only." if prompt_id == 2 else ""
    note += " Give ONE rater's score (do not double it)." if LLM_SCALE[prompt_id][2] == 2 else ""
    parts = [
        "You are an experienced teacher grading a student essay for a standardized writing assessment.",
        f"Below is the official description of the task and its scoring rubric.\n\n<task_and_rubric>\n{rubric}\n</task_and_rubric>",
        f"Score the student essay as a single rater would, using an integer from {lo} to {hi}.{note}",
        "Names, places and similar details were anonymized as tokens like @PERSON1 or @CAPS2; ignore them.",
    ]
    for i, (text, score) in enumerate(examples, 1):
        parts.append(f"<example_{i}>\n<essay>\n{text}\n</essay>\n<score>{score}</score>\n</example_{i}>")
    parts.append(f"<essay>\n{essay}\n</essay>")
    parts.append('Respond with JSON only: {"score": <integer>, "reason": "<one sentence>"}')
    return "\n\n".join(parts)


def parse_score(text: str, lo: int, hi: int):
    """Integer score from the model's reply, or None if there is no valid one in range."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.S)
    candidates = []
    if m:
        try:
            candidates.append(json.loads(m.group(0)).get("score"))
        except (json.JSONDecodeError, AttributeError):
            pass
    m = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text, flags=re.I)
    if m:
        candidates.append(m.group(1))
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        if v.is_integer() and lo <= v <= hi:
            return int(v)
    return None


class Cache:
    """Responses on disk keyed by hash(model + prompt), so reruns cost nothing. Holds essay-derived text: gitignored."""

    def __init__(self, model: str):
        CACHE_DIR.mkdir(exist_ok=True)
        self.path = CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.jsonl"
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.data[rec["key"]] = rec
        self.model = model

    def key(self, prompt):
        return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()

    def get(self, prompt):
        return self.data.get(self.key(prompt))

    def put(self, prompt, rec):
        rec = {"key": self.key(prompt), **rec}
        self.data[rec["key"]] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


class GeminiBackend:
    """Gemini API (free tier). Paces calls to stay under the free rate limit and retries on 429s."""

    def __init__(self, model: str, min_interval: float = 4.0, thinking_budget=None):
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types

        load_dotenv(ROOT / ".env")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing: put it in .env (see .env.example)")
        # JSON-only replies need the v1beta API
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1beta"))
        self.types, self.model = types, model
        self.min_interval, self.thinking_budget, self._last = min_interval, thinking_budget, 0.0

    def generate(self, prompt: str):
        cfg = {"temperature": 0.0, "response_mime_type": "application/json"}
        if self.thinking_budget is not None:
            cfg["thinking_config"] = self.types.ThinkingConfig(thinking_budget=self.thinking_budget)
        for attempt in range(6):
            time.sleep(max(0.0, self.min_interval - (time.time() - self._last)))
            self._last = t0 = time.time()
            try:
                r = self.client.models.generate_content(model=self.model, contents=prompt,
                                                        config=self.types.GenerateContentConfig(**cfg))
            except Exception as e:  # SDK raises ClientError/ServerError; retry only rate limits and outages
                msg = str(e)
                if "PerDay" in msg:  # daily free quota used up: waiting won't help, rerun tomorrow (cache resumes)
                    raise RuntimeError(f"daily free-tier quota reached for {self.model}") from e
                if attempt < 5 and any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE")):
                    wait = 30 * (attempt + 1)
                    print(f"  rate-limited/unavailable, waiting {wait}s")
                    time.sleep(wait)
                    continue
                raise
            u = r.usage_metadata
            return r.text or "", {"latency_s": time.time() - t0,
                                  "input_tokens": getattr(u, "prompt_token_count", None),
                                  "output_tokens": (getattr(u, "candidates_token_count", 0) or 0)
                                  + (getattr(u, "thoughts_token_count", 0) or 0)}
