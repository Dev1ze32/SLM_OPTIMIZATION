"""
Central configuration for the routing PoC.

Edit values here to swap components without touching pipeline logic —
e.g. pointing GENERATOR_MODEL at a local 4-bit Llama 3.2 3B / Qwen3 4B
endpoint later, or retuning thresholds against your labeled eval set.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root into os.environ, if present

# --- Gate 3: answer generation ---
GENERATOR_MODEL = "gpt-4o-mini"
GENERATOR_TEMPERATURE = 0.0
GENERATOR_MAX_TOKENS = 400
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- Shared embedding model (Gate 1 scope check + Gate 2 dense retrieval) ---
# Small pretrained sentence-embedding model. No training happens here:
# the model already knows how to embed text: we just do a nearest-
# neighbor cosine-similarity lookup against labeled exemplar queries.
# bge-small-en-v1.5 is retrieval-tuned (384-dim, 512-token max), unlike
# all-MiniLM-L6-v2 which is tuned for sentence similarity. The query is
# embedded once and reused by both gates.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Gate 1 must never allocate VRAM: the 4-bit SLM needs all 8 GB. Left
# unset, sentence-transformers auto-selects cuda whenever it is available.
EMBEDDING_DEVICE = "cpu"
# Caps torch's CPU thread pool so Gate 1 doesn't grab every core by
# default; keeps CPU-side latency numbers comparable across runs.
EMBEDDING_CPU_THREADS = 4
# NOTE: recalibrate after the switch to bge-small — its cosine similarity
# distribution runs higher than all-MiniLM-L6-v2, so 0.50 is not transferable.
SCOPE_SIMILARITY_THRESHOLD = 0.50  # placeholder; tune against labeled eval set

# --- Gate 2: hybrid retrieval ---
# Lexical retrieval
BM25_TOP_K = 3
BM25_COVERAGE_THRESHOLD = 1.0  # tune against your labeled eval set

# Dense retrieval
DENSE_SCORE_THRESHOLD = 0.65  # tune against your labeled eval set

# Rank fusion
RRF_K = 60  # reciprocal rank fusion constant; tune against your labeled eval set

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")
SCOPE_EXEMPLARS_PATH = os.path.join(DATA_DIR, "scope_exemplars.json")

# --- Prompt contract (from prompt_and_routing_architecture.md) ---
SYSTEM_PROMPT = """You are an offline university-helpdesk prototype.
Answer only using the supplied LMS evidence.
Do not invent policies, dates, fees, requirements, contacts, or procedures.
Give a concise English answer and cite the supplied document title and page or section.
If the evidence does not support an answer, do not guess."""

# --- Predefined non-answer messages (output contract) ---
SCOPE_MESSAGE = (
    "This assistant only handles university helpdesk questions "
    "(policies, procedures, and information from the student handbook and LMS)."
)
REFERRAL_MESSAGE = (
    "I could not verify this from the selected LMS materials. "
    "Please contact the appropriate university office."
)