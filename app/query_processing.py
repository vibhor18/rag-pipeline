"""
Query processing pipeline:
  1. PII detection  — refuse queries containing personal identifiable info
  2. Intent detection — classify as greeting / chitchat / rag / refusal
  3. Query transformation — rewrite for better retrieval
"""
from __future__ import annotations
 
import json
import logging
import re
 
from app.config import PII_PATTERNS
from app.mistral_client import chat_completion
from app.models import IntentResult
 
logger = logging.getLogger(__name__)
 
 
# ──────────────────────────────────────────────
# PII detection
# ──────────────────────────────────────────────
 
def detect_pii(text: str) -> bool:
    """Check if the query contains PII patterns (SSN, credit card, etc.)."""
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return True
    return False
 
 
# ──────────────────────────────────────────────
# Intent detection
# ──────────────────────────────────────────────
 
_INTENT_PROMPT = """You are an intent classifier for a document Q&A system.
Given the user's message, classify it into EXACTLY ONE of these intents:
 
- "greeting": casual greetings like "hello", "hi", "hey", "good morning"
- "chitchat": general conversation not related to documents (e.g., "what's the weather", "tell me a joke")
- "rag": the user is asking a question that should be answered using the knowledge base documents
- "refusal": the query asks for medical diagnoses, legal advice, or contains harmful content
 
Respond with ONLY a JSON object:
{"intent": "<one of: greeting, chitchat, rag, refusal>", "reason": "<brief explanation>"}
 
User message: """
 
 
async def detect_intent(query: str) -> IntentResult:
    """
    Detect the intent of a user query.
    Uses PII regex first (fast), then LLM classification.
    """
    # Fast PII check
    if detect_pii(query):
        return IntentResult(
            intent="refusal",
            reason="Query appears to contain personal identifiable information.",
            pii_detected=True,
        )
 
    try:
        response = await chat_completion(
            messages=[{"role": "user", "content": _INTENT_PROMPT + query}],
            temperature=0.0,
            max_tokens=150,
            json_mode=True,
        )
        data = json.loads(response)
        intent = data.get("intent", "rag")
        reason = data.get("reason", "")
 
        if intent not in ("greeting", "chitchat", "rag", "refusal"):
            intent = "rag"  # default to RAG if classification is unclear
 
        return IntentResult(intent=intent, reason=reason)
    except Exception as e:
        logger.warning("Intent detection failed: %s – defaulting to RAG", e)
        return IntentResult(intent="rag", reason="Fallback: classification failed")
 
 
# ──────────────────────────────────────────────
# Query transformation
# ──────────────────────────────────────────────
 
_TRANSFORM_PROMPT = """You are a search query optimizer. Given a user's question about their documents, rewrite it into a better search query that will retrieve the most relevant passages.
 
Rules:
- Expand abbreviations and acronyms
- Add relevant synonyms
- Remove filler words
- Keep the core meaning
- Output ONLY the rewritten query, nothing else
 
User question: """
 
 
async def transform_query(query: str) -> str:
    """
    Rewrite the user's query for better retrieval.
    Expands abbreviations, adds synonyms, removes noise.
    """
    try:
        rewritten = await chat_completion(
            messages=[{"role": "user", "content": _TRANSFORM_PROMPT + query}],
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = rewritten.strip().strip('"').strip("'")
        if len(rewritten) < 3 or len(rewritten) > 500:
            return query
        return rewritten
    except Exception as e:
        logger.warning("Query transformation failed: %s – using original", e)
        return query