"""
Response generation with:
  - Intent-based prompt templates (answer shaping)
  - Citation-aware generation
  - Post-hoc hallucination filtering
"""
from __future__ import annotations
 
import json
import logging
 
from app.config import SIMILARITY_THRESHOLD
from app.mistral_client import chat_completion
from app.models import Citation, QueryResponse, SearchResult
 
logger = logging.getLogger(__name__)
 
 
# ──────────────────────────────────────────────
# Prompt templates by intent
# ──────────────────────────────────────────────
 
_RAG_TEMPLATE = """You are a helpful document assistant. Answer the user's question based ONLY on the provided context passages.
 
RULES:
- Only use information from the provided context.
- If the context doesn't contain enough information, say "I don't have enough information in the documents to answer this."
- Cite your sources by referencing [Source N] where N corresponds to the passage number.
- Be concise and accurate.
 
CONTEXT PASSAGES:
{context}
 
USER QUESTION: {question}
 
ANSWER:"""
 
_LIST_TEMPLATE = """You are a helpful document assistant. The user wants a list or enumeration based on the provided context.
 
RULES:
- Extract items from the context and present them as a clean numbered or bulleted list.
- Only include items supported by the context.
- Cite sources as [Source N].
- If the context is insufficient, say so.
 
CONTEXT PASSAGES:
{context}
 
USER QUESTION: {question}
 
Respond with a well-structured list:"""
 
_COMPARISON_TEMPLATE = """You are a helpful document assistant. The user wants a comparison or analysis based on the provided context.
 
RULES:
- Structure your response as a comparison with clear categories.
- Use a table format if comparing multiple attributes.
- Only use information from the provided context.
- Cite sources as [Source N].
 
CONTEXT PASSAGES:
{context}
 
USER QUESTION: {question}
 
Provide a structured comparison:"""
 
_GREETING_RESPONSE = (
    "Hello! I'm your document assistant. You can upload PDF files and ask me "
    "questions about their content. How can I help you today?"
)
 
_CHITCHAT_RESPONSE = (
    "I'm designed to help you with questions about your uploaded documents. "
    "Please upload some PDF files and ask me questions about their content!"
)
 
_REFUSAL_RESPONSE = (
    "I'm sorry, but I can't process this request. "
    "If your query contained personal information (SSN, credit card numbers, etc.), "
    "please avoid sharing sensitive data. "
    "For medical or legal questions, please consult a qualified professional."
)
 
_NO_DOCS_RESPONSE = (
    "No documents have been uploaded yet. Please upload PDF files first, "
    "then I can answer questions about their content."
)
 
 
# ──────────────────────────────────────────────
# Template selection
# ──────────────────────────────────────────────
 
def _select_template(question: str) -> str:
    """Pick the best prompt template based on question phrasing."""
    q_lower = question.lower()
 
    list_signals = ["list", "enumerate", "what are", "name all", "give me all", "how many"]
    comparison_signals = ["compare", "difference", "versus", "vs", "contrast", "pros and cons"]
 
    if any(s in q_lower for s in comparison_signals):
        return _COMPARISON_TEMPLATE
    if any(s in q_lower for s in list_signals):
        return _LIST_TEMPLATE
    return _RAG_TEMPLATE
 
 
def _build_context(results: list[SearchResult]) -> str:
    """Format search results into a context string for the LLM."""
    parts = []
    for i, sr in enumerate(results, 1):
        parts.append(
            f"[Source {i}] (File: {sr.chunk.filename}, Page: {sr.chunk.page_number})\n"
            f"{sr.chunk.text}\n"
        )
    return "\n---\n".join(parts)
 
 
# ──────────────────────────────────────────────
# Hallucination filter
# ──────────────────────────────────────────────
 
_HALLUCINATION_CHECK_PROMPT = """You are a fact-checker. Given an answer and supporting evidence passages, check each sentence of the answer.
 
For each sentence, determine if it is:
- SUPPORTED: directly backed by the evidence
- UNSUPPORTED: makes a claim not found in the evidence
- NEUTRAL: transitional or structural text (greetings, "Based on the documents...")
 
Respond with ONLY a JSON object:
{{"sentences": [{{"text": "sentence text", "verdict": "SUPPORTED|UNSUPPORTED|NEUTRAL"}}], "has_hallucinations": true/false}}
 
EVIDENCE:
{evidence}
 
ANSWER TO CHECK:
{answer}"""
 
 
async def _check_hallucinations(
    answer: str, context: str
) -> tuple[str, bool]:
    """
    Post-hoc evidence check: scan answer sentences for unsupported claims.
    Returns (filtered_answer, had_hallucinations).
    """
    try:
        response = await chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": _HALLUCINATION_CHECK_PROMPT.format(
                        evidence=context, answer=answer
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=1000,
            json_mode=True,
        )
        data = json.loads(response)
        sentences = data.get("sentences", [])
        has_hallucinations = data.get("has_hallucinations", False)
 
        if not has_hallucinations:
            return answer, False
 
        # Filter out unsupported sentences
        kept = [
            s["text"]
            for s in sentences
            if s.get("verdict") in ("SUPPORTED", "NEUTRAL")
        ]
        if not kept:
            return (
                "I found some relevant documents, but I couldn't generate a "
                "fully supported answer. Please try rephrasing your question.",
                True,
            )
        filtered = " ".join(kept)
        return filtered, True
    except Exception as e:
        logger.warning("Hallucination check failed: %s", e)
        return answer, False
 
 
# ──────────────────────────────────────────────
# Main generation pipeline
# ──────────────────────────────────────────────
 
async def generate_response(
    question: str,
    intent: str,
    search_results: list[SearchResult],
    transformed_query: str = "",
    has_documents: bool = True,
) -> QueryResponse:
    """
    Generate a final response based on intent and search results.
    Handles all intent types and applies post-processing.
    """
    # Non-RAG intents
    if intent == "greeting":
        return QueryResponse(answer=_GREETING_RESPONSE, intent=intent)
 
    if intent == "chitchat":
        return QueryResponse(answer=_CHITCHAT_RESPONSE, intent=intent)
 
    if intent == "refusal":
        return QueryResponse(answer=_REFUSAL_RESPONSE, intent=intent)
 
    if not has_documents:
        return QueryResponse(answer=_NO_DOCS_RESPONSE, intent=intent)
 
    # ── RAG intent ──
 
    # Check if we have results above the similarity threshold
    relevant_results = [
        sr for sr in search_results if sr.score >= SIMILARITY_THRESHOLD
    ]
 
    if not relevant_results:
        return QueryResponse(
            answer=(
                "I don't have sufficient evidence in the uploaded documents to "
                "answer this question. The retrieved passages didn't meet the "
                "relevance threshold. Please try rephrasing or upload more "
                "relevant documents."
            ),
            intent=intent,
            query_transformed=transformed_query,
            warning="No passages met the similarity threshold.",
        )
 
    # Build context and select template
    context = _build_context(relevant_results)
    template = _select_template(question)
    prompt = template.format(context=context, question=question)
 
    # Generate answer
    answer = await chat_completion(
        messages=[{"role": "user", "content": prompt}],
    )
 
    # Hallucination filter
    filtered_answer, had_hallucinations = await _check_hallucinations(
        answer, context
    )
 
    # Build citations
    citations = [
        Citation(
            chunk_id=sr.chunk.id,
            filename=sr.chunk.filename,
            page_number=sr.chunk.page_number,
            text_snippet=sr.chunk.text[:200] + ("..." if len(sr.chunk.text) > 200 else ""),
            relevance_score=round(sr.score, 4),
        )
        for sr in relevant_results
    ]
 
    warning = None
    if had_hallucinations:
        warning = "Some unsupported claims were filtered from the response."
 
    return QueryResponse(
        answer=filtered_answer,
        intent=intent,
        citations=citations,
        query_transformed=transformed_query,
        warning=warning,
    )
 