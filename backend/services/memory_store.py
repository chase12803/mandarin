"""Should we store? Small model decides from last turn; insert into Memory only when appropriate."""
from backend.models import db, Memory
from backend.services.models_config import get_model_info, get_memory_extractor_model_id
from backend.services.prompt_loader import load_prompt
from backend.services.settings_store import get_ai_memory_enabled
from backend.providers import base as providers_base

_MIN_FACT_LENGTH = 10


def _pick_small_model():
    """Prefer GPT-5 Nano; fall back to smallest available model per provider from models.yaml."""
    if get_model_info("openai/gpt-5-nano-2025-08-07"):
        return "openai/gpt-5-nano-2025-08-07"
    # Smallest per provider = last in each provider's list in models.yaml
    from backend.services.models_config import get_models_list
    by_provider = {}
    for m in get_models_list():
        if not m["available"]:
            continue
        by_provider[m["provider"]] = m["id"]
    for mid in (by_provider.get("openai"), by_provider.get("anthropic"), by_provider.get("google")):
        if mid and get_model_info(mid):
            return mid
    return None


def _build_existing_memories_text(user_content):
    """Return formatted text of relevant existing memories (RAG query)."""
    try:
        from backend.services.rag import query as rag_query
        hits = rag_query(user_content, top_k=8)
    except Exception:
        return "No existing memories."
    if not hits:
        return "No existing memories."
    lines = [f"- {content}" for (_id, content) in hits]
    return "\n".join(lines)


def _build_existing_context_text(context_ids):
    """Return formatted text of human context files for given ids (full content, no truncation)."""
    if not context_ids:
        return ""
    from backend.services.prompt_builder import _read_context
    parts = []
    for cid in context_ids:
        parsed = _read_context(cid)
        if not parsed:
            continue
        name, content = parsed
        text = (content or "").strip()
        if text:
            parts.append(f"- [{name}]: {text}")
    if not parts:
        return ""
    return "\n".join(parts)


def _get_all_contexts_text():
    """Return formatted text of ALL human context files (for deduplication checking)."""
    from backend.services.prompt_builder import list_contexts, _read_context
    try:
        all_contexts = list_contexts()
        parts = []
        for ctx in all_contexts:
            parsed = _read_context(ctx["id"])
            if not parsed:
                continue
            name, content = parsed
            text = (content or "").strip()
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else ""
    except Exception:
        return ""


def _is_obvious_non_fact(raw):
    """Reject obvious non-facts: single Yes/No, or ends with ?"""
    s = raw.strip()
    if s.endswith("?"):
        return True
    if s.upper() in ("YES", "NO"):
        return True
    return False


def _check_similarity_in_text(candidate_text, target_text, threshold=0.9):
    """Check if candidate_text has similarity >= threshold to target_text using embeddings.
    Returns True if similarity is high enough to consider it a duplicate."""
    if not target_text or not candidate_text:
        return False
    try:
        from backend.services.rag import _get_embed_fn
        import numpy as np
        embed_fn = _get_embed_fn()
        candidate_vec = np.array(embed_fn([candidate_text])[0])
        target_vec = np.array(embed_fn([target_text])[0])
        # Compute cosine similarity
        similarity = np.dot(candidate_vec, target_vec) / (np.linalg.norm(candidate_vec) * np.linalg.norm(target_vec))
        return similarity >= threshold
    except Exception:
        # If embedding check fails, fall back to simple substring check for exact matches
        candidate_lower = candidate_text.lower().strip()
        target_lower = target_text.lower()
        # Check if candidate appears verbatim in target (allowing for some whitespace differences)
        if candidate_lower in target_lower:
            return True
        # Check if target appears verbatim in candidate
        if target_lower in candidate_lower:
            return True
        return False


def _check_similarity_in_long_text(candidate_text, target_text, threshold=0.9):
    """Check if candidate_text is similar to any chunk of target_text (for long context files).
    Uses chunked embedding comparison so we don't truncate long contexts and miss duplicates."""
    if not target_text or not candidate_text:
        return False
    try:
        from backend.services.rag import chunk_text_for_embedding, _get_embed_fn
        import numpy as np
        chunks = chunk_text_for_embedding(target_text)
        if not chunks:
            return False
        embed_fn = _get_embed_fn()
        candidate_vec = np.array(embed_fn([candidate_text])[0])
        norm_c = np.linalg.norm(candidate_vec)
        if norm_c == 0:
            return False
        for chunk in chunks:
            if not chunk.strip():
                continue
            chunk_vec = np.array(embed_fn([chunk])[0])
            similarity = np.dot(candidate_vec, chunk_vec) / (norm_c * np.linalg.norm(chunk_vec))
            if similarity >= threshold:
                return True
        return False
    except Exception:
        candidate_lower = candidate_text.lower().strip()
        target_lower = target_text.lower()
        if candidate_lower in target_lower or target_lower in candidate_lower:
            return True
        return False


def extract_and_store(user_content, assistant_content, app, context_ids=None):
    """Run in background: ask small model if there is a fact worth storing; if yes and not duplicate, insert Memory."""
    if not get_ai_memory_enabled():
        return
    model_id = get_memory_extractor_model_id() or _pick_small_model()
    if not model_id:
        return

    # Phase 1: Data gathering (full content; no truncation)
    user_text = (user_content or "")
    assistant_text = (assistant_content or "")
    existing_memories_text = _build_existing_memories_text(user_content or "")
    existing_context_text = _build_existing_context_text(context_ids or [])

    # Phase 2: Build prompt from template
    template = load_prompt("memory_extraction")
    if not template:
        return
    existing_context_block = ""
    if existing_context_text:
        existing_context_block = (
            "\n\nExisting context (do not duplicate - this information is already available every time):\n"
            + existing_context_text
        )
    system_content = template.replace("{{EXISTING_MEMORIES}}", existing_memories_text).replace(
        "{{EXISTING_CONTEXT}}", existing_context_block
    ).replace("{{USER_TEXT}}", user_text).replace("{{ASSISTANT_TEXT}}", assistant_text)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Reply with exactly one line: NOTHING or the single fact. No explanation."},
    ]
    try:
        with app.app_context():
            parts = list(providers_base.generate(messages, model_id, stream=True))
            raw = "".join(parts).strip()

            # Phase 3: Response parsing (plain-text)
            if not raw:
                return
            if len(raw) < _MIN_FACT_LENGTH:
                return
            if "NOTHING" in raw.upper():
                return
            if _is_obvious_non_fact(raw):
                return

            candidate_fact = raw.strip()

            # Phase 4: Dedupe check — skip if candidate is too similar to existing memories or appears in ANY context
            # Check 1: Similarity to existing memories (similarity > 0.9)
            # top_k=5 is sufficient since duplicates with >0.9 similarity will be at the top of results
            try:
                from backend.services.rag import query as rag_query
                dup_hits = rag_query(candidate_fact, top_k=5, min_similarity=0.90)
                if dup_hits:
                    return  # treat as duplicate, do not save
            except Exception:
                pass

            # Check 2: Check if candidate appears in ANY context (not just current ones) - chunked comparison for long contexts
            try:
                from backend.services.prompt_builder import list_contexts, _read_context
                all_contexts = list_contexts()
                for ctx in all_contexts:
                    parsed = _read_context(ctx["id"])
                    if not parsed:
                        continue
                    name, content = parsed
                    text = (content or "").strip()
                    if text and _check_similarity_in_long_text(candidate_fact, text, threshold=0.9):
                        return  # duplicate found in a context file
            except Exception:
                pass

            # Phase 5: Persist and index
            mem = Memory(content=candidate_fact, tags=[])
            db.session.add(mem)
            db.session.commit()
            try:
                from backend.services.rag import add_memory
                add_memory(mem.id, mem.content)
            except Exception as e:
                print(f"RAG add_memory failed for extracted memory id={mem.id}: {e}")
    except Exception:
        pass
