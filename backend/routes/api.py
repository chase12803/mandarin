"""API blueprint: chats, models, streaming, contexts, memory."""
import base64
import re
import json
import threading
from datetime import datetime
from pathlib import Path

import yaml
from flask import Blueprint, request, jsonify, Response, stream_with_context, current_app

import config
from backend.models import db, Chat, Message, Memory
from backend.services.message_content import message_to_llm_content
from backend.services.file_extraction import extract_attachments
from backend.services.prompt_builder import (
    Rule,
    Command,
    build_system_message,
    get_command_body_if_invoked,
    list_contexts,
    load_rules,
    load_commands,
    resolve_active_rules,
)
from backend.services.prompt_builder import _context_name_from_first_line
from backend.services.prompt_loader import load_prompt
from backend.services.models_config import (
    get_models_list,
    get_model_info,
    get_chat_namer_model_id,
    get_default_model_id,
    set_default_model,
)
from backend.services.settings_store import (
    get_settings_for_api,
    get_default_web_search_mode,
    get_ai_memory_enabled,
    update_settings,
    invalidate_provider_clients,
)
from backend.services.web_search_mode import (
    WEB_SEARCH_MODE_OFF,
    command_web_search_mode_for_api,
    is_web_search_enabled,
    is_command_web_search_mode_explicit,
    mode_from_legacy_enabled,
    parse_web_search_mode,
    resolve_command_web_search_mode,
    resolve_chat_web_search_mode,
)

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/models", methods=["GET"])
def list_models():
    refresh_raw = (request.args.get("refresh") or "").strip().lower()
    force_refresh = refresh_raw in ("1", "true", "yes")
    return jsonify(get_models_list(force_refresh=force_refresh))


@api_bp.route("/settings", methods=["GET"])
def get_settings():
    settings = get_settings_for_api()
    settings["default_model"] = get_default_model_id()
    return jsonify(settings)


@api_bp.route("/settings", methods=["PUT"])
def put_settings():
    data = request.get_json() or {}
    if "default_model" in data:
        try:
            set_default_model((data.get("default_model") or "").strip() or None)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    updates = {}
    if "api_keys" in data and isinstance(data["api_keys"], dict):
        updates["api_keys"] = {}
        for k, v in data["api_keys"].items():
            if k in ("openai", "anthropic", "google", "tavily") and v is not None:
                s = str(v).strip()
                updates["api_keys"][k] = s  # allow empty string for removal
    if "default_web_search_mode" in data:
        mode = parse_web_search_mode(data.get("default_web_search_mode"))
        if mode is None:
            return jsonify({"error": "default_web_search_mode must be one of: off, native, tavily"}), 400
        updates["default_web_search_mode"] = mode
    if "ai_memory_enabled" in data:
        updates["ai_memory_enabled"] = bool(data.get("ai_memory_enabled"))
    update_settings(updates)
    if "api_keys" in updates:
        invalidate_provider_clients()
    settings = get_settings_for_api()
    settings["default_model"] = get_default_model_id()
    return jsonify(settings)


def _safe_context_id(id_str):
    """Allow only alphanumeric, hyphen, underscore."""
    if not id_str or not isinstance(id_str, str):
        return None
    if not re.match(r"^[a-zA-Z0-9_-]+$", id_str):
        return None
    return id_str


def _safe_rule_or_command_id(id_str):
    """Allow only alphanumeric, hyphen, underscore for rule/command ids."""
    if not id_str or not isinstance(id_str, str):
        return None
    if not re.match(r"^[a-zA-Z0-9_-]+$", id_str):
        return None
    return id_str


def _mode_from_payload(data, *, default_mode=WEB_SEARCH_MODE_OFF):
    """
    Read web search mode from payload, with legacy boolean compatibility.
    Returns (mode, error_str_or_none).
    """
    if "web_search_mode" in data:
        raw = data.get("web_search_mode")
        mode = parse_web_search_mode(raw if raw is not None else WEB_SEARCH_MODE_OFF)
        if mode is None:
            return None, "web_search_mode must be one of: off, native, tavily"
        return mode, None
    if "web_search_enabled" in data:
        return mode_from_legacy_enabled(bool(data.get("web_search_enabled"))), None
    return default_mode, None


def _request_mode_override_from_payload(data):
    """
    Read optional one-time request web search mode override.
    Returns (mode_or_none, error_str_or_none).
    """
    if "web_search_mode" not in data:
        return None, None
    mode = parse_web_search_mode(data.get("web_search_mode"))
    if mode is None:
        return None, "web_search_mode must be one of: off, native, tavily"
    return mode, None


@api_bp.route("/contexts", methods=["GET"])
def list_contexts_route():
    return jsonify(list_contexts())


@api_bp.route("/contexts/<id>", methods=["GET"])
def get_context(id):
    safe = _safe_context_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    path = config.CONTEXTS_DIR / f"{safe}.md"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return Response(path.read_text(encoding="utf-8"), mimetype="text/markdown")


@api_bp.route("/contexts/<id>", methods=["PUT"])
def put_context(id):
    safe = _safe_context_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    path = config.CONTEXTS_DIR / f"{safe}.md"
    body = request.get_data(as_text=True) or ""
    path.write_text(body, encoding="utf-8")
    name = _context_name_from_first_line(body) if body else safe
    return jsonify({"id": safe, "name": name})


@api_bp.route("/contexts/<id>", methods=["DELETE"])
def delete_context(id):
    safe = _safe_context_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    path = config.CONTEXTS_DIR / f"{safe}.md"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    path.unlink()
    return "", 204


# ----- Rules (file-backed) -----
@api_bp.route("/rules", methods=["GET"])
def list_rules_route():
    rules = load_rules()
    out = []
    for r in rules.values():
        out.append(
            {
                "id": r.id,
                "name": r.name,
                "always_on": r.always_on,
                "tags": r.tags,
            }
        )
    # Sort by name for stable UI.
    out.sort(key=lambda x: x["name"].lower())
    return jsonify(out)


@api_bp.route("/rules/<id>", methods=["GET"])
def get_rule(id):
    safe = _safe_rule_or_command_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    rules = load_rules()
    r = rules.get(safe)
    if not r:
        return jsonify({"error": "not found"}), 404
    return jsonify(
        {
            "id": r.id,
            "name": r.name,
            "always_on": r.always_on,
            "tags": r.tags,
            "body": r.body,
        }
    )


@api_bp.route("/rules/<id>", methods=["PUT"])
def put_rule(id):
    safe = _safe_rule_or_command_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    data = request.get_json() or {}
    name = (data.get("name") or "").strip() or safe
    always_on = bool(data.get("always_on", False))
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    body = (data.get("body") or "").rstrip()
    meta = {
        "id": safe,
        "name": name,
        "always_on": always_on,
        "tags": tags,
    }
    text = f"---\n{yaml.safe_dump(meta, sort_keys=False).strip()}\n---\n\n{body}\n"
    path = config.RULES_DIR / f"{safe}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    # Refresh cache by reloading.
    _ = load_rules()
    return jsonify(
        {
            "id": safe,
            "name": name,
            "always_on": always_on,
            "tags": tags,
        }
    )


@api_bp.route("/rules/<id>", methods=["DELETE"])
def delete_rule(id):
    safe = _safe_rule_or_command_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    path = config.RULES_DIR / f"{safe}.md"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    path.unlink()
    return "", 204


# ----- Commands (file-backed) -----
@api_bp.route("/commands", methods=["GET"])
def list_commands_route():
    cmds = load_commands()
    out = []
    for c in cmds.values():
        web_search_mode = command_web_search_mode_for_api(c)
        out.append(
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "tags": c.tags,
                "web_search_enabled": is_web_search_enabled(web_search_mode),
                "web_search_mode": web_search_mode,
                "web_search_mode_explicit": is_command_web_search_mode_explicit(c),
            }
        )
    out.sort(key=lambda x: x["name"].lower())
    return jsonify(out)


@api_bp.route("/commands/<id>", methods=["GET"])
def get_command(id):
    safe = _safe_rule_or_command_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    cmds = load_commands()
    c = cmds.get(safe)
    if not c:
        return jsonify({"error": "not found"}), 404
    out = {
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "tags": c.tags,
        "body": c.body,
    }
    if c.task is not None:
        out["task"] = c.task
    if c.success_criteria is not None:
        out["success_criteria"] = c.success_criteria
    if c.guidelines is not None:
        out["guidelines"] = c.guidelines
    if getattr(c, "context_ids", None) is not None:
        out["context_ids"] = c.context_ids
    web_search_mode = command_web_search_mode_for_api(c)
    out["web_search_enabled"] = is_web_search_enabled(web_search_mode)
    out["web_search_mode"] = web_search_mode
    out["web_search_mode_explicit"] = is_command_web_search_mode_explicit(c)
    return jsonify(out)


@api_bp.route("/commands/<id>", methods=["PUT"])
def put_command(id):
    safe = _safe_rule_or_command_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    data = request.get_json() or {}
    name = (data.get("name") or "").strip() or safe
    description = (data.get("description") or "").strip()
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    # Accept structured sections or raw body (backward compatibility)
    task = (data.get("task") or "").strip()
    success_criteria = (data.get("success_criteria") or "").strip()
    guidelines = (data.get("guidelines") or "").strip()
    body = (data.get("body") or "").rstrip()
    if "task" in data or "success_criteria" in data or "guidelines" in data:
        body = f"## Task\n\n{task}\n\n## Success Criteria\n\n{success_criteria}\n\n## Guidelines\n\n{guidelines}\n"
    context_ids = data.get("context_ids")
    if context_ids is not None and not isinstance(context_ids, list):
        context_ids = []
    if context_ids is not None:
        context_ids = [str(x) for x in context_ids if x and re.match(r"^[a-zA-Z0-9_-]+$", str(x))]
    web_search_mode, err = _mode_from_payload(data, default_mode=WEB_SEARCH_MODE_OFF)
    if err:
        return jsonify({"error": err}), 400
    web_search_enabled = is_web_search_enabled(web_search_mode)
    meta = {
        "id": safe,
        "name": name,
        "description": description,
        "tags": tags,
        "web_search_mode": web_search_mode,
        "web_search_enabled": web_search_enabled,
    }
    if context_ids is not None:
        meta["context_ids"] = context_ids
    text = f"---\n{yaml.safe_dump(meta, sort_keys=False).strip()}\n---\n\n{body}\n"
    path = config.COMMANDS_DIR / f"{safe}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _ = load_commands()
    return jsonify(
        {
            "id": safe,
            "name": name,
            "description": description,
            "tags": tags,
            "web_search_mode": web_search_mode,
            "web_search_enabled": web_search_enabled,
            "web_search_mode_explicit": True,
        }
    )


@api_bp.route("/commands/<id>", methods=["DELETE"])
def delete_command(id):
    safe = _safe_rule_or_command_id(id)
    if not safe:
        return jsonify({"error": "invalid id"}), 400
    path = config.COMMANDS_DIR / f"{safe}.md"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    path.unlink()
    return "", 204


# ----- Memory (LLM-generated) -----
@api_bp.route("/memory", methods=["GET"])
def list_memory():
    tag = request.args.get("tag")
    mems = Memory.query.order_by(Memory.created_at.desc()).all()
    if tag:
        mems = [m for m in mems if m.tags and tag in m.tags]
    return jsonify([m.to_dict() for m in mems])


@api_bp.route("/memory", methods=["POST"])
def create_memory():
    data = request.get_json() or {}
    content = (data.get("content") or "").strip()
    tags = data.get("tags")
    if not content:
        return jsonify({"error": "content is required"}), 400
    if tags is not None and not isinstance(tags, list):
        tags = []
    mem = Memory(content=content, tags=tags or [])
    db.session.add(mem)
    db.session.commit()
    try:
        from backend.services.rag import add_memory
        add_memory(mem.id, mem.content)
    except Exception as e:
        print(f"RAG add_memory failed for new memory id={mem.id}: {e}")
    return jsonify(mem.to_dict()), 201


@api_bp.route("/memory/<int:mem_id>", methods=["PATCH"])
def update_memory(mem_id):
    mem = Memory.query.get_or_404(mem_id)
    data = request.get_json() or {}
    if "content" in data:
        mem.content = (data["content"] or "").strip()
    if "tags" in data:
        mem.tags = data["tags"] if isinstance(data["tags"], list) else mem.tags
    db.session.commit()
    try:
        from backend.services.rag import add_memory, delete_memory as rag_delete
        rag_delete(mem_id)
        add_memory(mem.id, mem.content)
    except Exception as e:
        print(f"RAG update failed for memory id={mem_id}: {e}")
    return jsonify(mem.to_dict())


@api_bp.route("/memory/<int:mem_id>", methods=["DELETE"])
def delete_memory_route(mem_id):
    mem = Memory.query.get_or_404(mem_id)
    db.session.delete(mem)
    db.session.commit()
    try:
        from backend.services.rag import delete_memory
        delete_memory(mem_id)
    except Exception:
        pass
    return "", 204


@api_bp.route("/chats", methods=["GET"])
def list_chats():
    chats = Chat.query.order_by(Chat.updated_at.desc()).all()
    return jsonify([c.to_dict() for c in chats])


@api_bp.route("/chats", methods=["POST"])
def create_chat():
    data = request.get_json() or {}
    context_ids = data.get("context_ids", [])
    default_mode = get_default_web_search_mode()
    web_search_mode, err = _mode_from_payload(data, default_mode=default_mode)
    if err:
        return jsonify({"error": err}), 400
    chat = Chat(
        title="New chat",
        context_ids=context_ids,
        web_search_enabled=is_web_search_enabled(web_search_mode),
        web_search_mode=web_search_mode,
    )
    db.session.add(chat)
    db.session.commit()
    return jsonify(chat.to_dict()), 201


@api_bp.route("/chats/<int:chat_id>", methods=["GET"])
def get_chat(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    out = chat.to_dict()
    out["messages"] = [m.to_dict() for m in chat.messages]
    return jsonify(out)


@api_bp.route("/chats/<int:chat_id>", methods=["PATCH"])
def update_chat(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    data = request.get_json() or {}
    if "context_ids" in data:
        chat.context_ids = data["context_ids"] if isinstance(data["context_ids"], list) else []
    if "title" in data:
        title = (data.get("title") or "").strip()
        chat.title = title[:80] if title else chat.title
    if "web_search_mode" in data or "web_search_enabled" in data:
        web_search_mode, err = _mode_from_payload(
            data,
            default_mode=resolve_chat_web_search_mode(chat),
        )
        if err:
            return jsonify({"error": err}), 400
        chat.web_search_mode = web_search_mode
        chat.web_search_enabled = is_web_search_enabled(web_search_mode)
    db.session.commit()
    return jsonify(chat.to_dict())


@api_bp.route("/chats/<int:chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    chat = Chat.query.get_or_404(chat_id)
    db.session.delete(chat)
    db.session.commit()
    return "", 204


def _stream_content_chunked(content, chunk_size=50):
    """Yield SSE chunk events for content in small pieces so the UI can display progressively."""
    import time
    if not content:
        return
    for i in range(0, len(content), chunk_size):
        yield f"data: {json.dumps({'t': 'chunk', 'c': content[i:i + chunk_size]})}\n\n"
        # Small delay to simulate natural streaming (prevents all chunks arriving at once)
        time.sleep(0.01)


def _stream_provider_chunks(provider_gen, chunk_size=50):
    """Stream from provider generator, re-chunking into smaller pieces for smoother UI updates.
    Yields (sse_event, chunk_text) tuples so caller can accumulate full_content."""
    buffer = ""
    for chunk in provider_gen:
        buffer += chunk
        # Yield accumulated buffer in small chunks
        while len(buffer) >= chunk_size:
            piece = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            yield (f"data: {json.dumps({'t': 'chunk', 'c': piece})}\n\n", piece)
    # Yield remaining buffer
    if buffer:
        yield (f"data: {json.dumps({'t': 'chunk', 'c': buffer})}\n\n", buffer)


def _title_fallback(first_user_content):
    """When LLM title generation is unavailable, use first ~40 chars of user message."""
    t = (first_user_content or "").strip().replace("\n", " ")[:40]
    return t.strip() or "New chat"


def _generate_title(first_user_content):
    """Generate a short title from first ~100 chars using the chat_namer model from models.yaml."""
    from backend.providers import base as providers_base
    model_id = get_chat_namer_model_id()
    if not model_id:
        return _title_fallback(first_user_content)
    snippet = (first_user_content or "")[:100]
    title_prompt = load_prompt("chat_title")
    if not title_prompt:
        title_prompt = "Generate an extremely short chat title: 2–4 words max, no punctuation. Reply with only the title, nothing else.\n\n{{SNIPPET}}"
    content = title_prompt.replace("{{SNIPPET}}", snippet)
    messages = [{"role": "user", "content": content}]
    try:
        title_parts = list(providers_base.generate(messages, model_id, stream=True))
        title = "".join(title_parts).strip() or _title_fallback(first_user_content)
        return (title[:80] if title else _title_fallback(first_user_content))
    except Exception:
        return _title_fallback(first_user_content)


@api_bp.route("/chats/<int:chat_id>/messages/regenerate", methods=["POST"])
def regenerate_message(chat_id):
    """Stream a new assistant reply for an existing user message. Does not add a new user message."""
    chat = Chat.query.get_or_404(chat_id)
    data = request.get_json() or {}
    user_message_id = data.get("message_id")
    model_id = (data.get("model_id") or "").strip()
    if not user_message_id:
        return jsonify({"error": "message_id is required"}), 400
    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if not get_model_info(model_id):
        return jsonify({"error": "model not available"}), 400
    request_web_search_mode_override, err = _request_mode_override_from_payload(data)
    if err:
        return jsonify({"error": err}), 400
    user_msg = Message.query.filter_by(chat_id=chat_id, id=int(user_message_id)).first_or_404()
    if user_msg.role != "user":
        return jsonify({"error": "message_id must be a user message"}), 400
    content = user_msg.content
    cmd_name, cmd_body = get_command_body_if_invoked(content)
    if cmd_name is not None and cmd_body is None:
        return jsonify({"error": f"Command /{cmd_name} not found."}), 400
    user_content_for_llm = (
        f"Command instructions:\n{cmd_body}\n\nUser message: {content.split(None, 1)[1] if content.split() else content}"
        if cmd_body
        else content
    )
    mem_enabled = get_ai_memory_enabled()
    fallback_memories = (
        [m.content for m in Memory.query.order_by(Memory.created_at.desc()).limit(10).all()]
        if mem_enabled
        else []
    )
    # Resolve active rules for this request (original user content, plus any rules referenced in the command body).
    commands_used = [cmd_name] if cmd_name else []
    rules_for_request = resolve_active_rules(content, commands_used)
    cmds_regen = load_commands()
    cmd_regen = cmds_regen.get(cmd_name) if cmd_name else None
    effective_context_ids = list(chat.context_ids) if chat.context_ids else []
    if cmd_regen and getattr(cmd_regen, "context_ids", None):
        for cid in cmd_regen.context_ids:
            if cid not in effective_context_ids:
                effective_context_ids.append(cid)
    system = build_system_message(
        effective_context_ids,
        rag_query=content,
        fallback_memories=fallback_memories,
        rules_for_request=rules_for_request,
        memory_enabled=mem_enabled,
    )
    messages_for_llm = []
    if system:
        messages_for_llm.append({"role": "system", "content": system})
    for m in chat.messages:
        if m.role not in ("user", "assistant"):
            continue
        if m.id == user_msg.id:
            content = message_to_llm_content(user_msg)
            if cmd_body:
                if isinstance(content, list):
                    content = [{"type": "text", "text": user_content_for_llm}] + content[1:]
                else:
                    content = user_content_for_llm
            messages_for_llm.append({"role": "user", "content": content})
            break
        messages_for_llm.append({"role": m.role, "content": message_to_llm_content(m) if m.role == "user" else m.content})

    chat_web_search_mode = resolve_chat_web_search_mode(chat)
    base_request_web_search_mode = (
        resolve_command_web_search_mode(cmd_regen, chat_web_search_mode)
        if cmd_regen
        else chat_web_search_mode
    )
    resolved_request_web_search_mode = (
        request_web_search_mode_override
        if request_web_search_mode_override is not None
        else base_request_web_search_mode
    )

    def stream():
        from backend.providers import base as providers_base
        from backend.services.command_evaluator import evaluate_command_response, execute_task_stream
        meta = None
        try:
            yield f"data: {json.dumps({'t': 'started'})}\n\n"
            use_evaluation_regen = (
                cmd_regen is not None
                and getattr(cmd_regen, "task", None)
                and getattr(cmd_regen, "success_criteria", None)
            )
            user_instructions = content.split(None, 1)[1] if cmd_name and content.split() else (content or "")
            messages_before_user = [{"role": m["role"], "content": m["content"]} for m in messages_for_llm[:-1]]

            if use_evaluation_regen:
                yield f"data: {json.dumps({'t': 'executing', 'msg': 'Completing task...'})}\n\n"
                full_content = ""
                previous_feedback = None
                web_search_meta = None
                command_web_search_mode = resolved_request_web_search_mode
                for attempt in range(1, 4):
                    buffer = []
                    for item in execute_task_stream(
                        cmd_regen,
                        user_instructions,
                        system,
                        messages_before_user,
                        model_id,
                        previous_feedback=previous_feedback,
                        web_search_mode=command_web_search_mode,
                    ):
                        if isinstance(item, tuple):
                            if item[0] == "status":
                                yield f"data: {json.dumps({'t': 'executing', 'msg': item[1]})}\n\n"
                            elif item[0] == "meta":
                                web_search_meta = item[1]
                            elif item[0] == "chunk":
                                buffer.append(item[1])
                        else:
                            buffer.append(item)
                    attempt_content = "".join(buffer)
                    yield f"data: {json.dumps({'t': 'evaluating', 'attempt': attempt})}\n\n"
                    passed = False
                    feedback = ""
                    for eval_attempt in range(1, 4):
                        try:
                            passed, feedback = evaluate_command_response(
                                cmd_regen.task,
                                cmd_regen.success_criteria,
                                cmd_regen.guidelines or "",
                                user_instructions,
                                attempt_content,
                                model_id,
                                timeout=60,
                            )
                            break
                        except (TimeoutError, Exception):
                            if eval_attempt >= 3:
                                passed = False
                                feedback = "Evaluation failed after multiple attempts"
                                break
                    if passed:
                        for chunk in buffer:
                            yield f"data: {json.dumps({'t': 'chunk', 'c': chunk})}\n\n"
                        full_content = attempt_content
                        yield f"data: {json.dumps({'t': 'passed', 'attempt': attempt})}\n\n"
                        break
                    elif attempt < 3:
                        previous_feedback = feedback
                        yield f"data: {json.dumps({'t': 'retrying', 'attempt': attempt + 1})}\n\n"
                    else:
                        for chunk in buffer:
                            yield f"data: {json.dumps({'t': 'chunk', 'c': chunk})}\n\n"
                        full_content = attempt_content
                if web_search_meta is not None:
                    meta = {"web_search": web_search_meta}
            else:
                web_search_mode = resolved_request_web_search_mode
                if is_web_search_enabled(web_search_mode):
                    try:
                        full_content, web_search_meta = None, []
                        for event in providers_base.generate_with_web_search(
                            messages_for_llm,
                            model_id,
                            web_search_mode=web_search_mode,
                        ):
                            if event[0] == "status":
                                yield f"data: {json.dumps({'t': 'executing', 'msg': event[1]})}\n\n"
                            elif event[0] == "result":
                                full_content, web_search_meta = event[1]
                                break
                        if full_content is None:
                            full_content = ""
                    except Exception as e:
                        yield f"data: {json.dumps({'t': 'error', 'error': str(e)})}\n\n"
                        return
                    if full_content:
                        for sse in _stream_content_chunked(full_content):
                            yield sse
                    meta = {"web_search": web_search_meta} if web_search_meta else None
                else:
                    print("\n" + "=" * 60 + " LLM PROMPT (regenerate) " + "=" * 60)
                    for msg in messages_for_llm:
                        role = msg.get("role", "")
                        c = msg.get("content")
                        content_preview = (c[:2000] if isinstance(c, str) else "[multimodal]")
                        if isinstance(c, str) and len(c) > 2000:
                            content_preview += "\n... [truncated]"
                        print(f"\n--- {role.upper()} ---\n{content_preview}")
                    print("=" * 60 + "\n")
                    buffer = []
                    for sse, chunk_text in _stream_provider_chunks(providers_base.generate(messages_for_llm, model_id, stream=True)):
                        buffer.append(chunk_text)
                        yield sse
                    full_content = "".join(buffer)
            assistant_msg = Message(chat_id=chat_id, role="assistant", content=full_content, meta=meta)
            db.session.add(assistant_msg)
            db.session.commit()
            if get_ai_memory_enabled():
                from backend.services.memory_store import extract_and_store

                threading.Thread(
                    target=extract_and_store,
                    args=(content, full_content, current_app._get_current_object()),
                    kwargs={"context_ids": chat.context_ids or []},
                    daemon=True,
                ).start()
            yield f"data: {json.dumps({'t': 'done', 'id': assistant_msg.id, 'title': chat.title})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'t': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_bp.route("/chats/<int:chat_id>/messages/<int:message_id>", methods=["PATCH"])
def patch_message(chat_id, message_id):
    """Update a message's content and delete all messages after it in the chat."""
    chat = Chat.query.get_or_404(chat_id)
    msg = Message.query.filter_by(chat_id=chat_id, id=message_id).first_or_404()
    data = request.get_json() or {}
    content = (data.get("content") or "").strip()
    if content != msg.content:
        msg.content = content
    # Delete all messages after this one (by id order)
    Message.query.filter(Message.chat_id == chat_id, Message.id > message_id).delete(synchronize_session=False)
    db.session.commit()
    db.session.refresh(chat)
    out = [m.to_dict() for m in chat.messages]
    return jsonify(out)


def _parse_attachments_from_request(data):
    """Parse attachments from JSON body. Returns (list of (bytes, filename, content_type), error_response_or_None)."""
    raw = data.get("attachments")
    if not raw or not isinstance(raw, list):
        return [], None
    if len(raw) > config.MAX_ATTACHMENTS_PER_MESSAGE:
        return None, (jsonify({"error": f"Too many attachments (max {config.MAX_ATTACHMENTS_PER_MESSAGE})"}), 400)
    files = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, (jsonify({"error": "Each attachment must be { filename, content_type, data (base64) }"}), 400)
        filename = (item.get("filename") or f"file_{i}").strip() or f"file_{i}"
        content_type = (item.get("content_type") or item.get("contentType") or "").strip() or None
        b64 = item.get("data")
        if not b64:
            return None, (jsonify({"error": f"Attachment {filename}: missing 'data' (base64)"}), 400)
        try:
            raw_bytes = base64.standard_b64decode(b64)
        except Exception:
            return None, (jsonify({"error": f"Attachment {filename}: invalid base64"}), 400)
        if len(raw_bytes) > config.MAX_ATTACHMENT_SIZE_BYTES:
            return None, (jsonify({"error": f"File too large: {filename} (max {config.MAX_ATTACHMENT_SIZE_BYTES} bytes)"}), 400)
        files.append((raw_bytes, filename, content_type))
    return files, None


@api_bp.route("/chats/<int:chat_id>/messages", methods=["POST"])
def add_message(chat_id):
    """Persist user message, stream assistant reply as SSE, persist assistant. Validate /command. Accept optional attachments (JSON)."""
    chat = Chat.query.get_or_404(chat_id)
    data = request.get_json() or {}
    content = (data.get("content") or "").strip()
    model_id = (data.get("model_id") or "").strip()
    if not content:
        return jsonify({"error": "content is required"}), 400

    # Command validation: if /name invoked, command must exist
    cmd_name, cmd_body = get_command_body_if_invoked(content)
    if cmd_name is not None and cmd_body is None:
        return jsonify({"error": f"Command /{cmd_name} not found. Please retry with a valid command or without a command."}), 400

    if not model_id:
        return jsonify({"error": "model_id is required"}), 400
    if not get_model_info(model_id):
        return jsonify({"error": "model not available"}), 400
    request_web_search_mode_override, err = _request_mode_override_from_payload(data)
    if err:
        return jsonify({"error": err}), 400

    # Parse and extract attachments (if any)
    attachments_for_db = None
    content_parts_for_llm = None
    files, err = _parse_attachments_from_request(data)
    if err:
        return err
    if files:
        try:
            attachments_for_db, content_parts_for_llm = extract_attachments(files)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # Persist user message (with or without attachments)
    user_msg = Message(
        chat_id=chat_id,
        role="user",
        content=content,
        attachments=attachments_for_db if attachments_for_db else None,
    )
    db.session.add(user_msg)
    db.session.commit()

    # Build user content for LLM: if command, prepend command body
    if cmd_body:
        user_content_for_llm = f"Command instructions:\n{cmd_body}\n\nUser message: {content.split(None, 1)[1] if content.split() else content}"
    else:
        user_content_for_llm = content

    # Current user turn: string or multimodal parts (user text first, then attachment parts)
    if content_parts_for_llm:
        current_user_content = [{"type": "text", "text": user_content_for_llm}] + content_parts_for_llm
    else:
        current_user_content = user_content_for_llm

    mem_enabled = get_ai_memory_enabled()
    fallback_memories = (
        [m.content for m in Memory.query.order_by(Memory.created_at.desc()).limit(10).all()]
        if mem_enabled
        else []
    )
    # Resolve rules based on original user content and any command body.
    commands_used = [cmd_name] if cmd_name else []
    rules_for_request = resolve_active_rules(content, commands_used)
    # When a command is used, merge chat context_ids with command's context_ids (command contexts auto-included).
    cmds = load_commands()
    cmd = cmds.get(cmd_name) if cmd_name else None
    effective_context_ids = list(chat.context_ids) if chat.context_ids else []
    if cmd and getattr(cmd, "context_ids", None):
        for cid in cmd.context_ids:
            if cid not in effective_context_ids:
                effective_context_ids.append(cid)
    system = build_system_message(
        effective_context_ids,
        rag_query=content,
        fallback_memories=fallback_memories,
        rules_for_request=rules_for_request,
        memory_enabled=mem_enabled,
    )
    messages_for_llm = []
    if system:
        messages_for_llm.append({"role": "system", "content": system})
    for m in chat.messages:
        if m.id == user_msg.id:
            continue
        if m.role in ("user", "assistant"):
            messages_for_llm.append({"role": m.role, "content": message_to_llm_content(m) if m.role == "user" else m.content})
    messages_for_llm.append({"role": "user", "content": current_user_content})

    cmds = load_commands()
    cmd = cmds.get(cmd_name) if cmd_name else None
    use_evaluation = (
        cmd is not None
        and getattr(cmd, "task", None)
        and getattr(cmd, "success_criteria", None)
    )
    user_instructions = content.split(None, 1)[1] if cmd_name and content.split() else (content or "")
    messages_before_user = [{"role": m["role"], "content": m["content"]} for m in messages_for_llm[:-1]]
    chat_web_search_mode = resolve_chat_web_search_mode(chat)
    base_request_web_search_mode = (
        resolve_command_web_search_mode(cmd, chat_web_search_mode)
        if cmd
        else chat_web_search_mode
    )
    resolved_request_web_search_mode = (
        request_web_search_mode_override
        if request_web_search_mode_override is not None
        else base_request_web_search_mode
    )

    def stream():
        from backend.providers import base as providers_base
        from backend.services.command_evaluator import evaluate_command_response, execute_task_stream

        try:
            meta = None
            yield f"data: {json.dumps({'t': 'started'})}\n\n"

            if use_evaluation:
                yield f"data: {json.dumps({'t': 'executing', 'msg': 'Completing task...'})}\n\n"
                full_content = ""
                previous_feedback = None
                web_search_meta = None
                command_web_search_mode = resolved_request_web_search_mode
                for attempt in range(1, 4):
                    # Collect chunks without yielding them yet - wait for evaluation
                    buffer = []
                    for item in execute_task_stream(
                        cmd,
                        user_instructions,
                        system,
                        messages_before_user,
                        model_id,
                        previous_feedback=previous_feedback,
                        web_search_mode=command_web_search_mode,
                    ):
                        if isinstance(item, tuple):
                            if item[0] == "status":
                                yield f"data: {json.dumps({'t': 'executing', 'msg': item[1]})}\n\n"
                            elif item[0] == "meta":
                                web_search_meta = item[1]
                            elif item[0] == "chunk":
                                buffer.append(item[1])
                        else:
                            buffer.append(item)
                    attempt_content = "".join(buffer)

                    yield f"data: {json.dumps({'t': 'evaluating', 'attempt': attempt})}\n\n"
                    passed = False
                    feedback = ""
                    for eval_attempt in range(1, 4):
                        try:
                            passed, feedback = evaluate_command_response(
                                cmd.task,
                                cmd.success_criteria,
                                cmd.guidelines or "",
                                user_instructions,
                                attempt_content,
                                model_id,
                                timeout=60,
                            )
                            break
                        except (TimeoutError, Exception):
                            if eval_attempt >= 3:
                                passed = False
                                feedback = "Evaluation failed after multiple attempts"
                                break

                    if passed:
                        # Success: yield all chunks now, then break
                        for chunk in buffer:
                            yield f"data: {json.dumps({'t': 'chunk', 'c': chunk})}\n\n"
                        full_content = attempt_content
                        yield f"data: {json.dumps({'t': 'passed', 'attempt': attempt})}\n\n"
                        break
                    elif attempt < 3:
                        # Failed but more attempts left: don't show this attempt, retry
                        previous_feedback = feedback
                        yield f"data: {json.dumps({'t': 'retrying', 'attempt': attempt + 1})}\n\n"
                    else:
                        # Final attempt failed: show it anyway
                        for chunk in buffer:
                            yield f"data: {json.dumps({'t': 'chunk', 'c': chunk})}\n\n"
                        full_content = attempt_content
                if web_search_meta is not None:
                    meta = {"web_search": web_search_meta}
            else:
                web_search_mode = resolved_request_web_search_mode
                if is_web_search_enabled(web_search_mode):
                    try:
                        full_content, web_search_meta = None, []
                        for event in providers_base.generate_with_web_search(
                            messages_for_llm,
                            model_id,
                            web_search_mode=web_search_mode,
                        ):
                            if event[0] == "status":
                                yield f"data: {json.dumps({'t': 'executing', 'msg': event[1]})}\n\n"
                            elif event[0] == "result":
                                full_content, web_search_meta = event[1]
                                break
                        if full_content is None:
                            full_content = ""
                    except Exception as e:
                        yield f"data: {json.dumps({'t': 'error', 'error': str(e)})}\n\n"
                        return
                    if full_content:
                        for sse in _stream_content_chunked(full_content):
                            yield sse
                    meta = {"web_search": web_search_meta} if web_search_meta else None
                else:
                    print("\n" + "=" * 60 + " LLM PROMPT " + "=" * 60)
                    for msg in messages_for_llm:
                        role = msg.get("role", "")
                        c = msg.get("content")
                        content_preview = (c[:2000] if isinstance(c, str) else "[multimodal]")
                        if isinstance(c, str) and len(c) > 2000:
                            content_preview += "\n... [truncated]"
                        print(f"\n--- {role.upper()} ---\n{content_preview}")
                    print("=" * 60 + "\n")
                    buffer = []
                    for sse, chunk_text in _stream_provider_chunks(providers_base.generate(messages_for_llm, model_id, stream=True)):
                        buffer.append(chunk_text)
                        yield sse
                    full_content = "".join(buffer)

            assistant_msg = Message(chat_id=chat_id, role="assistant", content=full_content, meta=meta)
            db.session.add(assistant_msg)
            db.session.flush()
            msg_count = Message.query.filter_by(chat_id=chat_id).count()
            is_first_reply = msg_count == 2
            needs_title = (not chat.title or (chat.title or "").strip() == "New chat")
            if is_first_reply or needs_title:
                new_title = _generate_title(content)
                if new_title and new_title.strip() and (new_title.strip() != "New chat"):
                    title_to_send = new_title.strip()[:80]
                else:
                    title_to_send = _title_fallback(content)
                Chat.query.filter_by(id=chat_id).update(
                    {"title": title_to_send, "updated_at": datetime.utcnow()},
                    synchronize_session=False,
                )
            else:
                title_to_send = chat.title
            db.session.commit()
            if get_ai_memory_enabled():
                from backend.services.memory_store import extract_and_store

                threading.Thread(
                    target=extract_and_store,
                    args=(content, full_content, current_app._get_current_object()),
                    kwargs={"context_ids": chat.context_ids or []},
                    daemon=True,
                ).start()
            yield f"data: {json.dumps({'t': 'done', 'id': assistant_msg.id, 'title': title_to_send})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'t': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
