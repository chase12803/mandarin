"""Build system message from base prompt, rules + human context files. Skip missing."""
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from zoneinfo import ZoneInfo

import yaml

import config
from backend.services.prompt_loader import load_prompt
from backend.services.web_search_mode import (
    WEB_SEARCH_MODE_OFF,
    mode_from_legacy_enabled,
    parse_web_search_mode,
)


def _read_base_system_prompt():
    """Read base system prompt from prompts/system.md; substitute {{DATE}}, {{DAY}}, {{TIME}}, {{USER_NAME}}."""
    text = load_prompt("system")
    if not text:
        text = "# Role\n\nYou are a personal assistant. Use the Rules, Context, and Relevant memory sections below when provided."
    now = datetime.now()
    today = now.strftime("%A, %B %d, %Y")
    day_of_week = now.strftime("%A")
    now_est = now.astimezone(ZoneInfo("America/New_York"))
    time_of_day = now_est.strftime("%I:%M %p %Z")  # EST or EDT
    if time_of_day[0] == "0":
        time_of_day = time_of_day[1:]  # "2:30 PM EST" not "02:30 PM EST"
    user_name = config.USER_NAME if config.USER_NAME else "the user"
    text = text.replace("{{DATE}}", today).replace("{{DAY}}", day_of_week).replace("{{TIME}}", time_of_day).replace("{{USER_NAME}}", user_name)
    return text


class Rule:
    def __init__(self, id_: str, name: str, always_on: bool, tags: Optional[List[str]], body: str) -> None:
        self.id = id_
        self.name = name
        self.always_on = always_on  # True = apply to every request
        self.tags = tags or []
        self.body = body


class Command:
    def __init__(
        self,
        id_: str,
        name: str,
        description: str,
        tags: Optional[List[str]],
        body: str,
        task: Optional[str] = None,
        success_criteria: Optional[str] = None,
        guidelines: Optional[str] = None,
        context_ids: Optional[List[str]] = None,
        web_search_mode: str = WEB_SEARCH_MODE_OFF,
        web_search_mode_explicit: bool = False,
        web_search_enabled: bool = False,
    ) -> None:
        self.id = id_
        self.name = name
        self.description = description
        self.tags = tags or []
        self.body = body
        self.task = task
        self.success_criteria = success_criteria
        self.guidelines = guidelines
        self.context_ids = context_ids or []
        self.web_search_mode = web_search_mode
        self.web_search_mode_explicit = web_search_mode_explicit
        self.web_search_enabled = web_search_enabled


def _parse_command_sections(body: str) -> Dict[str, str]:
    """Parse markdown body into Task, Success Criteria, Guidelines sections.
    Returns dict with keys: task, success_criteria, guidelines.
    Falls back to treating entire body as task if sections not found.
    """
    out: Dict[str, str] = {"task": "", "success_criteria": "", "guidelines": ""}
    if not (body or "").strip():
        return out
    text = body.strip()
    # Match ## Task, ## Success Criteria, ## Guidelines (case-insensitive for header)
    section_pattern = re.compile(
        r"^##\s+(Task|Success\s+Criteria|Guidelines)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(section_pattern.finditer(text))
    if not matches:
        out["task"] = text
        return out
    for i, m in enumerate(matches):
        section_name = m.group(1).lower().replace(" ", "_")
        if "criteria" in section_name:
            section_key = "success_criteria"
        elif "task" in section_name:
            section_key = "task"
        elif "guidelines" in section_name:
            section_key = "guidelines"
        else:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        out[section_key] = content
    return out


_RULES_CACHE: Dict[str, Rule] = {}
_COMMANDS_CACHE: Dict[str, Command] = {}
_RULES_MTIME: float = 0.0
_COMMANDS_MTIME: float = 0.0


def _split_frontmatter(text: str) -> Tuple[Optional[dict], str]:
    """Split optional YAML frontmatter from body. Returns (meta_dict_or_None, body)."""
    if not text.startswith("---"):
        return None, text
    # Expect the common pattern:
    # ---\n
    # yaml...
    # ---\n
    # body...
    try:
        _, fm_and_rest = text.split("---", 1)
    except ValueError:
        return None, text
    fm_and_rest = fm_and_rest.lstrip("\n")
    if "\n---" not in fm_and_rest:
        return None, text
    header, _, body = fm_and_rest.partition("\n---")
    try:
        meta = yaml.safe_load(header) or {}
    except Exception as e:
        print(f"Failed to parse rule/command frontmatter: {e}")
        return None, text
    return meta, body.lstrip().lstrip("\n")


def _collect_dir_mtime(dir_path: Path) -> float:
    latest = 0.0
    if not dir_path.exists():
        return latest
    for p in dir_path.glob("*.md"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime > latest:
            latest = mtime
    return latest


def _demote_markdown_headings(text: str, levels: int = 1) -> str:
    """Demote all markdown headings by N levels (# -> ##, ## -> ###, etc.).

    This keeps rule/context headings visually nested under the main sections
    in the system prompt.
    """
    if not text or levels <= 0:
        return text

    def _repl(match: re.Match) -> str:
        hashes = match.group(1)
        rest = match.group(2)
        new_level = min(len(hashes) + levels, 6)
        return "#" * new_level + " " + rest

    return re.sub(r"^(#{1,6})\s+(.*)$", _repl, text, flags=re.MULTILINE)


def load_rules() -> Dict[str, Rule]:
    """Load rules from DATA_DIR/rules with simple mtime-based caching."""
    global _RULES_CACHE, _RULES_MTIME
    mtime = _collect_dir_mtime(config.RULES_DIR)
    if mtime <= _RULES_MTIME and _RULES_CACHE:
        return _RULES_CACHE
    rules: Dict[str, Rule] = {}
    if config.RULES_DIR.exists():
        for path in config.RULES_DIR.glob("*.md"):
            try:
                raw = path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Failed to read rule file {path}: {e}")
                continue
            meta, body = _split_frontmatter(raw)
            stem = path.stem
            rid = stem
            name = stem
            always_on = False
            tags: List[str] = []
            if meta:
                rid = str(meta.get("id") or rid)
                name = str(meta.get("name") or name)
                # Support both new always_on and legacy scope/enabled keys.
                if "always_on" in meta:
                    always_on = bool(meta.get("always_on"))
                else:
                    scope = str(meta.get("scope") or "optional")
                    enabled = bool(meta.get("enabled", True))
                    always_on = enabled and (scope == "global")
                mtags = meta.get("tags") or []
                if isinstance(mtags, list):
                    tags = [str(t) for t in mtags]
            if not re.match(r"^[a-zA-Z0-9_-]+$", rid):
                print(f"Skipping rule with invalid id {rid!r} in {path}")
                continue
            if rid in rules:
                print(f"Duplicate rule id {rid!r} in {path}; skipping")
                continue
            rules[rid] = Rule(rid, name, always_on, tags, body.strip())
    _RULES_CACHE = rules
    _RULES_MTIME = mtime
    return rules


def load_commands() -> Dict[str, Command]:
    """Load commands from DATA_DIR/commands with simple mtime-based caching."""
    global _COMMANDS_CACHE, _COMMANDS_MTIME
    mtime = _collect_dir_mtime(config.COMMANDS_DIR)
    if mtime <= _COMMANDS_MTIME and _COMMANDS_CACHE:
        return _COMMANDS_CACHE
    cmds: Dict[str, Command] = {}
    if config.COMMANDS_DIR.exists():
        for path in config.COMMANDS_DIR.glob("*.md"):
            try:
                raw = path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Failed to read command file {path}: {e}")
                continue
            meta, body = _split_frontmatter(raw)
            stem = path.stem
            cid = stem
            name = stem
            description = ""
            tags: List[str] = []
            context_ids: List[str] = []
            web_search_mode = WEB_SEARCH_MODE_OFF
            web_search_mode_explicit = False
            web_search_enabled = False
            if meta:
                cid = str(meta.get("id") or cid)
                name = str(meta.get("name") or name)
                description = str(meta.get("description") or description)
                mtags = meta.get("tags") or []
                if isinstance(mtags, list):
                    tags = [str(t) for t in mtags]
                ctx_ids = meta.get("context_ids") or []
                if isinstance(ctx_ids, list):
                    context_ids = [str(x) for x in ctx_ids if x]
                explicit_mode = parse_web_search_mode(meta.get("web_search_mode"))
                if explicit_mode is not None:
                    web_search_mode = explicit_mode
                    web_search_mode_explicit = True
                    web_search_enabled = explicit_mode != WEB_SEARCH_MODE_OFF
                else:
                    web_search_enabled = bool(meta.get("web_search_enabled", meta.get("web_search", False)))
                    web_search_mode = mode_from_legacy_enabled(web_search_enabled)
                    web_search_mode_explicit = False
            if not re.match(r"^[a-zA-Z0-9_-]+$", cid):
                print(f"Skipping command with invalid id {cid!r} in {path}")
                continue
            if cid in cmds:
                print(f"Duplicate command id {cid!r} in {path}; skipping")
                continue
            body_stripped = body.strip()
            sections = _parse_command_sections(body_stripped)
            cmds[cid] = Command(
                cid,
                name,
                description,
                tags,
                body_stripped,
                task=sections.get("task") or None,
                success_criteria=sections.get("success_criteria") or None,
                guidelines=sections.get("guidelines") or None,
                context_ids=context_ids,
                web_search_mode=web_search_mode,
                web_search_mode_explicit=web_search_mode_explicit,
                web_search_enabled=web_search_enabled,
            )
    _COMMANDS_CACHE = cmds
    _COMMANDS_MTIME = mtime
    return cmds


def _context_name_from_first_line(text):
    """Expect first line like # Context Name; return the name."""
    first = (text.split("\n")[0] or "").strip()
    if first.startswith("#"):
        return first.lstrip("#").strip()
    return first or "Untitled"


def _read_context(context_id):
    """Read context file by id (filename). Return (name, content). Skip if missing."""
    path = config.CONTEXTS_DIR / f"{context_id}.md"
    if not path.exists():
        path = config.CONTEXTS_DIR / context_id
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    name = _context_name_from_first_line(text)
    return name, text


def list_contexts():
    """Return list of { id, name } for all .md files in contexts dir. Id = filename without .md."""
    out = []
    for path in config.CONTEXTS_DIR.glob("*.md"):
        id_ = path.stem
        text = path.read_text(encoding="utf-8")
        name = _context_name_from_first_line(text)
        out.append({"id": id_, "name": name})
    return out


def _expand_rag_query_for_retrieval(user_message):
    """For short first-person questions (e.g. 'How tall am I?'), append topic hints so RAG is more likely to retrieve relevant memories (height, weight, etc.)."""
    if not user_message or len(user_message) > 100:
        return user_message
    lower = user_message.lower()
    first_person = any(phrase in lower for phrase in (" i ", " my ", " me ", "am i", "do i", "what's my", "what is my"))
    if not first_person:
        return user_message
    return (user_message + " height weight physical attributes user facts").strip()


def _build_rules_section(active_rules: Optional[List[Rule]]) -> str:
    """Render a Rules section from a list of Rule objects, with legacy rules.md fallback."""
    lines: List[str] = []
    if active_rules:
        lines.append("## Rules")
        for rule in active_rules:
            # Separate rules clearly; use headings for readability.
            lines.append(f"### {rule.name}")
            if rule.body:
                # Demote headings inside rule bodies so they nest under the rule name.
                lines.append(_demote_markdown_headings(rule.body))
    # Legacy fallback: if no rules dir content but rules.md exists, append it.
    if (not active_rules or len(active_rules) == 0) and config.RULES_PATH.exists():
        legacy = config.RULES_PATH.read_text(encoding="utf-8").strip()
        if legacy:
            if not lines:
                lines.append("## Rules")
            lines.append(legacy)
    return "\n\n".join(lines).strip()


def build_system_message(
    context_ids=None,
    rag_query=None,
    fallback_memories=None,
    rules_for_request: Optional[List[Rule]] = None,
    memory_enabled: bool = True,
):
    """context_ids: list of context file ids.
    rag_query: optional user message to retrieve relevant memory.
    fallback_memories: optional list of memory content strings; used when RAG returns no hits or fails.
    rules_for_request: optional list of Rule objects to include in ## Rules.
    memory_enabled: when False, no ## Relevant memory section (RAG and fallback skipped)."""
    parts: List[str] = []
    base = _read_base_system_prompt()
    if base:
        parts.append(base)
    rules_section = _build_rules_section(rules_for_request or [])
    if rules_section:
        parts.append(rules_section)
    for cid in (context_ids or []):
        parsed = _read_context(cid)
        if parsed:
            name, content = parsed
            # Avoid repeating the context title: if the first line is a markdown
            # heading, strip it and only show the body under "## Context: name".
            lines = (content or "").splitlines()
            if lines and (lines[0].lstrip().startswith("#")):
                body = "\n".join(lines[1:]).lstrip("\n")
            else:
                body = content
            # Demote headings inside context bodies so they nest cleanly.
            body = _demote_markdown_headings(body)
            parts.append(f"## Context: {name}\n{body}")
    if memory_enabled and rag_query:
        memory_text = None
        # When there are few memories, use all of them from DB so retrieval is reliable.
        if fallback_memories and len(fallback_memories) <= 5:
            memory_text = "\n".join(str(c) for c in fallback_memories)
        else:
            try:
                from backend.services.rag import query as rag_query_fn
                expanded_query = _expand_rag_query_for_retrieval(rag_query)
                hits = rag_query_fn(expanded_query, top_k=5)
                if hits:
                    memory_text = "\n".join(h[1] for h in hits)
            except Exception as e:
                import traceback
                print(f"RAG query failed (using fallback if any): {e}")
                traceback.print_exc()
            if memory_text is None and fallback_memories:
                memory_text = "\n".join(str(c) for c in fallback_memories)
        if memory_text:
            parts.append("## Relevant memory\n" + memory_text)
    return "\n\n".join(parts) if parts else ""


def resolve_command(content):
    """If content starts with /name (alphabetic name), return (command_body, rest) or (None, content).
    Command name is alphabetic only. Returns (None, content) if no command or invalid."""
    if not content.strip().startswith("/"):
        return None, content
    match = re.match(r"^/([a-zA-Z]+)\s*(.*)$", content, re.DOTALL)
    if not match:
        return None, content
    name, rest = match.group(1), match.group(2).strip()
    path = config.COMMANDS_DIR / f"{name}.md"
    if not path.exists():
        return None, content  # caller will check and 400 if they want to require command
    body = path.read_text(encoding="utf-8")
    return body, rest


def get_command_body_if_invoked(content):
    """If content starts with /name, return (name, body) or (None, None). If command file missing, return (name, None)."""
    if not content.strip().startswith("/"):
        return None, None
    match = re.match(r"^/([a-zA-Z0-9_-]+)\s*(.*)$", content, re.DOTALL)
    if not match:
        return None, None
    name = match.group(1)
    path = config.COMMANDS_DIR / f"{name}.md"
    if not path.exists():
        return name, None
    return name, path.read_text(encoding="utf-8")


def _extract_rule_ids_from_text(text: str) -> Set[str]:
    """Find @rule-id occurrences; rule-id must be [a-zA-Z0-9_-]+."""
    if not text:
        return set()
    return set(m.group(1) for m in re.finditer(r"@([a-zA-Z0-9_-]+)", text))


def resolve_active_rules(user_content: str, commands_used: Optional[List[str]] = None) -> List[Rule]:
    """Determine which rules should apply for a single request.

    user_content: original user message (with any @rule-id mentions).
    commands_used: list of command IDs invoked for this request.
    """
    commands_used = commands_used or []
    rules = load_rules()
    cmds = load_commands()

    # Base: all always_on rules.
    active_ids: Set[str] = {r.id for r in rules.values() if r.always_on}

    # Direct from user @mentions.
    active_ids.update(_extract_rule_ids_from_text(user_content))

    # Direct from commands (their bodies may @mention rules).
    for cid in commands_used:
        cmd = cmds.get(cid)
        if not cmd:
            continue
        active_ids.update(_extract_rule_ids_from_text(cmd.body))

    if not active_ids:
        return [r for r in rules.values() if r.always_on]

    # Build dependency graph: rule A -> rule B when A's body @B.
    deps: Dict[str, Set[str]] = {}
    for r in rules.values():
        referenced = _extract_rule_ids_from_text(r.body)
        deps[r.id] = referenced

    resolved_ids: Set[str] = set()

    def _visit(rid: str, visiting: Set[str]) -> None:
        if rid in resolved_ids or rid in visiting:
            return
        visiting.add(rid)
        for dep_id in deps.get(rid, set()):
            if dep_id in rules:
                _visit(dep_id, visiting)
        visiting.remove(rid)
        if rid in rules:
            resolved_ids.add(rid)

    for rid in list(active_ids):
        _visit(rid, set())

    # Stable sort: always_on rules first, then by name.
    final_rules = [rules[rid] for rid in resolved_ids]
    final_rules.sort(key=lambda r: (0 if r.always_on else 1, r.name.lower()))
    return final_rules
