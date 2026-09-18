"""SQLite-backed persistent memory with FTS5 search and vector embeddings.

Reveries-style design:
- CRUD operations on structured memories
- Full-text search via FTS5
- Semantic search via cosine similarity on embeddings
- ``superseded_by`` semantics for memory lifecycle management
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from agent_framework.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Memory:
    """A single memory record.

    Attributes:
        id: Unique identifier.
        scope: Scope label (e.g., "global", "project", "session").
        kind: Kind of memory (e.g., "fact", "decision", "context").
        title: Short title/label.
        content: Full memory content.
        tags: Space-separated tags for filtering.
        importance: Importance score 0.0–1.0.
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 last-updated timestamp.
        superseded_by: ID of the memory that supersedes this one, or None.
        source: Origin of the memory (e.g., "user", "agent", "import").
        embedding: Optional embedding vector for semantic search.
    """
    id: str
    scope: str = "global"
    kind: str = "fact"
    title: str = ""
    content: str = ""
    tags: str = ""
    importance: float = 0.5
    created_at: str = ""
    updated_at: str = ""
    superseded_by: Optional[str] = None
    source: str = "agent"
    embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT UNIQUE NOT NULL,
    scope           TEXT    NOT NULL DEFAULT 'global',
    kind            TEXT    NOT NULL DEFAULT 'fact',
    title           TEXT    NOT NULL DEFAULT '',
    content         TEXT    NOT NULL DEFAULT '',
    tags            TEXT    NOT NULL DEFAULT '',
    importance      REAL   NOT NULL DEFAULT 0.5,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    superseded_by   TEXT    DEFAULT NULL,
    source          TEXT    NOT NULL DEFAULT 'agent',
    embedding       TEXT    DEFAULT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title, content, tags,
    tokenize='porter'
);

-- Sync FTS5 with memories table via triggers (conditional: only fire when FTS5 columns change)
CREATE TRIGGER mem_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, title, content, tags)
    VALUES (NEW._rowid, NEW.title, NEW.content, NEW.tags);
END;

CREATE TRIGGER mem_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE rowid = OLD._rowid;
END;

CREATE TRIGGER mem_au AFTER UPDATE ON memories
WHEN OLD.title != NEW.title OR OLD.content != NEW.content OR OLD.tags != NEW.tags BEGIN
    DELETE FROM memories_fts WHERE rowid = OLD._rowid;
    INSERT INTO memories_fts(rowid, title, content, tags)
    VALUES (NEW._rowid, NEW.title, NEW.content, NEW.tags);
END;
"""


class MemoryStore:
    """SQLite-backed memory store with FTS5 full-text search.

    Provides CRUD operations, full-text search, semantic search via
    cosine similarity, and memory lifecycle management (supersede).

    The store is thread-safe via a per-instance lock.

    Args:
        db_path: Path to the SQLite database file. Creates if not exists.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # -- connection management --

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _connect(self) -> sqlite3.Connection:
        # Use check_same_thread=False since we manage our own lock.
        # Default isolation_level ('DEFERRED') gives us explicit commits.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        return conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _init_db(self) -> None:
        """Initialize schema on first access."""
        with self._lock:
            _ = self.conn

    # -- row helpers --

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory dataclass."""
        emb_json = row["embedding"]
        embedding = json.loads(emb_json) if emb_json else None
        return Memory(
            id=row["id"],
            scope=row["scope"],
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            tags=row["tags"],
            importance=row["importance"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by=row["superseded_by"],
            source=row["source"],
            embedding=embedding,
        )

    # -- CRUD --

    def save(
        self,
        title: str,
        content: str,
        *,
        memory_id: Optional[str] = None,
        scope: str = "global",
        kind: str = "fact",
        tags: str = "",
        importance: float = 0.5,
        source: str = "agent",
        embedding: Optional[list[float]] = None,
    ) -> str:
        """Create a new memory entry.

        Args:
            title: Short label for the memory.
            content: Full memory content.
            memory_id: Optional explicit ID (auto-generated if None).
            scope: Scope label.
            kind: Kind/category.
            tags: Space-separated tags.
            importance: Importance 0.0–1.0.
            source: Origin label.
            embedding: Optional embedding vector.

        Returns:
            The memory ID.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mid = memory_id or str(uuid.uuid4())
        emb_json = json.dumps(embedding) if embedding else None

        with self._lock:
            self.conn.execute(
                """INSERT INTO memories
                   (id, scope, kind, title, content, tags, importance,
                    created_at, updated_at, source, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mid, scope, kind, title, content, tags, importance,
                 now, now, source, emb_json),
            )
            self.conn.commit()

        logger.debug("Saved memory %s: %s", mid, title[:50])
        return mid

    def get(self, memory_id: str) -> Optional[Memory]:
        """Fetch a single memory by ID.

        Args:
            memory_id: Memory unique identifier.

        Returns:
            Memory object or None if not found.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def update(
        self,
        memory_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[str] = None,
        importance: Optional[float] = None,
        embedding: Optional[list[float]] = None,
    ) -> bool:
        """Update fields of an existing memory.

        Args:
            memory_id: Memory unique identifier.
            title, content, tags, importance, embedding: Fields to update
                (only non-None values are modified).

        Returns:
            True if the memory was found and updated, False otherwise.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        updates: list[str] = []
        params: list[Any] = []

        for fname, val in [
            ("title", title), ("content", content), ("tags", tags),
            ("importance", importance),
        ]:
            if val is not None:
                updates.append(f"{fname} = ?")
                params.append(val)

        emb_json = json.dumps(embedding) if embedding is not None else None
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(emb_json)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(now)
        params.append(memory_id)

        with self._lock:
            cur = self.conn.execute(
                f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            self.conn.commit()
            return cur.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: Memory unique identifier.

        Returns:
            True if found and deleted, False otherwise.
        """
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            self.conn.commit()
            return cur.rowcount > 0

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark an old memory as superseded by a new one.

        Args:
            old_id: ID of the memory being superseded.
            new_id: ID of the replacement memory.

        Returns:
            True if the old memory was found and superseded.
        """
        with self._lock:
            cur = self.conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def list_memories(
        self,
        *,
        scope: Optional[str] = None,
        kind: Optional[str] = None,
        tags: Optional[str] = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[Memory]:
        """List memories with optional filters.

        Args:
            scope: Filter by scope.
            kind: Filter by kind.
            tags: Filter by tag (space-separated, AND logic).
            include_superseded: Include superseded memories.
            limit: Maximum results.

        Returns:
            List of Memory objects.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if scope:
            clauses.append("scope = ?")
            params.append(scope)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if tags:
            for tag in tags.split():
                clauses.append("tags LIKE ?")
                params.append(f"%{tag}%")
        if not include_superseded:
            clauses.append("superseded_by IS NULL")

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(limit)

        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE {where} "
                f"ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()

        return [self._row_to_memory(r) for r in rows]

    # -- search --

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        include_superseded: bool = False,
        embedding: Optional[list[float]] = None,
    ) -> list[Memory]:
        """Search memories using FTS5 full-text matching.

        Combines FTS5 results with cosine similarity on embeddings,
        then reranks by ``relevance × importance × ¬superseded``.

        Args:
            query: Search query string.
            top_k: Maximum results to return.
            include_superseded: Include superseded memories.

        Returns:
            List of Memory objects ranked by relevance score.
        """
        results: dict[str, float] = {}  # id → score

        # 1. FTS5 match — MUST be a standalone query (MATCH requires inline text)
        safe_query = query.replace("'", "''")
        with self._lock:
            fts_rows = self.conn.execute(
                f"SELECT rowid, rank FROM memories_fts "
                f"WHERE memories_fts MATCH '{safe_query}' "
                f"ORDER BY rank LIMIT ?",
                (top_k * 3,),
            ).fetchall()

        # 2. Look up full memory records and apply superseded filter
        if fts_rows:
            rowids = [r["rowid"] for r in fts_rows]
            placeholders = ",".join(["?"] * len(rowids))
            where_base = f"_rowid IN ({placeholders})"
            if not include_superseded:
                where_base += " AND superseded_by IS NULL"

            with self._lock:
                mem_rows = self.conn.execute(
                    f"SELECT * FROM memories WHERE {where_base}",
                    rowids,
                ).fetchall()

            # Build rank lookup from FTS results
            rank_map = {r["rowid"]: r["rank"] for r in fts_rows}

            for row in mem_rows:
                mem_id = row["id"]
                rid = row["_rowid"]
                rank = rank_map.get(rid)
                fts_score = 1.0 / (1.0 + abs(rank)) if rank else 0.1
                importance = row["importance"]
                superseded_penalty = 0.1 if row["superseded_by"] else 1.0
                score = fts_score * importance * superseded_penalty
                results[mem_id] = max(results.get(mem_id, 0), score)

        # 2. Embedding cosine similarity (if query embedding provided)
        if embedding:
            emb_results = self.search_embeddings(
                embedding,
                top_k=top_k * 3,
                include_superseded=include_superseded,
            )
            for m, score in emb_results:
                results[m.id] = max(results.get(m.id, 0), score)

        # Sort by combined score
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

        with self._lock:
            mems: list[Memory] = []
            for mem_id, _score in ranked:
                row = self.conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (mem_id,)
                ).fetchone()
                if row:
                    mems.append(self._row_to_memory(row))

        return mems

    def search_embeddings(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        include_superseded: bool = False,
        max_scan: int = 1000,
    ) -> list[tuple[Memory, float]]:
        """Semantic search using cosine similarity on stored embeddings.

        Args:
            query_embedding: Float vector for the query.
            top_k: Maximum results.
            include_superseded: Include superseded memories.
            max_scan: Maximum rows to scan (avoids full table scans on large stores).

        Returns:
            List of (Memory, score) tuples ranked by cosine similarity.
        """
        query_norm = _vector_norm(query_embedding)
        if query_norm == 0:
            return []

        where = "embedding IS NOT NULL"
        if not include_superseded:
            where += " AND superseded_by IS NULL"

        # P2 fix: LIMIT to max_scan to avoid O(n) full table scans
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM memories WHERE {where} LIMIT ?",
                (max_scan,),
            ).fetchall()

        scored: list[tuple[str, float]] = []
        for row in rows:
            emb_json = row["embedding"]
            if not emb_json:
                continue
            try:
                stored = json.loads(emb_json)
            except (json.JSONDecodeError, TypeError):
                continue

            sim = _cosine_similarity(query_embedding, stored)
            importance = row["importance"]
            superseded_penalty = 0.1 if row["superseded_by"] else 1.0
            score = sim * importance * superseded_penalty
            scored.append((str(row["id"]), score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [sid for sid, _ in scored[:top_k]]

        with self._lock:
            mems: list[tuple[Memory, float]] = []
            for mem_id in top_ids:
                row = self.conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (mem_id,)
                ).fetchone()
                if row:
                    # Retrieve the score for this memory
                    score = next((s for sid, s in scored if sid == mem_id), 0.0)
                    mems.append((self._row_to_memory(row), score))

        return mems

    # -- tool definitions for agent integration --

    @staticmethod
    def _tool_save(store: MemoryStore) -> Callable[..., str]:
        """Executor wrapper for memory_save tool."""
        def _executor(
            title: str, content: str,
            scope: str = "global", kind: str = "fact",
            tags: str = "", importance: float = 0.5,
            source: str = "agent",
        ) -> str:
            return store.save(
                title, content,
                scope=scope, kind=kind, tags=tags,
                importance=importance, source=source,
            )
        return _executor

    @staticmethod
    def _tool_search(store: MemoryStore) -> Callable[..., str]:
        """Executor wrapper for memory_search tool."""
        def _executor(query: str, top_k: int = 10) -> str:
            results = store.search(query, top_k=top_k)
            return json.dumps([
                {"id": m.id, "title": m.title, "content": m.content[:500],
                 "importance": m.importance, "tags": m.tags}
                for m in results
            ])
        return _executor

    @staticmethod
    def _tool_get(store: MemoryStore) -> Callable[..., str]:
        """Executor wrapper for memory_get tool."""
        def _executor(memory_id: str) -> str:
            m = store.get(memory_id)
            if m is None:
                return f"Memory '{memory_id}' not found"
            return json.dumps({
                "id": m.id, "scope": m.scope, "kind": m.kind,
                "title": m.title, "content": m.content,
                "tags": m.tags, "importance": m.importance,
                "source": m.source,
            })
        return _executor

    @staticmethod
    def _tool_update(store: MemoryStore) -> Callable[..., str]:
        """Executor wrapper for memory_update tool."""
        def _executor(
            memory_id: str, title: Optional[str] = None,
            content: Optional[str] = None, tags: Optional[str] = None,
            importance: Optional[float] = None,
        ) -> str:
            ok = store.update(
                memory_id, title=title, content=content,
                tags=tags, importance=importance,
            )
            return "Updated" if ok else f"Memory '{memory_id}' not found"
        return _executor

    @staticmethod
    def _tool_supersede(store: MemoryStore) -> Callable[..., str]:
        """Executor wrapper for memory_supersede tool."""
        def _executor(old_id: str, new_id: str) -> str:
            ok = store.supersede(old_id, new_id)
            if ok:
                return f"Memory '{old_id}' superseded by '{new_id}'"
            return f"Memory '{old_id}' not found"
        return _executor

    @staticmethod
    def _tool_list(store: MemoryStore) -> Callable[..., str]:
        """Executor wrapper for memory_list tool."""
        def _executor(
            scope: Optional[str] = None, kind: Optional[str] = None,
            limit: int = 20,
        ) -> str:
            mems = store.list_memories(scope=scope, kind=kind, limit=limit)
            return json.dumps([
                {"id": m.id, "title": m.title, "scope": m.scope,
                 "kind": m.kind, "tags": m.tags, "updated_at": m.updated_at}
                for m in mems
            ])
        return _executor

    def get_tools(self) -> list[ToolDef]:
        """Build ToolDef objects for all memory tools, wired to this store.

        Returns:
            List of ToolDef instances for memory_save, memory_search,
            memory_get, memory_update, memory_supersede, memory_list.
        """
        return [
            ToolDef(
                name="memory_save",
                description="Save a new memory to the knowledge base.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short title"},
                        "content": {"type": "string", "description": "Full content"},
                        "scope": {"type": "string", "description": "Scope label (global, project, session)", "default": "global"},
                        "kind": {"type": "string", "description": "Kind (fact, decision, context)", "default": "fact"},
                        "tags": {"type": "string", "description": "Space-separated tags", "default": ""},
                        "importance": {"type": "number", "description": "Importance 0-1", "default": 0.5},
                        "source": {"type": "string", "description": "Origin", "default": "agent"},
                    },
                    "required": ["title", "content"],
                },
                executor=self._tool_save(self),
            ),
            ToolDef(
                name="memory_search",
                description="Search memories by full-text query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Max results", "default": 10},
                    },
                    "required": ["query"],
                },
                executor=self._tool_search(self),
            ),
            ToolDef(
                name="memory_get",
                description="Get a single memory by ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID"},
                    },
                    "required": ["memory_id"],
                },
                executor=self._tool_get(self),
            ),
            ToolDef(
                name="memory_update",
                description="Update fields of an existing memory.",
                parameters={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory ID"},
                        "title": {"type": "string", "description": "New title"},
                        "content": {"type": "string", "description": "New content"},
                        "tags": {"type": "string", "description": "New tags"},
                        "importance": {"type": "number", "description": "New importance 0-1"},
                    },
                    "required": ["memory_id"],
                },
                executor=self._tool_update(self),
            ),
            ToolDef(
                name="memory_supersede",
                description="Mark an old memory as superseded by a new one.",
                parameters={
                    "type": "object",
                    "properties": {
                        "old_id": {"type": "string", "description": "ID of memory to supersede"},
                        "new_id": {"type": "string", "description": "ID of replacement memory"},
                    },
                    "required": ["old_id", "new_id"],
                },
                executor=self._tool_supersede(self),
            ),
            ToolDef(
                name="memory_list",
                description="List memories with optional filters.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "description": "Filter by scope"},
                        "kind": {"type": "string", "description": "Filter by kind"},
                        "limit": {"type": "integer", "description": "Max results", "default": 20},
                    },
                    "required": [],
                },
                executor=self._tool_list(self),
            ),
        ]


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------


def _vector_norm(v: Sequence[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0 if either vector is zero-length.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = _vector_norm(a)
    nb = _vector_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
