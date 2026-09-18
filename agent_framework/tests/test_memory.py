"""Tests for MemoryStore — CRUD, FTS5 search, supersede, tool definitions."""

from __future__ import annotations

from agent_framework.memory import Memory, MemoryStore


class TestMemoryStoreCRUD:
    def setup_method(self):
        self.store = MemoryStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_save_and_get(self):
        mid = self.store.save("Test Title", "Test content")
        mem = self.store.get(mid)
        assert mem is not None
        assert mem.title == "Test Title"
        assert mem.content == "Test content"
        assert mem.scope == "global"
        assert mem.kind == "fact"

    def test_save_with_custom_fields(self):
        mid = self.store.save(
            "Custom", "content",
            scope="project", kind="decision",
            tags="important urgent", importance=0.9, source="user",
        )
        mem = self.store.get(mid)
        assert mem.scope == "project"
        assert mem.kind == "decision"
        assert mem.importance == 0.9

    def test_save_custom_id(self):
        mid = self.store.save("Fixed ID", "content", memory_id="my-id-123")
        assert mid == "my-id-123"
        assert self.store.get("my-id-123") is not None

    def test_update(self):
        mid = self.store.save("Old", "old content")
        ok = self.store.update(mid, title="New", content="new content")
        assert ok
        mem = self.store.get(mid)
        assert mem.title == "New"
        assert mem.content == "new content"

    def test_update_nonexistent(self):
        ok = self.store.update("fake-id", title="x")
        assert not ok

    def test_delete(self):
        mid = self.store.save("Delete Me", "gone")
        assert self.store.delete(mid)
        assert self.store.get(mid) is None

    def test_delete_nonexistent(self):
        assert not self.store.delete("fake-id")

    def test_list_default(self):
        self.store.save("A", "content A")
        self.store.save("B", "content B")
        mems = self.store.list_memories()
        assert len(mems) == 2

    def test_list_filter_by_scope(self):
        self.store.save("Global", "x", scope="global")
        self.store.save("Project", "y", scope="project")
        mems = self.store.list_memories(scope="project")
        assert len(mems) == 1
        assert mems[0].title == "Project"

    def test_list_filter_by_tags(self):
        self.store.save("Tagged", "x", tags="foo bar")
        self.store.save("Untagged", "y", tags="baz")
        mems = self.store.list_memories(tags="foo")
        assert len(mems) == 1

    def test_list_limit(self):
        for i in range(10):
            self.store.save(f"M{i}", f"c{i}")
        mems = self.store.list_memories(limit=3)
        assert len(mems) == 3


class TestMemoryStoreSupersede:
    def setup_method(self):
        self.store = MemoryStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_supersede(self):
        old = self.store.save("Old version", "v1")
        new = self.store.save("New version", "v2")
        ok = self.store.supersede(old, new)
        assert ok

        old_mem = self.store.get(old)
        assert old_mem.superseded_by == new

    def test_superseded_excluded_from_list(self):
        old = self.store.save("Old", "v1")
        self.store.save("New", "v2")
        self.store.supersede(old, "some-new-id")

        mems = self.store.list_memories(include_superseded=False)
        ids = [m.id for m in mems]
        assert old not in ids

    def test_superseded_included_when_requested(self):
        old = self.store.save("Old", "v1")
        self.store.save("New", "v2")
        self.store.supersede(old, "some-new-id")

        mems = self.store.list_memories(include_superseded=True)
        ids = [m.id for m in mems]
        assert old in ids


class TestMemoryStoreSearch:
    def setup_method(self):
        self.store = MemoryStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_fts5_search(self):
        self.store.save("Python Guide", "Learn Python programming basics", tags="python tutorial")
        self.store.save("Cooking Tips", "How to cook pasta perfectly", tags="food cooking")

        results = self.store.search("Python")
        assert len(results) >= 1
        assert results[0].title == "Python Guide"

    def test_search_respects_superseded(self):
        old = self.store.save("Old Fact", "outdated info about X")
        self.store.save("New Fact", "updated info about X")
        self.store.supersede(old, "new-id")

        results = self.store.search("about X")
        for r in results:
            assert r.id != old, "Superseded memory should not appear"

    def test_search_empty_store(self):
        results = self.store.search("nothing")
        assert results == []


class TestMemoryStoreEmbeddings:
    def setup_method(self):
        self.store = MemoryStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_save_with_embedding(self):
        mid = self.store.save("Embedded", "content", embedding=[0.1, 0.2, 0.3])
        mem = self.store.get(mid)
        assert mem.embedding == [0.1, 0.2, 0.3]

    def test_cosine_similarity_search(self):
        self.store.save("A", "c", embedding=[1.0, 0.0])
        self.store.save("B", "c", embedding=[0.0, 1.0])
        self.store.save("C", "c", embedding=[1.0, 0.1])

        results = self.store.search_embeddings([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0].title == "A"  # exact match first
        # Score = cosine_sim (1.0) * importance (0.5 default) * superseded_penalty (1.0)
        assert results[0][1] == 0.5

    def test_cosine_empty_query(self):
        self.store.save("A", "c", embedding=[0.0, 0.0])
        results = self.store.search_embeddings([0.0, 0.0])
        assert results == []


class TestMemoryStoreTools:
    def setup_method(self):
        self.store = MemoryStore(":memory:")

    def teardown_method(self):
        self.store.close()

    def test_get_tools_returns_all_six(self):
        tools = self.store.get_tools()
        names = [t.name for t in tools]
        assert "memory_save" in names
        assert "memory_search" in names
        assert "memory_get" in names
        assert "memory_update" in names
        assert "memory_supersede" in names
        assert "memory_list" in names

    def test_tool_save(self):
        tools = self.store.get_tools()
        save_tool = next(t for t in tools if t.name == "memory_save")
        result = save_tool.executor(title="T", content="C")
        assert self.store.get(result) is not None

    def test_tool_search(self):
        self.store.save("Find Me", "searchable content here")
        tools = self.store.get_tools()
        search_tool = next(t for t in tools if t.name == "memory_search")
        result = search_tool.executor(query="searchable")
        assert "Find Me" in result

    def test_tool_get(self):
        mid = self.store.save("Get Me", "content")
        tools = self.store.get_tools()
        get_tool = next(t for t in tools if t.name == "memory_get")
        result = get_tool.executor(memory_id=mid)
        assert "Get Me" in result

    def test_tool_get_not_found(self):
        tools = self.store.get_tools()
        get_tool = next(t for t in tools if t.name == "memory_get")
        result = get_tool.executor(memory_id="nonexistent")
        assert "not found" in result

    def test_tool_update(self):
        mid = self.store.save("Before", "old")
        tools = self.store.get_tools()
        update_tool = next(t for t in tools if t.name == "memory_update")
        result = update_tool.executor(memory_id=mid, title="After")
        assert "Updated" in result

    def test_tool_supersede(self):
        old = self.store.save("Old", "v1")
        new = self.store.save("New", "v2")
        tools = self.store.get_tools()
        sup_tool = next(t for t in tools if t.name == "memory_supersede")
        result = sup_tool.executor(old_id=old, new_id=new)
        assert "superseded" in result

    def test_tool_list(self):
        self.store.save("L1", "c1")
        self.store.save("L2", "c2")
        tools = self.store.get_tools()
        list_tool = next(t for t in tools if t.name == "memory_list")
        result = list_tool.executor(limit=10)
        assert "L1" in result
        assert "L2" in result


class TestMemory:
    def test_dataclass_fields(self):
        m = Memory(id="test", title="T", content="C")
        assert m.id == "test"
        assert m.scope == "global"
        assert m.importance == 0.5
