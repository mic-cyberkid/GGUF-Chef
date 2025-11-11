import os
import pytest
from memory_manager import MemoryManager

@pytest.fixture
def mem():
    path = "test_memory.pkl"
    if os.path.exists(path):
        os.remove(path)
    m = MemoryManager(memory_path=path, memory_limit=5)
    yield m
    if os.path.exists(path):
        os.remove(path)

def test_add_and_trim(mem):
    for i in range(7):
        mem.add_message("user", f"msg {i}")
    assert len(mem.recent_history) == 5
    assert mem.recent_history[-1]["content"] == "msg 6"
