def test_memory_manager_init():
    from rose.storage.memory import MemoryManager
    mm = MemoryManager()
    assert mm is not None
