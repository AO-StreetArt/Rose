import unittest
from rose.storage.memory import MemoryManager


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.memory_manager = MemoryManager()

    def test_memory_manager_init(self):
        self.assertIsNotNone(self.memory_manager)
