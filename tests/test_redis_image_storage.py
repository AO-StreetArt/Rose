import os
import sys
import unittest

import numpy as np
from PIL import Image

# Add the rose directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rose"))

from storage.redis_image_storage import RedisImageStorage  # noqa: E402


class TestRedisImageStorage(unittest.TestCase):
    """Test cases for RedisImageStorage class."""

    def setUp(self):
        """Set up test fixtures."""
        # Skip tests if Redis is not available
        try:
            self.storage = RedisImageStorage(host="localhost", port=6379, ttl_hours=1)
            self.storage.redis_client.ping()
        except Exception:
            self.skipTest("Redis server not available")

        # Create test images
        self.test_numpy_image = np.zeros((50, 50, 3), dtype=np.uint8)
        self.test_numpy_image[:, :] = [255, 0, 0]  # Red image

        self.test_pil_image = Image.new("RGB", (100, 75), color="blue")

        self.test_metadata = {
            "description": "Test image",
            "test_type": "unit_test",
            "version": "1.0",
        }

        self.test_tags = ["test", "red", "demo"]

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, "storage"):
            # Clean up any test data
            try:
                all_images = self.storage.list_all_images()
                for img in all_images:
                    self.storage.delete_image(img["key"])
            except Exception:
                pass
            finally:
                self.storage.close()

    def test_initialization(self):
        """Test RedisImageStorage initialization."""
        self.assertIsNotNone(self.storage)
        self.assertEqual(self.storage.ttl_hours, 1)
        self.assertEqual(self.storage.max_memory_mb, 1024)

    def test_store_and_retrieve_numpy_image(self):
        """Test storing and retrieving numpy array images."""
        # Store image
        key = self.storage.store_image(
            image_data=self.test_numpy_image,
            metadata=self.test_metadata,
            tags=self.test_tags,
        )

        self.assertIsNotNone(key)
        self.assertTrue(key.startswith("img:"))

        # Retrieve image
        retrieved_image = self.storage.retrieve_image(key, format="numpy")
        self.assertIsNotNone(retrieved_image)
        self.assertEqual(retrieved_image.shape, self.test_numpy_image.shape)
        np.testing.assert_array_equal(retrieved_image, self.test_numpy_image)

        # Clean up
        self.storage.delete_image(key)

    def test_store_and_retrieve_pil_image(self):
        """Test storing and retrieving PIL images."""
        # Store image
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags,
        )

        self.assertIsNotNone(key)

        # Retrieve as PIL
        retrieved_pil = self.storage.retrieve_image(key, format="PIL")
        self.assertIsNotNone(retrieved_pil)
        self.assertEqual(retrieved_pil.size, self.test_pil_image.size)
        self.assertEqual(retrieved_pil.mode, self.test_pil_image.mode)

        # Retrieve as numpy
        retrieved_numpy = self.storage.retrieve_image(key, format="numpy")
        self.assertIsNotNone(retrieved_numpy)
        self.assertEqual(retrieved_numpy.shape, (75, 100, 3))

        # Clean up
        self.storage.delete_image(key)

    def test_metadata_storage(self):
        """Test metadata storage and retrieval."""
        key = self.storage.store_image(
            image_data=self.test_numpy_image,
            metadata=self.test_metadata,
            tags=self.test_tags,
        )

        # Retrieve metadata
        retrieved_metadata = self.storage.get_metadata(key)
        self.assertIsNotNone(retrieved_metadata)

        # Check that stored metadata is preserved
        for key_name, value in self.test_metadata.items():
            self.assertIn(key_name, retrieved_metadata)
            self.assertEqual(retrieved_metadata[key_name], value)

        # Check that additional metadata was added
        self.assertIn("stored_at", retrieved_metadata)
        self.assertIn("image_size_bytes", retrieved_metadata)
        self.assertIn("tags", retrieved_metadata)

        # Clean up
        self.storage.delete_image(key)

    def test_tag_search(self):
        """Test searching images by tags."""
        # Store multiple images with different tags and unique content
        # Image 1: Red image
        image1 = np.zeros((50, 50, 3), dtype=np.uint8)
        image1[:, :] = [255, 0, 0]  # Red

        key1 = self.storage.store_image(
            image_data=image1, metadata={"description": "Image 1"}, tags=["red", "test"]
        )

        # Image 2: Blue image
        image2 = np.zeros((50, 50, 3), dtype=np.uint8)
        image2[:, :] = [0, 0, 255]  # Blue

        key2 = self.storage.store_image(
            image_data=image2,
            metadata={"description": "Image 2"},
            tags=["blue", "test"],
        )

        # Image 3: Green image
        image3 = np.zeros((50, 50, 3), dtype=np.uint8)
        image3[:, :] = [0, 255, 0]  # Green

        key3 = self.storage.store_image(
            image_data=image3, metadata={"description": "Image 3"}, tags=["red", "blue"]
        )

        # Search by single tag
        red_images = self.storage.search_by_tags(["red"])
        self.assertEqual(len(red_images), 2)  # key1 and key3

        # Search by multiple tags with OR operator
        test_images = self.storage.search_by_tags(["test"], operator="OR")
        self.assertEqual(len(test_images), 2)  # key1 and key2

        # Search by multiple tags with AND operator
        red_test_images = self.storage.search_by_tags(["red", "test"], operator="AND")
        self.assertEqual(len(red_test_images), 1)  # only key1

        # Clean up
        self.storage.delete_image(key1)
        self.storage.delete_image(key2)
        self.storage.delete_image(key3)

    def test_list_all_images(self):
        """Test listing all stored images."""
        # Store a few images with unique content
        keys = []
        for i in range(3):
            # Create unique image for each iteration
            unique_image = np.zeros((50, 50, 3), dtype=np.uint8)
            unique_image[:, :] = [255, i * 50, 0]  # Different colors for each image

            key = self.storage.store_image(
                image_data=unique_image,
                metadata={"description": f"Image {i}"},
                tags=[f"test_{i}"],
            )
            keys.append(key)

        # List all images
        all_images = self.storage.list_all_images()
        self.assertGreaterEqual(len(all_images), 3)

        # Check that our test images are included
        stored_keys = [img["key"] for img in all_images]
        for key in keys:
            self.assertIn(key, stored_keys)

        # Clean up
        for key in keys:
            self.storage.delete_image(key)

    def test_delete_image(self):
        """Test image deletion."""
        # Store an image
        key = self.storage.store_image(
            image_data=self.test_numpy_image,
            metadata=self.test_metadata,
            tags=self.test_tags,
        )

        # Verify it exists
        retrieved_image = self.storage.retrieve_image(key)
        self.assertIsNotNone(retrieved_image)

        # Delete the image
        success = self.storage.delete_image(key)
        self.assertTrue(success)

        # Verify it's gone
        retrieved_image = self.storage.retrieve_image(key)
        self.assertIsNone(retrieved_image)

        # Verify metadata is gone
        metadata = self.storage.get_metadata(key)
        self.assertIsNone(metadata)

    def test_storage_stats(self):
        """Test storage statistics retrieval."""
        # Store a few images with unique content
        keys = []
        for i in range(2):
            # Create unique image for each iteration
            unique_image = np.zeros((50, 50, 3), dtype=np.uint8)
            unique_image[:, :] = [0, 255, i * 100]  # Different colors for each image

            key = self.storage.store_image(
                image_data=unique_image,
                metadata={"description": f"Image {i}"},
                tags=[f"test_{i}", "demo"],
            )
            keys.append(key)

        # Get stats
        stats = self.storage.get_storage_stats()

        # Check basic stats
        self.assertIn("total_images", stats)
        self.assertIn("total_size_mb", stats)
        self.assertIn("tag_counts", stats)

        # Verify image count
        self.assertGreaterEqual(stats["total_images"], 2)

        # Verify tag counts
        self.assertIn("demo", stats["tag_counts"])
        self.assertGreaterEqual(stats["tag_counts"]["demo"], 2)

        # Clean up
        for key in keys:
            self.storage.delete_image(key)

    def test_context_manager(self):
        """Test context manager functionality."""
        with RedisImageStorage(host="localhost", port=6379) as storage:
            # Store an image
            key = storage.store_image(
                image_data=self.test_numpy_image,
                metadata=self.test_metadata,
                tags=self.test_tags,
            )

            # Verify it was stored
            retrieved_image = storage.retrieve_image(key)
            self.assertIsNotNone(retrieved_image)

            # Clean up
            storage.delete_image(key)

        # Connection should be closed automatically


if __name__ == "__main__":
    # Check if Redis is available before running tests
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379)
        r.ping()
        print("Redis server available. Running tests...")
        unittest.main()
    except Exception as e:
        print(f"Redis server not available: {e}")
        print("Skipping Redis tests. Make sure Redis is running on localhost:6379")
