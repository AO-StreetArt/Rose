import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rose.storage.memory_image_storage import MemoryImageStorage


class TestMemoryImageStorage(unittest.TestCase):
    """Test cases for MemoryImageStorage class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.storage = MemoryImageStorage(max_memory_mb=10, ttl_hours=1, cleanup_interval=5)
        
        # Create test images
        self.test_image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_pil_image = Image.fromarray(self.test_image_array)
        self.test_bytes = self.test_pil_image.tobytes()
        
        # Test metadata
        self.test_metadata = {
            "source": "test_image",
            "width": 100,
            "height": 100,
            "channels": 3
        }
        
        # Test tags
        self.test_tags = ["test", "image", "unit_test"]

    def tearDown(self):
        """Clean up after each test method."""
        self.storage.clear_all()

    def test_init(self):
        """Test initialization of MemoryImageStorage."""
        self.assertEqual(self.storage.max_memory_mb, 10)
        self.assertEqual(self.storage.ttl_hours, 1)
        self.assertEqual(self.storage.cleanup_interval, 5)
        self.assertEqual(self.storage.operation_count, 0)
        self.assertEqual(len(self.storage), 0)

    def test_generate_image_key(self):
        """Test image key generation."""
        # Test with numpy array
        key1 = self.storage._generate_image_key(self.test_image_array)
        self.assertIsInstance(key1, str)
        self.assertTrue(len(key1) > 0)
        
        # Test with PIL image
        key2 = self.storage._generate_image_key(self.test_pil_image)
        self.assertIsInstance(key2, str)
        
        # Test with bytes
        key3 = self.storage._generate_image_key(self.test_bytes)
        self.assertIsInstance(key3, str)
        
        # Test with prefix
        key4 = self.storage._generate_image_key(self.test_image_array, "test_prefix")
        self.assertTrue(key4.startswith("test_prefix_"))

    def test_encode_image(self):
        """Test image encoding."""
        # Test numpy array encoding
        encoded_array = self.storage._encode_image(self.test_image_array)
        self.assertIsInstance(encoded_array, bytes)
        self.assertTrue(len(encoded_array) > 0)
        
        # Test PIL image encoding
        encoded_pil = self.storage._encode_image(self.test_pil_image)
        self.assertIsInstance(encoded_pil, bytes)
        
        # Test bytes encoding (should return as-is)
        encoded_bytes = self.storage._encode_image(self.test_bytes)
        self.assertEqual(encoded_bytes, self.test_bytes)

    def test_decode_image(self):
        """Test image decoding."""
        encoded_image = self.storage._encode_image(self.test_pil_image)
        
        # Test PIL format
        decoded_pil = self.storage._decode_image(encoded_image, "PIL")
        self.assertIsInstance(decoded_pil, Image.Image)
        
        # Test numpy format
        decoded_numpy = self.storage._decode_image(encoded_image, "numpy")
        self.assertIsInstance(decoded_numpy, np.ndarray)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.storage._decode_image(encoded_image, "invalid_format")

    def test_store_image_with_auto_key(self):
        """Test storing image with auto-generated key."""
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags
        )
        
        self.assertIsInstance(key, str)
        self.assertIn(key, self.storage)
        self.assertEqual(len(self.storage), 1)
        
        # Check metadata
        stored_metadata = self.storage.get_metadata(key)
        self.assertIsNotNone(stored_metadata)
        self.assertEqual(stored_metadata["source"], "test_image")
        self.assertIn("stored_at", stored_metadata)
        self.assertIn("image_size_bytes", stored_metadata)
        self.assertEqual(stored_metadata["tags"], self.test_tags)

    def test_store_image_with_custom_key(self):
        """Test storing image with custom key."""
        custom_key = "custom_test_key"
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags,
            key=custom_key
        )
        
        self.assertEqual(key, custom_key)
        self.assertIn(custom_key, self.storage)

    def test_store_image_with_ttl(self):
        """Test storing image with TTL."""
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags,
            ttl_hours=0.001  # Very short TTL for testing
        )
        
        self.assertIn(key, self.storage.expiry_times)
        
        # Wait for expiration (simulate time passing)
        with patch('rose.storage.memory_image_storage.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(hours=1)
            result = self.storage.retrieve_image(key)
            self.assertIsNone(result)

    def test_store_image_without_tags(self):
        """Test storing image without tags."""
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata
        )
        
        self.assertIn(key, self.storage)
        metadata = self.storage.get_metadata(key)
        self.assertEqual(metadata["tags"], [])

    def test_retrieve_image(self):
        """Test image retrieval."""
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags
        )
        
        # Test PIL format retrieval
        retrieved_pil = self.storage.retrieve_image(key, "PIL")
        self.assertIsInstance(retrieved_pil, Image.Image)
        
        # Test numpy format retrieval
        retrieved_numpy = self.storage.retrieve_image(key, "numpy")
        self.assertIsInstance(retrieved_numpy, np.ndarray)
        
        # Test retrieval of non-existent key
        non_existent = self.storage.retrieve_image("non_existent_key")
        self.assertIsNone(non_existent)

    def test_get_metadata(self):
        """Test metadata retrieval."""
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags
        )
        
        metadata = self.storage.get_metadata(key)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["source"], "test_image")
        self.assertEqual(metadata["width"], 100)
        
        # Test metadata for non-existent key
        non_existent_metadata = self.storage.get_metadata("non_existent_key")
        self.assertIsNone(non_existent_metadata)

    def test_search_by_tags_or_operator(self):
        """Test tag search with OR operator."""
        # Store multiple images with different tags
        key1 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image1"},
            tags=["tag1", "tag2"]
        )
        
        key2 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image2"},
            tags=["tag2", "tag3"]
        )
        
        key3 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image3"},
            tags=["tag3", "tag4"]
        )
        
        # Search with OR operator
        results = self.storage.search_by_tags(["tag1", "tag3"], "OR")
        self.assertEqual(len(results), 3)  # All images should match
        
        # Search with single tag
        results = self.storage.search_by_tags(["tag2"], "OR")
        self.assertEqual(len(results), 2)  # image1 and image2 should match

    def test_search_by_tags_and_operator(self):
        """Test tag search with AND operator."""
        # Store multiple images with different tags
        key1 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image1"},
            tags=["tag1", "tag2", "tag3"]
        )
        
        key2 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image2"},
            tags=["tag2", "tag3"]
        )
        
        key3 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image3"},
            tags=["tag1", "tag3"]
        )
        
        # Search with AND operator
        results = self.storage.search_by_tags(["tag2", "tag3"], "AND")
        self.assertEqual(len(results), 2)  # image1 and image2 should match
        
        # Search with non-matching tags
        results = self.storage.search_by_tags(["tag1", "tag4"], "AND")
        self.assertEqual(len(results), 0)

    def test_search_by_tags_invalid_operator(self):
        """Test tag search with invalid operator."""
        with self.assertRaises(ValueError):
            self.storage.search_by_tags(["tag1"], "INVALID")

    def test_list_all_images(self):
        """Test listing all images."""
        # Store multiple images
        for i in range(5):
            self.storage.store_image(
                image_data=self.test_pil_image,
                metadata={"id": f"image{i}"},
                tags=[f"tag{i}"]
            )
        
        # Test without limit
        all_images = self.storage.list_all_images()
        self.assertEqual(len(all_images), 5)
        
        # Test with limit
        limited_images = self.storage.list_all_images(limit=3)
        self.assertEqual(len(limited_images), 3)
        
        # Test with limit larger than total
        large_limit_images = self.storage.list_all_images(limit=10)
        self.assertEqual(len(large_limit_images), 5)

    def test_delete_image(self):
        """Test image deletion."""
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags
        )
        
        self.assertIn(key, self.storage)
        
        # Delete the image
        result = self.storage.delete_image(key)
        self.assertTrue(result)
        self.assertNotIn(key, self.storage)
        self.assertEqual(len(self.storage), 0)
        
        # Test deletion of non-existent key
        result = self.storage.delete_image("non_existent_key")
        self.assertFalse(result)

    def test_cleanup_expired(self):
        """Test cleanup of expired images."""
        # Store image with very short TTL
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags,
            ttl_hours=0.001
        )
        
        self.assertIn(key, self.storage)
        
        # Simulate time passing and cleanup
        with patch('rose.storage.memory_image_storage.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(hours=1)
            cleaned_count = self.storage.cleanup_expired()
            self.assertEqual(cleaned_count, 1)
            self.assertNotIn(key, self.storage)

    def test_get_storage_stats(self):
        """Test storage statistics."""
        # Store some images
        for i in range(3):
            self.storage.store_image(
                image_data=self.test_pil_image,
                metadata={"id": f"image{i}"},
                tags=[f"tag{i}"]
            )
        
        stats = self.storage.get_storage_stats()
        
        self.assertIn("total_images", stats)
        self.assertIn("total_size_mb", stats)
        self.assertIn("tag_counts", stats)
        self.assertIn("memory_usage_mb", stats)
        self.assertIn("operation_count", stats)
        
        self.assertEqual(stats["total_images"], 3)
        self.assertEqual(stats["operation_count"], 3)
        self.assertGreater(stats["total_size_mb"], 0)

    def test_transfer_to_redis(self):
        """Test transfer to Redis storage."""
        # Store some images
        key1 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image1"},
            tags=["tag1", "tag2"]
        )
        
        key2 = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata={"id": "image2"},
            tags=["tag2", "tag3"]
        )
        
        # Mock Redis storage
        mock_redis = Mock()
        mock_redis.store_image = Mock(return_value="mock_key")
        
        # Transfer to Redis
        transferred_count = self.storage.transfer_to_redis(mock_redis)
        self.assertEqual(transferred_count, 2)
        
        # Verify Redis store_image was called for each image
        self.assertEqual(mock_redis.store_image.call_count, 2)

    def test_clear_all(self):
        """Test clearing all images."""
        # Store some images
        for i in range(5):
            self.storage.store_image(
                image_data=self.test_pil_image,
                metadata={"id": f"image{i}"},
                tags=[f"tag{i}"]
            )
        
        self.assertEqual(len(self.storage), 5)
        
        # Clear all
        cleared_count = self.storage.clear_all()
        self.assertEqual(cleared_count, 5)
        self.assertEqual(len(self.storage), 0)
        self.assertEqual(self.storage.operation_count, 0)

    def test_memory_limit_check(self):
        """Test memory limit checking."""
        # Create storage with very low memory limit
        low_memory_storage = MemoryImageStorage(max_memory_mb=0.001)
        
        # Store an image that exceeds the limit
        with patch.object(low_memory_storage, '_cleanup_expired') as mock_cleanup:
            low_memory_storage.store_image(
                image_data=self.test_pil_image,
                metadata=self.test_metadata,
                tags=self.test_tags
            )
            
            # Check if cleanup was called
            mock_cleanup.assert_called()

    def test_operation_count_increment(self):
        """Test operation count increment."""
        initial_count = self.storage.operation_count
        
        # Store an image
        self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags
        )
        
        self.assertEqual(self.storage.operation_count, initial_count + 1)

    def test_len_and_contains(self):
        """Test length and contains methods."""
        self.assertEqual(len(self.storage), 0)
        self.assertFalse("test_key" in self.storage)
        
        # Store an image
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=self.test_tags
        )
        
        self.assertEqual(len(self.storage), 1)
        self.assertTrue(key in self.storage)

    def test_error_handling(self):
        """Test error handling in various methods."""
        # Test store_image with invalid image data
        with self.assertRaises(Exception):
            self.storage.store_image(
                image_data="invalid_image_data",
                metadata=self.test_metadata
            )
        
        # Test retrieve_image with invalid format
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata
        )
        
        with self.assertRaises(ValueError):
            self.storage.retrieve_image(key, "invalid_format")

    def test_tag_indexing_cleanup(self):
        """Test that empty tag sets are cleaned up."""
        key = self.storage.store_image(
            image_data=self.test_pil_image,
            metadata=self.test_metadata,
            tags=["unique_tag"]
        )
        
        # Verify tag is indexed
        self.assertIn("unique_tag", self.storage.tags)
        self.assertIn(key, self.storage.tags["unique_tag"])
        
        # Delete the image
        self.storage.delete_image(key)
        
        # Verify tag set is cleaned up
        self.assertNotIn("unique_tag", self.storage.tags)


if __name__ == "__main__":
    unittest.main()
