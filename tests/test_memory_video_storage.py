#!/usr/bin/env python3
"""
Unit tests for the MemoryVideoStorage class.

This module tests all functionality of the in-memory video storage system including:
- Video storage and retrieval
- Metadata management
- Tag-based search
- Timestamp-based search
- TTL and cleanup functionality
- Error handling
"""

import pytest
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from rose.storage.memory_video_storage import MemoryVideoStorage


class TestMemoryVideoStorage:
    """Test cases for the MemoryVideoStorage class."""

    @pytest.fixture
    def storage(self):
        """Create a fresh MemoryVideoStorage instance for each test."""
        return MemoryVideoStorage(
            max_memory_mb=100,
            ttl_hours=24,
            cleanup_interval=10,
            max_video_size_mb=10
        )

    @pytest.fixture
    def sample_video_data(self):
        """Create sample video data for testing."""
        # Create a simple video-like header followed by random data
        header = b'\x00\x00\x00\x18ftypmp4'  # MP4 header
        random_data = os.urandom(1024)  # 1KB of random data
        return header + random_data

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "title": "Test Video",
            "description": "A test video for unit testing",
            "duration": 30.5,
            "resolution": "1920x1080",
            "fps": 30.0,
            "source": "test_camera"
        }

    @pytest.fixture
    def sample_tags(self):
        """Create sample tags for testing."""
        return ["test", "demo", "video"]

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        storage = MemoryVideoStorage()
        
        assert storage.max_memory_mb == 2048
        assert storage.ttl_hours == 48
        assert storage.cleanup_interval == 50
        assert storage.max_video_size_mb == 100
        assert storage.operation_count == 0
        assert len(storage.videos) == 0
        assert len(storage.metadata) == 0

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        storage = MemoryVideoStorage(
            max_memory_mb=512,
            ttl_hours=12,
            cleanup_interval=25,
            max_video_size_mb=50
        )
        
        assert storage.max_memory_mb == 512
        assert storage.ttl_hours == 12
        assert storage.cleanup_interval == 25
        assert storage.max_video_size_mb == 50

    def test_generate_video_key_with_filename(self, storage, sample_video_data):
        """Test video key generation with filename."""
        filename = "test_video.mp4"
        key = storage._generate_video_key(sample_video_data, filename)
        
        assert key.startswith("test_video.mp4_")
        assert len(key) > 20
        assert "_" in key

    def test_generate_video_key_without_filename(self, storage, sample_video_data):
        """Test video key generation without filename."""
        key = storage._generate_video_key(sample_video_data)
        
        assert not key.startswith("test_video")
        assert len(key) > 10
        assert "_" in key

    def test_generate_video_key_with_prefix(self, storage, sample_video_data):
        """Test video key generation with custom prefix."""
        prefix = "custom_"
        key = storage._generate_video_key(sample_video_data, prefix=prefix)
        
        assert key.startswith(prefix)
        assert len(key) > len(prefix) + 10

    def test_generate_video_key_with_file_like_object(self, storage):
        """Test video key generation with file-like object."""
        # Create a mock file-like object
        mock_file = Mock()
        mock_file.read.return_value = b'test_video_data'
        mock_file.seek = Mock()
        
        key = storage._generate_video_key(mock_file)
        
        assert len(key) > 10
        mock_file.read.assert_called_once()
        mock_file.seek.assert_called_once_with(0)

    def test_validate_video_data_valid(self, storage, sample_video_data):
        """Test validation of valid video data."""
        result = storage._validate_video_data(sample_video_data)
        
        assert result == sample_video_data
        assert len(result) > 0

    def test_validate_video_data_empty(self, storage):
        """Test validation of empty video data."""
        with pytest.raises(ValueError, match="Video data cannot be empty"):
            storage._validate_video_data(b"")

    def test_validate_video_data_too_large(self, storage):
        """Test validation of video data that exceeds size limit."""
        large_data = os.urandom(15 * 1024 * 1024)  # 15MB, exceeds 10MB limit
        
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            storage._validate_video_data(large_data)

    def test_validate_video_data_file_like_object(self, storage):
        """Test validation of file-like object."""
        mock_file = Mock()
        mock_file.read.return_value = b'test_video_data'
        mock_file.seek = Mock()
        
        result = storage._validate_video_data(mock_file)
        
        assert result == b'test_video_data'
        mock_file.read.assert_called_once()
        mock_file.seek.assert_called_once_with(0)

    def test_detect_mime_type_with_filename(self, storage):
        """Test MIME type detection with filename."""
        video_data = b'test_data'
        filename = "test.mp4"
        
        mime_type = storage._detect_mime_type(video_data, filename)
        
        assert mime_type == "video/mp4"

    def test_detect_mime_type_without_filename(self, storage):
        """Test MIME type detection without filename."""
        # Test MP4 detection
        mp4_data = b'\x00\x00\x00\x18ftypmp4' + b'data'
        mime_type = storage._detect_mime_type(mp4_data)
        
        assert mime_type == "video/mp4"

    def test_detect_mime_type_webm(self, storage):
        """Test MIME type detection for WebM format."""
        webm_data = b'\x1a\x45\xdf\xa3' + b'data'
        mime_type = storage._detect_mime_type(webm_data)
        
        assert mime_type == "video/webm"

    def test_detect_mime_type_avi(self, storage):
        """Test MIME type detection for AVI format."""
        avi_data = b'RIFF' + b'data' + b'AVI ' + b'more_data'
        mime_type = storage._detect_mime_type(avi_data)
        
        assert mime_type == "video/x-msvideo"

    def test_detect_mime_type_mpeg(self, storage):
        """Test MIME type detection for MPEG format."""
        mpeg_data = b'\x00\x00\x01\xb3' + b'data'
        mime_type = storage._detect_mime_type(mpeg_data)
        
        assert mime_type == "video/mpeg"

    def test_detect_mime_type_unknown(self, storage):
        """Test MIME type detection for unknown format."""
        unknown_data = b'unknown_video_format_data'
        mime_type = storage._detect_mime_type(unknown_data)
        
        assert mime_type == "video/octet-stream"

    def test_store_video_success(self, storage, sample_video_data, sample_metadata, sample_tags):
        """Test successful video storage."""
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            tags=sample_tags
        )
        
        assert key in storage.videos
        assert key in storage.metadata
        assert storage.videos[key] == sample_video_data
        
        # Check metadata
        stored_metadata = storage.metadata[key]
        assert stored_metadata["title"] == sample_metadata["title"]
        assert stored_metadata["stored_at"] is not None
        assert stored_metadata["video_size_bytes"] == len(sample_video_data)
        assert stored_metadata["mime_type"] == "video/mp4"
        assert stored_metadata["tags"] == sample_tags
        
        # Check tags
        for tag in sample_tags:
            assert key in storage.tags[tag]
        
        # Check global index
        assert key in storage.all_keys

    def test_store_video_with_custom_key(self, storage, sample_video_data, sample_metadata):
        """Test video storage with custom key."""
        custom_key = "custom_video_key"
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            key=custom_key
        )
        
        assert key == custom_key
        assert custom_key in storage.videos

    def test_store_video_with_filename(self, storage, sample_video_data, sample_metadata):
        """Test video storage with filename."""
        filename = "test_video.mp4"
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            filename=filename
        )
        
        assert storage.metadata[key]["filename"] == filename

    def test_store_video_with_custom_ttl(self, storage, sample_video_data, sample_metadata):
        """Test video storage with custom TTL."""
        custom_ttl = 2  # 2 hours
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            ttl_hours=custom_ttl
        )
        
        assert key in storage.expiry_times
        expiry_time = storage.expiry_times[key]
        expected_expiry = datetime.now().replace(microsecond=0) + timedelta(hours=custom_ttl)
        
        # Allow for small time differences
        time_diff = abs((expiry_time - expected_expiry).total_seconds())
        assert time_diff < 5

    def test_store_video_without_ttl(self, storage, sample_video_data, sample_metadata):
        """Test video storage without TTL."""
        storage.ttl_hours = 0
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata
        )
        
        assert key not in storage.expiry_times

    def test_store_video_invalid_data(self, storage, sample_metadata):
        """Test video storage with invalid data."""
        with pytest.raises(ValueError):
            storage.store_video(
                video_data=b"",  # Empty data
                metadata=sample_metadata
            )

    def test_store_video_data_too_large(self, storage, sample_metadata):
        """Test video storage with data that exceeds size limit."""
        large_data = os.urandom(15 * 1024 * 1024)  # 15MB
        
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            storage.store_video(
                video_data=large_data,
                metadata=sample_metadata
            )

    def test_retrieve_video_success(self, storage, sample_video_data, sample_metadata):
        """Test successful video retrieval."""
        key = storage.store_video(sample_video_data, sample_metadata)
        
        # Retrieve video data
        retrieved_data = storage.retrieve_video(key)
        assert retrieved_data == sample_video_data
        
        # Retrieve metadata
        retrieved_metadata = storage.retrieve_video(key, format="metadata")
        assert retrieved_metadata["title"] == sample_metadata["title"]

    def test_retrieve_video_not_found(self, storage):
        """Test video retrieval for non-existent key."""
        result = storage.retrieve_video("non_existent_key")
        assert result is None

    def test_retrieve_video_expired(self, storage, sample_video_data, sample_metadata):
        """Test video retrieval for expired video."""
        # Store with very short TTL
        key = storage.store_video(
            sample_video_data, 
            sample_metadata, 
            ttl_hours=0.001  # Very short TTL
        )
        
        # Manually set expiry time to past to trigger cleanup
        storage.expiry_times[key] = datetime.now() - timedelta(hours=1)
        
        # Should return None and clean up
        result = storage.retrieve_video(key)
        assert result is None
        assert key not in storage.videos

    def test_search_by_tags_or_operator(self, storage, sample_video_data, sample_metadata):
        """Test tag search with OR operator."""
        # Store videos with different tags
        key1 = storage.store_video(
            sample_video_data, 
            sample_metadata, 
            tags=["tag1", "tag2"]
        )
        
        key2 = storage.store_video(
            sample_video_data, 
            sample_metadata, 
            tags=["tag2", "tag3"]
        )
        
        # Search for videos with tag1 OR tag3
        results = storage.search_by_tags(["tag1", "tag3"])
        
        assert len(results) == 2
        keys = [r["key"] for r in results]
        assert key1 in keys
        assert key2 in keys

    def test_search_by_tags_and_operator(self, storage, sample_video_data, sample_metadata):
        """Test tag search with AND operator."""
        # Store videos with different tags
        key1 = storage.store_video(
            sample_video_data, 
            sample_metadata, 
            tags=["tag1", "tag2", "tag3"]
        )
        
        key2 = storage.store_video(
            sample_video_data, 
            sample_metadata, 
            tags=["tag2", "tag3"]
        )
        
        # Search for videos with tag1 AND tag2 AND tag3
        results = storage.search_by_tags(["tag1", "tag2", "tag3"], operator="AND")
        
        assert len(results) == 1
        assert results[0]["key"] == key1

    def test_search_by_tags_no_matches(self, storage):
        """Test tag search with no matches."""
        results = storage.search_by_tags(["nonexistent_tag"])
        assert len(results) == 0

    def test_search_by_timestamp_range(self, storage, sample_video_data, sample_metadata):
        """Test timestamp range search."""
        # Store videos at different times
        key1 = storage.store_video(sample_video_data, sample_metadata)
        
        # Small delay to ensure different timestamps
        from time import sleep
        sleep(0.1)
        
        key2 = storage.store_video(sample_video_data, sample_metadata)
        
        # Search for videos in the last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        results = storage.search_by_timestamp_range(start_time=one_hour_ago)
        
        assert len(results) == 2
        keys = [r["key"] for r in results]
        assert key1 in keys
        assert key2 in keys

    def test_search_by_timestamp_range_with_end_time(self, storage, sample_video_data, sample_metadata):
        """Test timestamp range search with end time."""
        # Store a video
        key = storage.store_video(sample_video_data, sample_metadata)
        
        # Search for videos in the next hour (should find none)
        one_hour_later = datetime.now() + timedelta(hours=1)
        results = storage.search_by_timestamp_range(end_time=one_hour_later)
        
        assert len(results) == 1
        assert results[0]["key"] == key

    def test_list_all_videos(self, storage, sample_video_data, sample_metadata):
        """Test listing all videos."""
        # Store multiple videos
        key1 = storage.store_video(sample_video_data, sample_metadata)
        key2 = storage.store_video(sample_video_data, sample_metadata)
        
        all_videos = storage.list_all_videos()
        
        assert len(all_videos) == 2
        keys = [v["key"] for v in all_videos]
        assert key1 in keys
        assert key2 in keys

    def test_update_metadata_success(self, storage, sample_video_data, sample_metadata):
        """Test successful metadata update."""
        key = storage.store_video(sample_video_data, sample_metadata)
        
        new_metadata = {
            "title": "Updated Title",
            "description": "Updated description"
        }
        
        success = storage.update_metadata(key, new_metadata)
        assert success is True
        
        # Check updated metadata
        updated_metadata = storage.metadata[key]
        assert updated_metadata["title"] == "Updated Title"
        assert updated_metadata["description"] == "Updated description"
        assert "updated_at" in updated_metadata

    def test_update_metadata_with_tags(self, storage, sample_video_data, sample_metadata):
        """Test metadata update with new tags."""
        key = storage.store_video(sample_video_data, sample_metadata, tags=["old_tag"])
        
        new_metadata = {
            "tags": ["new_tag1", "new_tag2"]
        }
        
        success = storage.update_metadata(key, new_metadata)
        assert success is True
        
        # Check that old tags are removed and new ones are added
        assert "old_tag" not in storage.tags or key not in storage.tags["old_tag"]
        assert key in storage.tags["new_tag1"]
        assert key in storage.tags["new_tag2"]

    def test_update_metadata_video_not_found(self, storage):
        """Test metadata update for non-existent video."""
        success = storage.update_metadata("non_existent_key", {"title": "New Title"})
        assert success is False

    def test_delete_video_success(self, storage, sample_video_data, sample_metadata, sample_tags):
        """Test successful video deletion."""
        key = storage.store_video(sample_video_data, sample_metadata, sample_tags)
        
        success = storage.delete_video(key)
        assert success is True
        
        # Check that video is removed from all storage structures
        assert key not in storage.videos
        assert key not in storage.metadata
        assert key not in storage.all_keys
        assert key not in storage.expiry_times
        
        # Check that tags are removed
        for tag in sample_tags:
            assert key not in storage.tags[tag]

    def test_delete_video_not_found(self, storage):
        """Test video deletion for non-existent key."""
        success = storage.delete_video("non_existent_key")
        assert success is False

    def test_get_storage_stats(self, storage, sample_video_data, sample_metadata):
        """Test storage statistics retrieval."""
        # Store a video
        storage.store_video(sample_video_data, sample_metadata, tags=["tag1", "tag2"])
        
        stats = storage.get_storage_stats()
        
        assert "total_videos" in stats
        assert "total_memory_mb" in stats
        assert "max_memory_mb" in stats
        assert "memory_usage_percent" in stats
        assert "tag_distribution" in stats
        assert "expired_videos" in stats
        assert "ttl_hours" in stats
        assert "operation_count" in stats
        
        assert stats["total_videos"] == 1
        assert stats["tag_distribution"]["tag1"] == 1
        assert stats["tag_distribution"]["tag2"] == 1

    def test_clear_all(self, storage, sample_video_data, sample_metadata):
        """Test clearing all stored data."""
        # Store multiple videos
        storage.store_video(sample_video_data, sample_metadata)
        storage.store_video(sample_video_data, sample_metadata)
        
        success = storage.clear_all()
        assert success is True
        
        # Check that all storage structures are empty
        assert len(storage.videos) == 0
        assert len(storage.metadata) == 0
        assert len(storage.tags) == 0
        assert len(storage.all_keys) == 0
        assert len(storage.expiry_times) == 0
        assert storage.operation_count == 0

    def test_memory_limit_check(self, storage, sample_video_data, sample_metadata):
        """Test memory limit checking and cleanup."""
        # Set very low memory limit
        storage.max_memory_mb = 0.001  # 1KB
        
        # Store a video that exceeds the limit
        large_data = os.urandom(2 * 1024)  # 2KB
        
        with patch.object(storage, '_cleanup_expired') as mock_cleanup:
            storage.store_video(large_data, sample_metadata)
            mock_cleanup.assert_called()

    def test_cleanup_expired(self, storage, sample_video_data, sample_metadata):
        """Test cleanup of expired videos."""
        # Store video with very short TTL
        key = storage.store_video(
            sample_video_data, 
            sample_metadata, 
            ttl_hours=0.001
        )
        
        # Manually set expiry time to past
        storage.expiry_times[key] = datetime.now() - timedelta(hours=1)
        
        # Trigger cleanup
        storage._cleanup_expired()
        
        # Video should be removed
        assert key not in storage.videos

    def test_len_operator(self, storage, sample_video_data, sample_metadata):
        """Test len() operator."""
        assert len(storage) == 0
        
        storage.store_video(sample_video_data, sample_metadata)
        assert len(storage) == 1
        
        storage.store_video(sample_video_data, sample_metadata)
        assert len(storage) == 2

    def test_contains_operator(self, storage, sample_video_data, sample_metadata):
        """Test 'in' operator."""
        key = storage.store_video(sample_video_data, sample_metadata)
        
        assert key in storage
        assert "non_existent_key" not in storage

    def test_operation_count_increment(self, storage, sample_video_data, sample_metadata):
        """Test that operation count increments on storage."""
        initial_count = storage.operation_count
        
        storage.store_video(sample_video_data, sample_metadata)
        
        assert storage.operation_count == initial_count + 1

    def test_cleanup_interval_trigger(self, storage, sample_video_data, sample_metadata):
        """Test that cleanup is triggered at cleanup interval."""
        storage.cleanup_interval = 2
        
        with patch.object(storage, '_check_memory_limit') as mock_check:
            # First storage operation
            storage.store_video(sample_video_data, sample_metadata)
            mock_check.assert_not_called()
            
            # Second storage operation should trigger cleanup
            storage.store_video(sample_video_data, sample_metadata)
            mock_check.assert_called_once()

    def test_error_handling_in_store_video(self, storage, sample_metadata):
        """Test error handling in store_video method."""
        # Mock _validate_video_data to raise an exception
        with patch.object(storage, '_validate_video_data', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                storage.store_video(b"test_data", sample_metadata)

    def test_error_handling_in_retrieve_video(self, storage):
        """Test error handling in retrieve_video method."""
        # Test with invalid key type - should handle gracefully
        result = storage.retrieve_video(None)
        assert result is None

    def test_error_handling_in_search_by_tags(self, storage):
        """Test error handling in search_by_tags method."""
        # Test with invalid tags - should handle gracefully
        results = storage.search_by_tags(None)
        assert results == []

    def test_error_handling_in_search_by_timestamp_range(self, storage):
        """Test error handling in timestamp range search."""
        # Test with invalid timestamp - should handle gracefully
        results = storage.search_by_timestamp_range(start_time="invalid_time")
        assert results == []

    def test_error_handling_in_update_metadata(self, storage):
        """Test error handling in metadata update."""
        # Test with invalid metadata - should handle gracefully
        success = storage.update_metadata("test_key", None)
        assert success is False

    def test_error_handling_in_delete_video(self, storage):
        """Test error handling in video deletion."""
        # Test with invalid key - should handle gracefully
        success = storage.delete_video(None)
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__])
