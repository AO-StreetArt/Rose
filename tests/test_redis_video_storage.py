#!/usr/bin/env python3
"""
Unit tests for the RedisVideoStorage class.

This module tests all functionality of the Redis-based video storage system including:
- Video storage and retrieval
- Metadata management
- Tag-based search
- Timestamp-based search
- TTL and cleanup functionality
- Error handling
- Redis connection management
"""

import pytest
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import redis

from rose.storage.redis_video_storage import RedisVideoStorage


class TestRedisVideoStorage:
    """Test cases for the RedisVideoStorage class."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client for testing."""
        mock_client = Mock(spec=redis.Redis)
        mock_client.ping.return_value = True
        mock_client.set.return_value = True
        mock_client.get.return_value = b'test_video_data'
        mock_client.delete.return_value = 1
        mock_client.expire.return_value = True
        mock_client.info.return_value = {"used_memory": 1048576}  # 1MB
        return mock_client

    @pytest.fixture
    def mock_text_client(self):
        """Create a mock Redis text client for testing."""
        mock_client = Mock(spec=redis.Redis)
        mock_client.ping.return_value = True
        mock_client.set.return_value = True
        mock_client.get.return_value = '{"title": "Test Video"}'
        mock_client.delete.return_value = 1
        mock_client.expire.return_value = True
        mock_client.sadd.return_value = 1
        mock_client.srem.return_value = 1
        mock_client.smembers.return_value = {b'video_key_1', b'video_key_2'}
        mock_client.scard.return_value = 2
        mock_client.sismember.return_value = True
        mock_client.zadd.return_value = 1
        mock_client.zrangebyscore.return_value = [b'ts:video_key_1', b'ts:video_key_2']
        mock_client.zrem.return_value = 1
        return mock_client

    @pytest.fixture
    def storage(self, mock_redis_client, mock_text_client):
        """Create a RedisVideoStorage instance with mocked Redis clients."""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis_class.side_effect = [mock_redis_client, mock_text_client]
            storage = RedisVideoStorage(
                host="localhost",
                port=6379,
                db=0,
                max_memory_mb=100,
                ttl_hours=24,
                max_video_size_mb=10
            )
            return storage

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
        with patch('redis.Redis') as mock_redis_class:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis_class.return_value = mock_client
            
            storage = RedisVideoStorage()
            
            assert storage.max_memory_mb == 2048
            assert storage.ttl_hours == 48
            assert storage.max_video_size_mb == 100
            assert storage.VIDEO_PREFIX == "vid:"
            assert storage.METADATA_PREFIX == "meta:"
            assert storage.TAG_PREFIX == "tag:"

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch('redis.Redis') as mock_redis_class:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis_class.return_value = mock_client
            
            storage = RedisVideoStorage(
                host="custom_host",
                port=6380,
                db=1,
                password="password123",
                max_memory_mb=512,
                ttl_hours=12,
                max_video_size_mb=50
            )
            
            assert storage.max_memory_mb == 512
            assert storage.ttl_hours == 12
            assert storage.max_video_size_mb == 50

    def test_init_connection_failure(self):
        """Test initialization with Redis connection failure."""
        with patch('redis.Redis') as mock_redis_class:
            mock_client = Mock()
            mock_client.ping.side_effect = redis.ConnectionError("Connection failed")
            mock_redis_class.return_value = mock_client
            
            with pytest.raises(redis.ConnectionError, match="Connection failed"):
                RedisVideoStorage()

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
        
        # Check that Redis operations were called
        storage.redis_client.set.assert_called()
        storage.text_client.set.assert_called()
        storage.text_client.sadd.assert_called()
        storage.text_client.zadd.assert_called()

    def test_store_video_with_custom_key(self, storage, sample_video_data, sample_metadata):
        """Test video storage with custom key."""
        custom_key = "custom_video_key"
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            key=custom_key
        )
        
        assert key == custom_key

    def test_store_video_with_filename(self, storage, sample_video_data, sample_metadata):
        """Test video storage with filename."""
        filename = "test_video.mp4"
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            filename=filename
        )
        
        # Check that metadata was stored with filename
        storage.text_client.set.assert_called()

    def test_store_video_with_custom_ttl(self, storage, sample_video_data, sample_metadata):
        """Test video storage with custom TTL."""
        custom_ttl = 2  # 2 hours
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            ttl_hours=custom_ttl
        )
        
        # Check that TTL was set
        storage.redis_client.expire.assert_called()
        storage.text_client.expire.assert_called()

    def test_store_video_without_ttl(self, storage, sample_video_data, sample_metadata):
        """Test video storage without TTL."""
        storage.ttl_hours = 0
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata
        )
        
        # Check that no TTL was set
        storage.redis_client.expire.assert_not_called()
        storage.text_client.expire.assert_not_called()

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

    def test_store_video_redis_error(self, storage, sample_video_data, sample_metadata):
        """Test video storage with Redis error."""
        storage.redis_client.set.side_effect = redis.RedisError("Redis error")
        
        with pytest.raises(redis.RedisError, match="Redis error"):
            storage.store_video(sample_video_data, sample_metadata)

    def test_retrieve_video_success(self, storage):
        """Test successful video retrieval."""
        # Mock successful retrieval
        storage.redis_client.get.return_value = b'video_data'
        storage.text_client.get.return_value = '{"title": "Test Video"}'
        
        # Retrieve video data
        retrieved_data = storage.retrieve_video("test_key")
        assert retrieved_data == b'video_data'
        
        # Retrieve metadata
        retrieved_metadata = storage.retrieve_video("test_key", format="metadata")
        assert retrieved_metadata["title"] == "Test Video"

    def test_retrieve_video_not_found(self, storage):
        """Test video retrieval for non-existent key."""
        storage.redis_client.get.return_value = None
        storage.text_client.get.return_value = None
        
        result = storage.retrieve_video("non_existent_key")
        assert result is None

    def test_retrieve_video_redis_error(self, storage):
        """Test video retrieval with Redis error."""
        storage.redis_client.get.side_effect = redis.RedisError("Redis error")
        
        result = storage.retrieve_video("test_key")
        assert result is None

    def test_search_by_tags_or_operator(self, storage):
        """Test tag search with OR operator."""
        # Mock tag search results
        storage.text_client.smembers.side_effect = [
            {b'video_key_1', b'video_key_2'},
            {b'video_key_2', b'video_key_3'}
        ]
        storage.text_client.get.return_value = '{"title": "Test Video"}'
        
        results = storage.search_by_tags(["tag1", "tag2"])
        
        assert len(results) == 3  # OR operation should return all unique keys
        storage.text_client.smembers.assert_called()

    def test_search_by_tags_and_operator(self, storage):
        """Test tag search with AND operator."""
        # Mock tag search results
        storage.text_client.smembers.side_effect = [
            {b'video_key_1', b'video_key_2'},
            {b'video_key_2'}
        ]
        storage.text_client.get.return_value = '{"title": "Test Video"}'
        
        results = storage.search_by_tags(["tag1", "tag2"], operator="AND")
        
        assert len(results) == 1
        storage.text_client.smembers.assert_called()

    def test_search_by_tags_no_matches(self, storage):
        """Test tag search with no matches."""
        storage.text_client.smembers.return_value = set()
        
        results = storage.search_by_tags(["nonexistent_tag"])
        assert len(results) == 0

    def test_search_by_tags_redis_error(self, storage):
        """Test tag search with Redis error."""
        storage.text_client.smembers.side_effect = redis.RedisError("Redis error")
        
        results = storage.search_by_tags(["tag1"])
        assert len(results) == 0

    def test_search_by_timestamp_range(self, storage):
        """Test timestamp range search."""
        # Mock timestamp search results
        storage.text_client.zrangebyscore.return_value = [b'ts:video_key_1', b'ts:video_key_2']
        storage.text_client.get.return_value = '{"title": "Test Video"}'
        
        one_hour_ago = datetime.now() - timedelta(hours=1)
        results = storage.search_by_timestamp_range(start_time=one_hour_ago)
        
        assert len(results) == 2
        storage.text_client.zrangebyscore.assert_called()

    def test_search_by_timestamp_range_with_end_time(self, storage):
        """Test timestamp range search with end time."""
        # Mock timestamp search results
        storage.text_client.zrangebyscore.return_value = [b'ts:video_key_1']
        storage.text_client.get.return_value = '{"title": "Test Video"}'
        
        one_hour_later = datetime.now() + timedelta(hours=1)
        results = storage.search_by_timestamp_range(end_time=one_hour_later)
        
        assert len(results) == 1

    def test_search_by_timestamp_range_redis_error(self, storage):
        """Test timestamp range search with Redis error."""
        storage.text_client.zrangebyscore.side_effect = redis.RedisError("Redis error")
        
        results = storage.search_by_timestamp_range()
        assert len(results) == 0

    def test_list_all_videos(self, storage):
        """Test listing all videos."""
        # Mock list results
        storage.text_client.smembers.return_value = {b'video_key_1', b'video_key_2'}
        storage.text_client.get.return_value = '{"title": "Test Video"}'
        
        all_videos = storage.list_all_videos()
        
        assert len(all_videos) == 2
        storage.text_client.smembers.assert_called()

    def test_list_all_videos_redis_error(self, storage):
        """Test listing all videos with Redis error."""
        storage.text_client.smembers.side_effect = redis.RedisError("Redis error")
        
        all_videos = storage.list_all_videos()
        assert len(all_videos) == 0

    def test_update_metadata_success(self, storage):
        """Test successful metadata update."""
        # Mock existing metadata
        storage.text_client.get.return_value = '{"title": "Old Title", "tags": ["old_tag"]}'
        storage.text_client.set.return_value = True
        
        new_metadata = {
            "title": "Updated Title",
            "description": "Updated description"
        }
        
        success = storage.update_metadata("test_key", new_metadata)
        assert success is True
        
        # Check that metadata was updated
        storage.text_client.set.assert_called()

    def test_update_metadata_with_tags(self, storage):
        """Test metadata update with new tags."""
        # Mock existing metadata
        storage.text_client.get.return_value = '{"title": "Old Title", "tags": ["old_tag"]}'
        storage.text_client.set.return_value = True
        
        new_metadata = {
            "tags": ["new_tag1", "new_tag2"]
        }
        
        success = storage.update_metadata("test_key", new_metadata)
        assert success is True
        
        # Check that tags were updated
        storage.text_client.srem.assert_called()
        storage.text_client.sadd.assert_called()

    def test_update_metadata_video_not_found(self, storage):
        """Test metadata update for non-existent video."""
        storage.text_client.get.return_value = None
        
        success = storage.update_metadata("non_existent_key", {"title": "New Title"})
        assert success is False

    def test_update_metadata_redis_error(self, storage):
        """Test metadata update with Redis error."""
        storage.text_client.get.side_effect = redis.RedisError("Redis error")
        
        success = storage.update_metadata("test_key", {"title": "New Title"})
        assert success is False

    def test_delete_video_success(self, storage):
        """Test successful video deletion."""
        # Mock existing metadata
        storage.text_client.get.return_value = '{"tags": ["tag1", "tag2"]}'
        storage.redis_client.delete.return_value = 1
        storage.text_client.delete.return_value = 1
        storage.text_client.srem.return_value = 1
        storage.text_client.zrem.return_value = 1
        
        success = storage.delete_video("test_key")
        assert success is True
        
        # Check that all Redis operations were called
        storage.redis_client.delete.assert_called()
        storage.text_client.delete.assert_called()
        storage.text_client.srem.assert_called()
        storage.text_client.zrem.assert_called()

    def test_delete_video_not_found(self, storage):
        """Test video deletion for non-existent key."""
        storage.text_client.get.return_value = None
        
        success = storage.delete_video("non_existent_key")
        assert success is False

    def test_delete_video_redis_error(self, storage):
        """Test video deletion with Redis error."""
        storage.text_client.get.side_effect = redis.RedisError("Redis error")
        
        success = storage.delete_video("test_key")
        assert success is False

    def test_get_storage_stats(self, storage):
        """Test storage statistics retrieval."""
        # Mock Redis info
        storage.text_client.scard.return_value = 2
        storage.redis_client.info.return_value = {"used_memory": 1048576}  # 1MB
        storage.text_client.smembers.return_value = {b'video_key_1', b'video_key_2'}
        storage.text_client.get.return_value = '{"tags": ["tag1", "tag2"]}'
        
        # Mock connection pool
        mock_pool = Mock()
        mock_pool.connection_kwargs = {"db": 0}
        storage.redis_client.connection_pool = mock_pool
        
        stats = storage.get_storage_stats()
        
        assert "total_videos" in stats
        assert "total_memory_mb" in stats
        assert "max_memory_mb" in stats
        assert "memory_usage_percent" in stats
        assert "tag_distribution" in stats
        assert "ttl_hours" in stats
        assert "redis_db" in stats
        
        assert stats["total_videos"] == 2

    def test_get_storage_stats_redis_error(self, storage):
        """Test storage statistics retrieval with Redis error."""
        storage.text_client.scard.side_effect = redis.RedisError("Redis error")
        
        stats = storage.get_storage_stats()
        assert stats == {}

    def test_clear_all(self, storage):
        """Test clearing all stored data."""
        # Mock list of keys
        storage.text_client.smembers.return_value = {b'video_key_1', b'video_key_2'}
        storage.text_client.get.return_value = '{"tags": ["tag1"]}'
        storage.redis_client.delete.return_value = 1
        storage.text_client.delete.return_value = 1
        storage.text_client.srem.return_value = 1
        storage.text_client.zrem.return_value = 1
        
        success = storage.clear_all()
        assert success is True
        
        # Check that global indices were cleared
        storage.text_client.delete.assert_called()

    def test_clear_all_redis_error(self, storage):
        """Test clearing all data with Redis error."""
        storage.text_client.smembers.side_effect = redis.RedisError("Redis error")
        
        success = storage.clear_all()
        assert success is False

    def test_len_operator(self, storage):
        """Test len() operator."""
        storage.text_client.scard.return_value = 5
        
        assert len(storage) == 5

    def test_len_operator_redis_error(self, storage):
        """Test len() operator with Redis error."""
        storage.text_client.scard.side_effect = redis.RedisError("Redis error")
        
        assert len(storage) == 0

    def test_contains_operator(self, storage):
        """Test 'in' operator."""
        storage.text_client.sismember.return_value = True
        
        assert "test_key" in storage

    def test_contains_operator_redis_error(self, storage):
        """Test 'in' operator with Redis error."""
        storage.text_client.sismember.side_effect = redis.RedisError("Redis error")
        
        assert "test_key" not in storage

    def test_error_handling_in_store_video(self, storage, sample_metadata):
        """Test error handling in store_video method."""
        # Mock _validate_video_data to raise an exception
        with patch.object(storage, '_validate_video_data', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                storage.store_video(b"test_data", sample_metadata)

    def test_error_handling_in_retrieve_video(self, storage):
        """Test error handling in retrieve_video method."""
        # Test with invalid key type - should handle gracefully
        # Reset mock to return None for None key
        storage.redis_client.get.return_value = None
        storage.text_client.get.return_value = None
        
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
        # Reset mock to return None for None key
        storage.text_client.get.return_value = None
        
        success = storage.delete_video(None)
        assert success is False

    def test_redis_connection_test(self, storage):
        """Test Redis connection testing."""
        # Test successful connection
        storage.redis_client.ping.return_value = True
        assert storage.redis_client.ping() is True
        
        # Test failed connection
        storage.redis_client.ping.side_effect = redis.ConnectionError("Connection failed")
        with pytest.raises(redis.ConnectionError):
            storage.redis_client.ping()

    def test_memory_usage_calculation(self, storage):
        """Test memory usage calculation in storage stats."""
        # Mock Redis info with specific memory usage
        storage.text_client.scard.return_value = 1
        storage.redis_client.info.return_value = {"used_memory": 2097152}  # 2MB
        storage.text_client.smembers.return_value = {b'video_key_1'}
        storage.text_client.get.return_value = '{"tags": ["tag1"]}'
        
        # Mock connection pool
        mock_pool = Mock()
        mock_pool.connection_kwargs = {"db": 0}
        storage.redis_client.connection_pool = mock_pool
        
        stats = storage.get_storage_stats()
        
        assert stats["total_memory_mb"] == 2.0
        assert stats["memory_usage_percent"] == 2.0  # 2MB / 100MB * 100

    def test_ttl_setting(self, storage, sample_video_data, sample_metadata):
        """Test TTL setting for stored videos."""
        # Test with custom TTL
        custom_ttl = 5  # 5 hours
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            ttl_hours=custom_ttl
        )
        
        # Check that TTL was set with correct duration
        expected_ttl_seconds = custom_ttl * 3600
        storage.redis_client.expire.assert_called()
        storage.text_client.expire.assert_called()

    def test_tag_indexing(self, storage, sample_video_data, sample_metadata, sample_tags):
        """Test tag indexing functionality."""
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata,
            tags=sample_tags
        )
        
        # Check that tags were indexed
        for tag in sample_tags:
            storage.text_client.sadd.assert_called()

    def test_timestamp_indexing(self, storage, sample_video_data, sample_metadata):
        """Test timestamp indexing functionality."""
        key = storage.store_video(
            video_data=sample_video_data,
            metadata=sample_metadata
        )
        
        # Check that timestamp was indexed
        storage.text_client.zadd.assert_called()

    def test_metadata_json_handling(self, storage):
        """Test JSON handling for metadata."""
        # Test metadata retrieval with valid JSON
        storage.text_client.get.return_value = '{"title": "Test", "tags": ["tag1"]}'
        
        metadata = storage.retrieve_video("test_key", format="metadata")
        assert metadata["title"] == "Test"
        assert metadata["tags"] == ["tag1"]
        
        # Test metadata retrieval with invalid JSON - should handle gracefully
        storage.text_client.get.return_value = 'invalid json'
        
        result = storage.retrieve_video("test_key", format="metadata")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
