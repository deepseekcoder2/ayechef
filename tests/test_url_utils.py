"""Tests for URL utility functions."""
import pytest


class TestNormalizeUrl:
    """Tests for normalize_url function."""
    
    def test_strips_trailing_slash(self):
        from utils.url_utils import normalize_url
        assert normalize_url("https://example.com/recipe/") == "https://example.com/recipe"
    
    def test_converts_http_to_https(self):
        from utils.url_utils import normalize_url
        assert normalize_url("http://example.com/recipe") == "https://example.com/recipe"
    
    def test_removes_www_prefix(self):
        from utils.url_utils import normalize_url
        assert normalize_url("https://www.example.com/recipe") == "https://example.com/recipe"
    
    def test_lowercases_url(self):
        from utils.url_utils import normalize_url
        assert normalize_url("https://Example.COM/Recipe") == "https://example.com/recipe"
    
    def test_strips_whitespace(self):
        from utils.url_utils import normalize_url
        assert normalize_url("  https://example.com/recipe  ") == "https://example.com/recipe"
    
    def test_empty_string_returns_empty(self):
        from utils.url_utils import normalize_url
        assert normalize_url("") == ""
    
    def test_none_returns_empty(self):
        from utils.url_utils import normalize_url
        assert normalize_url(None) == ""
    
    def test_combined_normalization(self):
        from utils.url_utils import normalize_url
        url = "  HTTP://WWW.Example.COM/Recipe/  "
        assert normalize_url(url) == "https://example.com/recipe"
