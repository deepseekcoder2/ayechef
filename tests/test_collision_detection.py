"""
Tests for pre-import collision detection.

Run: pytest tests/test_collision_detection.py -v
"""

import pytest
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import after setting up test DB path
import utils.collision_detection as cd


class TestGetSiteDisplayName:
    """Tests for get_site_display_name()"""
    
    def test_known_site(self):
        assert cd.get_site_display_name("https://thewoksoflife.com/banana-bread/") == "The Woks of Life"
        assert cd.get_site_display_name("https://www.bbcgoodfood.com/recipes/test") == "BBC Good Food"
        assert cd.get_site_display_name("https://seriouseats.com/recipe") == "Serious Eats"
    
    def test_unknown_site(self):
        result = cd.get_site_display_name("https://my-recipe-blog.com/recipe")
        assert result == "My Recipe Blog"
    
    def test_empty_url(self):
        assert cd.get_site_display_name("") == "Unknown Source"
        assert cd.get_site_display_name(None) == "Unknown Source"
    
    def test_www_prefix_stripped(self):
        assert cd.get_site_display_name("https://www.thewoksoflife.com/test") == "The Woks of Life"


class TestCheckNameCollision:
    """Tests for check_name_collision()"""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary recipe index database."""
        db_path = tmp_path / "recipe_index.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT
            )
        """)
        conn.execute(
            "INSERT INTO recipes VALUES (?, ?, ?)",
            ("test-id-1", "Banana Bread", "banana-bread")
        )
        conn.execute(
            "INSERT INTO recipes VALUES (?, ?, ?)",
            ("test-id-2", "Chocolate Cake", "chocolate-cake")
        )
        conn.commit()
        conn.close()
        return str(db_path)
    
    def test_collision_found(self, temp_db):
        with patch.object(cd, 'RECIPE_INDEX_DB', temp_db):
            result = cd.check_name_collision("Banana Bread")
            assert result is not None
            assert result[0] == "test-id-1"
            assert result[1] == "Banana Bread"
            assert result[2] == "banana-bread"
    
    def test_collision_case_insensitive(self, temp_db):
        with patch.object(cd, 'RECIPE_INDEX_DB', temp_db):
            result = cd.check_name_collision("banana bread")
            assert result is not None
            assert result[1] == "Banana Bread"
    
    def test_no_collision(self, temp_db):
        with patch.object(cd, 'RECIPE_INDEX_DB', temp_db):
            result = cd.check_name_collision("Apple Pie")
            assert result is None


class TestGetQualifiedName:
    """Tests for get_qualified_name()"""
    
    def test_qualified_name(self):
        result = cd.get_qualified_name("Banana Bread", "https://thewoksoflife.com/banana-bread/")
        assert result == "Banana Bread (The Woks of Life)"
    
    def test_unknown_site(self):
        result = cd.get_qualified_name("Test Recipe", "https://unknown-site.com/recipe")
        assert result == "Test Recipe (Unknown Site)"


class TestShouldQualifyName:
    """Tests for should_qualify_name()"""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        db_path = tmp_path / "recipe_index.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE recipes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                slug TEXT
            )
        """)
        conn.execute(
            "INSERT INTO recipes VALUES (?, ?, ?)",
            ("test-id-1", "Banana Bread", "banana-bread")
        )
        conn.commit()
        conn.close()
        return str(db_path)
    
    def test_collision_qualifies_name(self, temp_db):
        with patch.object(cd, 'RECIPE_INDEX_DB', temp_db):
            needs_qual, final_name = cd.should_qualify_name(
                "Banana Bread",
                "https://thewoksoflife.com/banana-bread/"
            )
            assert needs_qual is True
            assert final_name == "Banana Bread (The Woks of Life)"
    
    def test_no_collision_keeps_original(self, temp_db):
        with patch.object(cd, 'RECIPE_INDEX_DB', temp_db):
            needs_qual, final_name = cd.should_qualify_name(
                "Apple Pie",
                "https://thewoksoflife.com/apple-pie/"
            )
            assert needs_qual is False
            assert final_name == "Apple Pie"
