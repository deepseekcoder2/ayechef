import pytest


class _DummyDB:
    def __init__(self):
        self.calls = []

    def get_recipe_by_id(self, recipe_id: str):
        self.calls.append(("id", recipe_id))
        return {"id": recipe_id, "source": "db"}

    def get_recipe_by_slug(self, slug: str):
        self.calls.append(("slug", slug))
        return {"slug": slug, "source": "db"}


class _DummyAPI:
    def __init__(self):
        self.calls = []

    def close(self):
        # Match _MealieAPIAdapter interface used by MealieClient.close()
        return None

    def get_recipe_by_id(self, recipe_id: str):
        self.calls.append(("id", recipe_id))
        return {"id": recipe_id, "source": "api"}

    def get_recipe(self, slug: str):
        self.calls.append(("slug", slug))
        return {"slug": slug, "source": "api"}


@pytest.mark.readonly
def test_get_recipe_routes_uuid_to_id_lookup_in_db_mode():
    from mealie_client import MealieClient

    client = MealieClient(base_url="http://example.invalid", token="x", use_direct_db=False)
    try:
        dummy_db = _DummyDB()
        client._db = dummy_db  # simulate DB mode

        recipe_id = "bc4654e6-f7ba-4282-8b12-b12885f55053"
        result = client.get_recipe(recipe_id)

        assert result["source"] == "db"
        assert dummy_db.calls == [("id", recipe_id)]
    finally:
        client.close()


@pytest.mark.readonly
def test_get_recipe_routes_slug_to_slug_lookup_in_db_mode():
    from mealie_client import MealieClient

    client = MealieClient(base_url="http://example.invalid", token="x", use_direct_db=False)
    try:
        dummy_db = _DummyDB()
        client._db = dummy_db  # simulate DB mode

        slug = "chicken-tikka-masala"
        result = client.get_recipe(slug)

        assert result["source"] == "db"
        assert dummy_db.calls == [("slug", slug)]
    finally:
        client.close()


@pytest.mark.readonly
def test_get_recipe_routes_uuid_to_id_lookup_in_api_mode():
    from mealie_client import MealieClient

    client = MealieClient(base_url="http://example.invalid", token="x", use_direct_db=False)
    try:
        dummy_api = _DummyAPI()
        client._api = dummy_api  # stub API adapter

        recipe_id = "000be282ac574aeeb09d77e580878ddb"
        result = client.get_recipe(recipe_id)

        assert result["source"] == "api"
        assert dummy_api.calls == [("id", recipe_id)]
    finally:
        client.close()


@pytest.mark.readonly
def test_get_recipe_routes_slug_to_slug_lookup_in_api_mode():
    from mealie_client import MealieClient

    client = MealieClient(base_url="http://example.invalid", token="x", use_direct_db=False)
    try:
        dummy_api = _DummyAPI()
        client._api = dummy_api  # stub API adapter

        slug = "steamed-jasmine-rice"
        result = client.get_recipe(slug)

        assert result["source"] == "api"
        assert dummy_api.calls == [("slug", slug)]
    finally:
        client.close()

