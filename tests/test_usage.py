from datetime import datetime

from bson import ObjectId

from tirzah.sessions.usage import record_node_usage


class FakeUpdateResult:
    def __init__(self, modified_count):
        self.modified_count = modified_count


class FakeCollection:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def update_many(self, filter_query, update):
        modified_count = 0
        for row in self.rows:
            if not matches(row, filter_query):
                continue
            for field, value in update.get("$inc", {}).items():
                row[field] = row.get(field, 0) + value
            row.update(update.get("$set", {}))
            modified_count += 1
        return FakeUpdateResult(modified_count)


class FakeDb:
    def __init__(self, rows=None):
        self.nodes = FakeCollection(rows)


class DbWithoutNodes:
    pass


def test_record_node_usage_increments_only_valid_non_rejected_nodes() -> None:
    used_id = ObjectId()
    missing_score_id = ObjectId()
    rejected_id = ObjectId()
    db = FakeDb(
        [
            {
                "_id": used_id,
                "endorsement_label": "unreviewed",
                "usage_score": 2,
            },
            {
                "_id": missing_score_id,
                "endorsement_label": "explicit_endorsed",
            },
            {
                "_id": rejected_id,
                "endorsement_label": "rejected",
                "usage_score": 9,
            },
        ]
    )

    modified = record_node_usage(
        db,
        [str(used_id), str(used_id), "not-an-object-id", str(missing_score_id), str(rejected_id)],
    )

    assert modified == 2
    assert db.nodes.rows[0]["usage_score"] == 3
    assert db.nodes.rows[1]["usage_score"] == 1
    assert db.nodes.rows[2]["usage_score"] == 9
    assert isinstance(db.nodes.rows[0]["last_used_at"], datetime)
    assert isinstance(db.nodes.rows[1]["last_used_at"], datetime)
    assert "last_used_at" not in db.nodes.rows[2]


def test_record_node_usage_skips_missing_collection_or_invalid_ids() -> None:
    assert record_node_usage(DbWithoutNodes(), [str(ObjectId())]) == 0
    assert record_node_usage(FakeDb(), ["bad"]) == 0


def matches(row, query):
    for key, expected in query.items():
        actual = row.get(key)
        if isinstance(expected, dict) and "$in" in expected:
            if actual not in expected["$in"]:
                return False
        elif isinstance(expected, dict) and "$ne" in expected:
            if actual == expected["$ne"]:
                return False
        elif isinstance(expected, dict) and "$nin" in expected:
            if actual in expected["$nin"]:
                return False
        elif actual != expected:
            return False
    return True
