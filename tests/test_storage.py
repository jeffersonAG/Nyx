"""Pruebas de migración 0.1/0.3, propiedad y aislamiento SQLite."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import security, storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("NYX_DB_PATH")
        os.environ["NYX_DB_PATH"] = str(Path(self.tmp.name) / "nyx.db")
        security.init_auth_db()
        security.bootstrap_admin("admin_example", "temporary77")
        with storage.connect() as db:
            self.admin = db.execute("SELECT id FROM users WHERE role='admin'").fetchone()[0]
        self.standard = security.create_standard("user_example", "TestPassword123")["id"]

    def tearDown(self):
        if self.old is None:
            os.environ.pop("NYX_DB_PATH", None)
        else:
            os.environ["NYX_DB_PATH"] = self.old
        self.tmp.cleanup()

    def test_sessions_separate_and_safe(self):
        a = storage.create_session("Python", self.admin)
        b = storage.create_session("Electrónica", self.standard)
        storage.add_message("user", "Una matriz", a["id"], self.admin)
        storage.add_message("user", "Un circuito", b["id"], self.standard)
        self.assertEqual(storage.list_messages(a["id"], 200, self.standard), [])
        self.assertIsNone(storage.get_session(a["id"], self.standard))
        self.assertEqual(storage.list_sessions("matriz", self.admin)[0]["id"], a["id"])
        self.assertEqual(storage.list_sessions("", self.standard)[0]["id"], b["id"])
        self.assertFalse(storage.delete_session(a["id"], self.standard))
        self.assertIsNone(storage.rename_session(a["id"], "Intento", self.standard))
        with self.assertRaises(ValueError):
            storage.add_message("assistant", "intrusión", a["id"], self.standard)

    def test_tasks_memories_are_owned(self):
        task = storage.add_task("Estudiar", self.admin, "2026-11-05")
        note = storage.add_memory("Dato privado", self.admin)
        self.assertEqual(storage.list_tasks(self.standard), [])
        self.assertEqual(storage.list_memories(100, self.standard), [])
        self.assertIsNone(storage.update_task(task["id"], True, self.standard))
        self.assertFalse(storage.delete_task(task["id"], self.standard))
        self.assertFalse(storage.delete_memory(note["id"], self.standard))
        self.assertEqual(storage.list_tasks(self.admin)[0]["due_date"], "2026-11-05")
        self.assertEqual(storage.list_memories(100, self.admin)[0]["content"], "Dato privado")
        self.assertTrue(storage.update_task(task["id"], True, self.admin)["completed"])

    def test_profiles_have_own_city(self):
        storage.update_profile(user_id=self.admin, timezone="America/Costa_Rica",
                               city="Ciudad A", country="CR", tone="natural")
        self.assertEqual(storage.get_profile(self.admin)["city"], "Ciudad A")
        self.assertEqual(storage.get_profile(self.standard)["city"], "")
        with self.assertRaises(ValueError):
            storage.update_profile(user_id=self.standard, timezone="Invalid/TimeZone",
                                   city="", country="", tone="natural")

    def test_legacy_rows_adopted_once(self):
        with storage.connect() as db:
            db.execute("INSERT INTO sessions(title,owner_id) VALUES ('Legado',NULL)")
            db.execute("INSERT INTO tasks(title,owner_id) VALUES ('Tarea antigua',NULL)")
            db.execute("INSERT INTO memories(content,owner_id) VALUES ('Nota antigua',NULL)")
        storage.init_db()
        storage.adopt_legacy_records(self.admin)
        storage.init_db()
        storage.adopt_legacy_records(self.admin)
        self.assertTrue(any(s["title"] == "Legado" for s in storage.list_sessions("", self.admin)))
        self.assertTrue(any(t["title"] == "Tarea antigua" for t in storage.list_tasks(self.admin)))
        self.assertTrue(any(m["content"] == "Nota antigua" for m in storage.list_memories(100, self.admin)))
        self.assertFalse(any(s["title"] == "Legado" for s in storage.list_sessions("", self.standard)))

    def test_sql_injection_cannot_escape_owner(self):
        task = storage.add_task("'); DROP TABLE users; --", self.admin)
        self.assertEqual(storage.list_tasks(self.admin)[0]["id"], task["id"])
        self.assertEqual(storage.list_tasks(self.standard), [])
        self.assertEqual(security.account_count(), 2)


if __name__ == "__main__":
    unittest.main()
