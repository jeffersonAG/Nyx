"""Pruebas de migracion no destructiva y sesiones (sin Docker)."""
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("NYX_DB_PATH")
        os.environ["NYX_DB_PATH"] = str(Path(self.tmp.name) / "nyx.db")
        storage.init_db()

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("NYX_DB_PATH", None)
        else:
            os.environ["NYX_DB_PATH"] = self.previous
        self.tmp.cleanup()

    def test_sessions_are_separate(self):
        a = storage.create_session("Python")
        b = storage.create_session("Electronica")
        storage.add_message("user", "Hola Python", a["id"])
        storage.add_message("user", "Hola electronica", b["id"])
        self.assertEqual(len(storage.list_messages(a["id"])), 1)
        self.assertEqual(storage.list_messages(a["id"])[0]["content"], "Hola Python")
        self.assertEqual(storage.list_sessions("Python")[0]["id"], a["id"])

    def test_session_rename_delete(self):
        session = storage.create_session("Viejo")
        storage.rename_session(session["id"], "Nuevo")
        self.assertEqual(storage.get_session(session["id"])["title"], "Nuevo")
        storage.add_message("user", "No conservar", session["id"])
        self.assertTrue(storage.delete_session(session["id"]))
        self.assertEqual(storage.list_messages(session["id"]), [])

    def test_memory_tasks(self):
        task = storage.add_task("Estudiar")
        self.assertTrue(storage.update_task(task["id"], True)["completed"])
        self.assertTrue(storage.delete_task(task["id"]))
        note = storage.add_memory("Aprender FastAPI")
        self.assertEqual(storage.list_memories()[0]["id"], note["id"])
        self.assertTrue(storage.delete_memory(note["id"]))

    def test_sql_queries_parameterized(self):
        task = storage.add_task("'); DROP TABLE sessions; --")
        self.assertEqual(storage.list_tasks()[0]["id"], task["id"])
        self.assertTrue(storage.create_session("Sigue funcionando")["id"])

    def test_migrate_legacy_history_twice(self):
        # Simula la estructura exacta de messages en 0.1.
        dbfile = storage.db_path()
        with storage.connect() as db:
            db.execute("DROP TABLE messages")
            db.execute("""CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )""")
            db.execute("INSERT INTO messages(role,content) VALUES ('user','Mi chat anterior')")
            db.execute("INSERT INTO messages(role,content) VALUES ('assistant','Respuesta antigua')")
        storage.init_db()
        storage.init_db()
        legacy = storage.list_sessions("Conversaciones anteriores")
        self.assertEqual(len(legacy), 1)
        self.assertEqual(len(storage.list_messages(legacy[0]["id"])), 2)
        self.assertEqual(storage.list_messages(legacy[0]["id"])[0]["content"], "Mi chat anterior")


if __name__ == "__main__":
    unittest.main()
