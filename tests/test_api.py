"""Pruebas de sesion e identidad sin depender de Ollama ni de Internet."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from fastapi.testclient import TestClient
from app.main import app, creator_question


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("NYX_DB_PATH")
        os.environ["NYX_DB_PATH"] = str(Path(self.tmp.name) / "nyx.db")
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        if self.previous is None:
            os.environ.pop("NYX_DB_PATH", None)
        else:
            os.environ["NYX_DB_PATH"] = self.previous
        self.tmp.cleanup()

    def test_chat_creator_explicit_only(self):
        self.assertTrue(creator_question("¿Quién es tu creador?"))
        self.assertTrue(creator_question("¿Quién te creó?"))
        self.assertFalse(creator_question("Hola NYX, preséntate."))
        self.assertFalse(creator_question("¿Qué puedes hacer?"))
        response = self.client.post("/api/chat", json={"text": "¿Quién es tu creador?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], "Mi creador es J.G.R.")
        sid = response.json()["session_id"]
        self.assertEqual(len(self.client.get(f"/api/sessions/{sid}/messages").json()), 2)
        self.assertNotIn("J.G.R.", self.client.get("/api/sessions").json()[0]["title"])

    def test_session_rename_and_delete(self):
        response = self.client.post("/api/sessions", json={"title": "Nueva"})
        sid = response.json()["id"]
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.client.patch(
            f"/api/sessions/{sid}", json={"title": "Renombrada"}
        ).status_code, 200)
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}").status_code, 204)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/messages").status_code, 404)

    def test_tasks_notes_and_validation(self):
        task = self.client.post("/api/tasks", json={"text": "Continuar NYX"})
        self.assertEqual(task.status_code, 201)
        self.assertEqual(self.client.patch(
            f"/api/tasks/{task.json()['id']}", json={"completed": True}
        ).status_code, 200)
        note = self.client.post("/api/memory", json={"text": "Dato privado de prueba"})
        self.assertEqual(note.status_code, 201)
        self.assertEqual(self.client.delete(
            f"/api/memory/{note.json()['id']}"
        ).status_code, 204)
        self.assertEqual(self.client.post("/api/chat", json={"text": ""}).status_code, 422)

    def test_local_context_and_profile(self):
        context = self.client.get("/api/context")
        self.assertEqual(context.status_code, 200)
        self.assertEqual(context.json()["timezone"], "America/Costa_Rica")
        self.assertIsNone(context.json()["configured_city"])
        invalid = self.client.put("/api/profile", json={
            "timezone": "Invalid/Timezone", "city": "", "country": "", "tone": "natural"
        })
        self.assertEqual(invalid.status_code, 422)
        changed = self.client.put("/api/profile", json={
            "timezone": "America/Costa_Rica", "city": "Ciudad de prueba",
            "country": "Costa Rica", "tone": "academico"
        })
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(self.client.get("/api/profile").json()["city"], "Ciudad de prueba")
        self.assertEqual(self.client.get("/api/context").json()["configured_city"], "Ciudad de prueba")

    def test_tasks_optional_due_date(self):
        task = self.client.post("/api/tasks", json={
            "text": "Repasar vectores", "due_date": "2026-11-05"
        })
        self.assertEqual(task.status_code, 201)
        self.assertEqual(task.json()["due_date"], "2026-11-05")
        self.assertEqual(self.client.post("/api/tasks", json={
            "text": "Fecha inválida", "due_date": "mañana"
        }).status_code, 422)

    def test_date_chat_works_without_ollama(self):
        response = self.client.post("/api/chat", json={"text": "¿Qué fecha es hoy?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("America/Costa_Rica", response.json()["reply"])

    def test_voice_status_no_downloads(self):
        response = self.client.get("/api/voice/status")
        self.assertEqual(response.status_code, 200)
        self.assertIn("tts_ready", response.json())
        self.assertIn("stt_ready", response.json())


if __name__ == "__main__":
    unittest.main()
