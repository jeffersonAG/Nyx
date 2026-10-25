"""Núcleo de herramientas con usuarios aislados, sin Ollama durante pruebas."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import cognition, security, storage
from app.math_engine import matrix_operation


class CognitiveTests(unittest.TestCase):
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

    def test_context_uses_own_profile(self):
        storage.update_profile(user_id=self.admin, timezone="America/Costa_Rica",
                               city="Ciudad A", country="CR", tone="academico")
        self.assertEqual(cognition.current_context(self.admin)["configured_city"], "Ciudad A")
        self.assertIsNone(cognition.current_context(self.standard)["configured_city"])

    def test_date_without_model(self):
        s = storage.create_session("Fecha", self.admin)
        storage.add_message("user", "¿Qué día es hoy?", s["id"], self.admin)
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock) as model:
            reply, changes = asyncio.run(cognition.generate_reply(
                "¿Qué día es hoy?", s["id"], "http://ollama", "nyx:0.4", self.admin))
            model.assert_not_awaited()
        self.assertIn("America/Costa_Rica", reply)
        self.assertEqual(changes, [])

    def test_modules_scope_to_user(self):
        storage.add_memory("Privado", self.admin)
        storage.add_task("Comprar pieza", self.admin)
        s = storage.create_session("Cálculo", self.admin)
        storage.add_message("user", "Producto vectorial", s["id"], self.admin)
        self.assertEqual(cognition.execute_tool("list_tasks", {}, "consulta", self.standard)["tasks"], [])
        self.assertEqual(cognition.execute_tool("list_memories", {}, "consulta", self.standard)["memories"], [])
        self.assertEqual(cognition.execute_tool("search_history",
            {"query": "vectorial"}, "busca", self.standard)["sessions"], [])

    def test_direct_commands(self):
        s = storage.create_session("Acciones", self.standard)
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock) as model:
            reply, events = asyncio.run(cognition.generate_reply(
                "Crea una tarea para estudiar matrices mañana",
                s["id"], "http://ollama", "nyx:0.4", self.standard))
            note, second = asyncio.run(cognition.generate_reply(
                "Recuerda que prefiero respuestas cortas",
                s["id"], "http://ollama", "nyx:0.4", self.standard))
            model.assert_not_awaited()
        self.assertEqual(events, ["create_task"])
        self.assertEqual(second, ["remember_note"])
        self.assertIn("registré", reply)
        self.assertIn("memoria local", note)
        self.assertEqual(len(storage.list_tasks(self.standard)), 1)
        self.assertEqual(len(storage.list_memories(100, self.standard)), 1)
        self.assertEqual(storage.list_tasks(self.admin), [])

    def test_tool_calls_scoped(self):
        s = storage.create_session("Nota", self.standard)
        storage.add_message("user", "¿Podrías crear una tarea para estudiar matrices?",
                            s["id"], self.standard)
        responses = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "create_task", "arguments": {
                    "title": "Estudiar matrices", "due_date": ""
                }}
            }]},
            {"role": "assistant", "content": "He registrado tu tarea."}
        ]
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock,
                          side_effect=responses):
            reply, events = asyncio.run(cognition.generate_reply(
                "¿Podrías crear una tarea para estudiar matrices?", s["id"],
                "http://ollama", "nyx:0.4", self.standard))
        self.assertEqual(events, ["create_task"])
        self.assertIn("registrado", reply)
        self.assertEqual(storage.list_tasks(self.standard)[0]["title"], "Estudiar matrices")

    def test_module_count_without_ollama_or_permission_prompt(self):
        storage.add_task("Completar proyecto", self.standard)
        storage.add_task("Revisar laboratorio", self.standard)
        storage.add_task("Privada del administrador", self.admin)
        session = storage.create_session("Conteos", self.standard)
        storage.add_message("user", "cuantas tareas tengo pendientes ?", session["id"], self.standard)
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock) as model:
            reply, events = asyncio.run(cognition.generate_reply(
                "cuantas tareas tengo pendientes ?", session["id"],
                "http://ollama", "nyx:0.4", self.standard
            ))
            model.assert_not_awaited()
        self.assertIn("2 tareas", reply)
        self.assertNotIn("permiso", reply.lower())
        self.assertEqual(events, [])

    def test_list_tasks_direct_access(self):
        storage.add_task("Estudiar Python", self.standard)
        session = storage.create_session("Lista", self.standard)
        reply, _ = asyncio.run(cognition.generate_reply(
            "muéstrame mis tareas pendientes", session["id"],
            "http://ollama", "nyx:0.4", self.standard
        ))
        self.assertIn("Estudiar Python", reply)

    def test_online_weather_city(self):
        from app import network
        session = storage.create_session("Clima", self.standard)
        weather = {
            "city": "Ciudad Neily", "country": "Costa Rica",
            "temperature_c": 29, "feels_like_c": 33, "humidity_pct": 80,
            "precipitation_mm": 0, "wind_kmh": 5, "condition": "nublado",
            "observed_at": "12:00", "timezone": "America/Costa_Rica"
        }
        with patch.object(network, "current_weather",
                          new_callable=AsyncMock, return_value=weather) as fetcher:
            reply, events = asyncio.run(cognition.generate_reply(
                "puedes decirme el clima actual en ciudad neily",
                session["id"], "http://ollama", "nyx:0.4", self.standard
            ))
            fetcher.assert_awaited_once()
        self.assertIn("29 °C", reply)
        self.assertEqual(events, [])
        storage.set_online_enabled(self.standard, False)
        with patch.object(network, "current_weather", new_callable=AsyncMock) as fetcher:
            reply, _ = asyncio.run(cognition.generate_reply(
                "puedes decirme el clima actual en ciudad neily",
                session["id"], "http://ollama", "nyx:0.4", self.standard
            ))
            fetcher.assert_not_awaited()
        self.assertIn("modo local", reply)

    def test_public_search_explicit_only(self):
        from app import network
        self.assertEqual(network.extract_wiki_query("hola"), None)
        self.assertEqual(network.extract_wiki_query("Busca en Wikipedia sobre vectores"),
                         "vectores")
        session = storage.create_session("Investigación", self.standard)
        with patch.object(network, "search_encyclopedia", new_callable=AsyncMock,
                          return_value={"results": [{"title": "Vectores",
                              "url":"https://es.wikipedia.org/wiki/Vector"}]}) as fetcher:
            reply, _ = asyncio.run(cognition.generate_reply(
                "Busca en Wikipedia sobre vectores", session["id"],
                "http://ollama", "nyx:0.4", self.standard
            ))
            fetcher.assert_awaited_once()
        self.assertIn("https://es.wikipedia.org/wiki/Vector", reply)

    def test_math_validation(self):
        self.assertEqual(matrix_operation("determinant", [[2, 1], [1, -3]])["value"], "-7")
        self.assertEqual(matrix_operation("dot", [[2, 3]], [[4, 5]])["value"], "23")
        self.assertEqual(matrix_operation("cross", [[1, 0, 0]], [[0, 1, 0]])["value"],
                         [["0"], ["0"], ["1"]])
        self.assertIn("error", cognition.execute_tool("matrix_calculate",
                          {"operation": "rank", "matrix_a": [["__import__('os')"]]},
                          "Calcula", self.standard))


if __name__ == "__main__":
    unittest.main()
