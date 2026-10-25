"""Pruebas del núcleo cognitivo sin descargar modelos ni llamar a Internet."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import storage, cognition
from app.math_engine import matrix_operation


class CognitiveTests(unittest.TestCase):
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

    def test_profile_timezone_and_unconfigured_location(self):
        context = cognition.current_context()
        self.assertEqual(context["timezone"], "America/Costa_Rica")
        self.assertIsNone(context["configured_city"])
        self.assertFalse(context["location_is_current"])
        self.assertIn("date", context)
        storage.update_profile(timezone="America/Costa_Rica", city="Lugar de prueba",
                               country="Costa Rica", tone="academico")
        self.assertEqual(cognition.current_context()["configured_city"], "Lugar de prueba")

    def test_date_does_not_need_model(self):
        session = storage.create_session("Hora")
        storage.add_message("user", "¿Qué día es hoy?", session["id"])
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock) as model:
            reply, events = asyncio.run(cognition.generate_reply(
                "¿Qué día es hoy?", session["id"], "http://ollama", "nyx:0.3"
            ))
            model.assert_not_awaited()
        self.assertIn("America/Costa_Rica", reply)
        self.assertEqual(events, [])

    def test_direct_task_and_memory_without_model(self):
        session = storage.create_session("Acciones")
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock) as model:
            task_reply, task_events = asyncio.run(cognition.generate_reply(
                "Crea una tarea para repasar matrices mañana",
                session["id"], "http://ollama", "nyx:0.3"
            ))
            memory_reply, note_events = asyncio.run(cognition.generate_reply(
                "Recuerda que prefiero explicaciones breves",
                session["id"], "http://ollama", "nyx:0.3"
            ))
            model.assert_not_awaited()
        self.assertEqual(task_events, ["create_task"])
        self.assertIn("registré", task_reply)
        self.assertEqual(len(storage.list_tasks()), 1)
        self.assertIsNotNone(storage.list_tasks()[0]["due_date"])
        self.assertEqual(note_events, ["remember_note"])
        self.assertIn("memoria local", memory_reply)
        self.assertEqual(storage.list_memories()[0]["content"], "prefiero explicaciones breves")

    def test_location_is_not_inferred(self):
        context = cognition.current_context()
        self.assertIn("No tengo acceso", cognition._direct_reply("¿Dónde estoy?", context))
        self.assertIn("no tengo configurada", cognition._direct_reply("¿De dónde soy?", context).lower())

    def test_tools_read_database(self):
        task = storage.add_task("Estudiar álgebra", "2026-11-05")
        note = storage.add_memory("Prefiero respuestas precisas")
        session = storage.create_session("Matrices")
        storage.add_message("user", "Producto vectorial", session["id"])
        self.assertEqual(cognition.execute_tool("list_tasks", {}, "mis tareas")["tasks"][0]["id"], task["id"])
        self.assertEqual(cognition.execute_tool("list_memories", {}, "mis notas")["memories"][0]["id"], note["id"])
        self.assertEqual(cognition.execute_tool("search_history", {"query": "vectorial"},
                                                "busca en mi historia")["sessions"][0]["id"], session["id"])

    def test_actions_require_explicit_user_request(self):
        rejected = cognition.execute_tool("create_task", {"title": "Enviar correo"}, "Hola NYX")
        self.assertIn("error", rejected)
        self.assertEqual(storage.list_tasks(), [])
        accepted = cognition.execute_tool("create_task", {
            "title": "Estudiar álgebra", "due_date": "2026-11-05"
        }, "Crea una tarea para estudiar álgebra")
        self.assertEqual(accepted["created_task"]["due_date"], "2026-11-05")
        self.assertIn("error", cognition.execute_tool("complete_task", {
            "task_id": 1
        }, "Dime qué tareas tengo"))
        self.assertIn("error", cognition.execute_tool("remember_note", {
            "content": "Vivo en lugar de prueba"
        }, "Hola NYX"))
        saved = cognition.execute_tool("remember_note", {
            "content": "Prefiero español"
        }, "Recuerda que prefiero español")
        self.assertEqual(saved["saved_memory"]["content"], "Prefiero español")

    def test_cannot_delete_or_execute_code_from_model(self):
        self.assertIn("error", cognition.execute_tool("delete_all", {}, "Borra todo"))
        self.assertIn("error", cognition.execute_tool("matrix_calculate", {
            "operation": "determinant", "matrix_a": [["__import__('os').system('echo BAD')"]]
        }, "Calcula el determinante"))

    def test_math_concepts(self):
        determinant = matrix_operation("determinant", [[2, 1], [1, -3]])
        self.assertEqual(determinant["value"], "-7")
        inverse = matrix_operation("inverse", [[2, 1], [1, -3]])
        self.assertEqual(inverse["value"], [["3/7", "1/7"], ["1/7", "-2/7"]])
        transpose = matrix_operation("transpose", [[1, 2], [3, 4]])
        self.assertEqual(transpose["value"], [["1", "3"], ["2", "4"]])
        solved = matrix_operation("solve", [[2, 1], [1, -3]], vector_b=[3, 4])
        self.assertEqual(solved["value"], [["13/7"], ["-5/7"]])
        cross = matrix_operation("cross", [[1, 0, 0]], [[0, 1, 0]])
        self.assertEqual(cross["value"], [["0"], ["0"], ["1"]])
        dot = matrix_operation("dot", [[2, 3]], [[4, 5]])
        self.assertEqual(dot["value"], "23")

    def test_math_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            matrix_operation("inverse", [[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            matrix_operation("multiply", [[1, 2]], [[2, 3]])
        with self.assertRaises(ValueError):
            matrix_operation("rank", [[1] * 9])
        with self.assertRaises(ValueError):
            matrix_operation("dot", [[True, 2]], [[1, 2]])

    def test_ollama_tool_call_roundtrip(self):
        session = storage.create_session("Tareas")
        storage.add_message("user", "¿Podrías crear una tarea para estudiar matrices?", session["id"])
        outputs = [
            {"role": "assistant", "content": "", "tool_calls": [{
                "function": {"name": "create_task", "arguments": {
                    "title": "Estudiar matrices", "due_date": ""
                }}
            }]},
            {"role": "assistant", "content": "He registrado la tarea Estudiar matrices."}
        ]
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock,
                          side_effect=outputs) as model:
            reply, events = asyncio.run(cognition.generate_reply(
                "¿Podrías crear una tarea para estudiar matrices?", session["id"],
                "http://ollama", "nyx:0.3"
            ))
            self.assertEqual(model.await_count, 2)
        self.assertIn("registrado", reply)
        self.assertEqual(events, ["create_task"])
        self.assertEqual(storage.list_tasks()[0]["title"], "Estudiar matrices")

    def test_profile_data_is_not_creator_identity(self):
        profile = storage.get_profile()
        self.assertNotIn("creator", profile)
        self.assertNotIn("J.G.R.", str(profile))


if __name__ == "__main__":
    unittest.main()
