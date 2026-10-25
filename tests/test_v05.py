"""Regresión NYX 0.5: calendario, recuperación y proveedores simulados."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import cognition, security, storage, network, research


class V05Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("NYX_DB_PATH")
        os.environ["NYX_DB_PATH"] = str(Path(self.tmp.name) / "nyx.db")
        security.init_auth_db()
        security.bootstrap_admin("v05_admin", "TempPass123456")
        with storage.connect() as db:
            self.admin = db.execute("SELECT id FROM users WHERE role='admin'").fetchone()[0]
        self.other = security.create_standard("v05_standard", "TempPass123456")["id"]

    def tearDown(self):
        if self.old is None:
            os.environ.pop("NYX_DB_PATH", None)
        else:
            os.environ["NYX_DB_PATH"] = self.old
        self.tmp.cleanup()

    def test_natural_task_name_question_without_llm(self):
        storage.add_task("Llevar moto al taller", self.admin)
        storage.add_task("Informe de red", self.admin)
        storage.add_task("Otra persona", self.other)
        session = storage.create_session("Tareas", self.admin)
        with patch.object(cognition, "ask_ollama", new_callable=AsyncMock) as model:
            answer, _ = asyncio.run(cognition.generate_reply(
                "¿Cómo se llama cada tarea?", session["id"], "http://ollama",
                "nyx:0.4", self.admin))
            model.assert_not_awaited()
        self.assertIn("Informe de red", answer)
        self.assertIn("Llevar moto", answer)
        self.assertNotIn("Otra persona", answer)

    def test_weather_does_not_force_home_country(self):
        self.assertEqual(network.extract_weather_location(
            "¿Cuál es el clima actual en París, Francia?", self.admin),
            "París, Francia")
        self.assertEqual(network.extract_weather_location(
            "¿Qué tiempo hace en Madrid?", self.admin), "Madrid")

    def test_fx_pair_and_response(self):
        self.assertEqual(research.extract_currency_request(
            "¿Cuál es el tipo de cambio actual?")[:2], ("USD","CRC"))
        self.assertEqual(research.extract_currency_request(
            "tipo de cambio de EUR a USD")[:2], ("EUR","USD"))
        with patch.object(research, "exchange_rate", new_callable=AsyncMock,
            return_value={"base":"USD","quote":"CRC","rate":510.2,
                "date":"2026-10-25","source":"Frankfurter"}) as remote:
            session = storage.create_session("Divisas", self.admin)
            answer, _ = asyncio.run(cognition.generate_reply(
                "¿Cuál es el tipo de cambio actual?", session["id"],
                "http://ollama","nyx:0.4",self.admin))
            remote.assert_awaited_once_with("USD","CRC")
        self.assertIn("510.2000", answer)
        storage.set_online_enabled(self.admin, False)
        with patch.object(research, "exchange_rate", new_callable=AsyncMock) as remote:
            answer, _ = asyncio.run(cognition.generate_reply(
                "¿Cuál es el tipo de cambio actual?", session["id"],
                "http://ollama","nyx:0.4",self.admin))
            remote.assert_not_awaited()
        self.assertIn("modo local", answer)

    def test_explicit_web_search_only(self):
        self.assertIsNone(research.extract_web_query("Hola, NYX"))
        self.assertEqual(research.extract_web_query(
            "investiga en internet sobre redes VLAN"), "redes VLAN")
        with patch.object(research, "web_search", new_callable=AsyncMock,
            return_value={"query":"redes VLAN","provider":"Bing",
                "results":[{"title":"Redes VLAN","url":"https://example.com/vlan",
                            "excerpt":"Separación lógica de redes."}]}) as remote:
            session = storage.create_session("Internet",self.admin)
            answer, _ = asyncio.run(cognition.generate_reply(
                "investiga en internet sobre redes VLAN",session["id"],
                "http://ollama","nyx:0.4",self.admin))
            remote.assert_awaited_once()
        self.assertIn("Separación lógica",answer)

    def test_calendar_crud_conflicts_and_isolation(self):
        data={"title":"Reunión técnica","description":"Revisar Wi-Fi",
              "kind":"meeting","start_at":"2026-11-05T13:00",
              "end_at":"2026-11-05T14:00","reminder_minutes":30}
        event=storage.add_calendar_event(self.admin,**data)
        self.assertEqual(storage.list_calendar_events(self.other), [])
        with self.assertRaises(ValueError):
            storage.add_calendar_event(self.admin,**{**data,"title":"Solapada"})
        self.assertIsNone(storage.update_calendar_event(event["id"],self.other,title="No"))
        changed=storage.update_calendar_event(event["id"],self.admin,title="Revisar cobertura")
        self.assertEqual(changed["title"],"Revisar cobertura")
        self.assertFalse(storage.delete_calendar_event(event["id"],self.other))
        self.assertTrue(storage.delete_calendar_event(event["id"],self.admin))

    def test_calendar_reminder_acknowledgement(self):
        now=datetime.now(ZoneInfo("America/Costa_Rica")).replace(second=0,microsecond=0)
        start=now+timedelta(minutes=5)
        event=storage.add_calendar_event(self.admin,title="Revisión",
            start_at=start.strftime("%Y-%m-%dT%H:%M"),
            end_at=(start+timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M"),
            kind="reminder",reminder_minutes=10)
        self.assertEqual(len(storage.due_calendar_reminders(self.admin)),1)
        self.assertEqual(storage.due_calendar_reminders(self.other),[])
        self.assertFalse(storage.acknowledge_calendar_reminder(event["id"],self.other))
        self.assertTrue(storage.acknowledge_calendar_reminder(event["id"],self.admin))
        self.assertEqual(storage.due_calendar_reminders(self.admin),[])

    def test_keyword_memory_is_private(self):
        storage.add_memory("La impresora Neptune utiliza una cama PEI", self.admin)
        storage.add_memory("La red usa Fortinet", self.admin)
        storage.add_memory("Impresora privada ajena", self.other)
        found=storage.search_memories("¿Qué sabes de la impresora?", self.admin)
        self.assertEqual(len(found),1)
        self.assertIn("Neptune",found[0]["content"])


if __name__ == "__main__":
    unittest.main()
