"""Regression: recalled experiences and user feedback reach Qwen as bounded, labeled data."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from app import brain,cognition,security,storage


class BrainCognitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.old=os.environ.get("NYX_DB_PATH")
        self.old_model=os.environ.get("NYX_BRAIN_EMBED_MODEL")
        os.environ["NYX_DB_PATH"]=str(Path(self.tmp.name)/"brain_model.db")
        os.environ.pop("NYX_BRAIN_EMBED_MODEL",None)
        security.init_auth_db()
        brain.init_db()
        security.bootstrap_admin("cognitive_admin","InitialPassword123456")
        with storage.connect() as db:
            self.admin=db.execute("SELECT id FROM users WHERE role='admin'").fetchone()[0]
        self.standard=security.create_standard("cognitive_user","AnotherPassword123")["id"]

    def tearDown(self):
        if self.old is None:
            os.environ.pop("NYX_DB_PATH",None)
        else:
            os.environ["NYX_DB_PATH"]=self.old
        if self.old_model is None:
            os.environ.pop("NYX_BRAIN_EMBED_MODEL",None)
        else:
            os.environ["NYX_BRAIN_EMBED_MODEL"]=self.old_model
        self.tmp.cleanup()

    def test_prior_case_helps_another_session_without_exposing_other_user(self):
        brain.create_case(self.admin,kind="experience",
            title="Interrupción en enlace Wi-Fi del laboratorio",
            situation="Antena wifi sin red por una fuente PoE fallando",
            action_taken="Comprobar tensión en el inyector PoE",
            result="Se normalizó la conectividad",outcome="success",
            evidence="Pérdida de paquetes pasó de 80% a 0%")
        brain.create_case(self.standard,kind="experience",
            title="Antena Wi-Fi de otra persona",
            situation="Ajustes privados del otro usuario",
            action_taken="Revisar su equipo",outcome="success")
        session=storage.create_session("Consulta técnica",self.admin)
        storage.add_message("user",
            "¿Qué experiencia tenemos sobre antena Wi-Fi y alimentación PoE?",
            session["id"],self.admin)
        with patch.object(cognition,"ask_ollama",new_callable=AsyncMock,
            return_value={"role":"assistant","content":"Comprobar tensión y revisar la evidencia."}) as model:
            reply,events=asyncio.run(cognition.generate_reply(
                "¿Qué experiencia tenemos sobre antena Wi-Fi y alimentación PoE?",
                session["id"],"http://ollama","nyx:0.4",self.admin))
        self.assertIn("evidencia",reply)
        self.assertEqual(events,[])
        sent=model.await_args.args[0]
        context="\n".join(str(message["content"]) for message in sent if message["role"]=="system")
        self.assertIn("inyector PoE",context)
        self.assertIn("no_verificado",context)
        self.assertNotIn("Ajustes privados del otro usuario",context)

    def test_unhelpful_feedback_not_copied_as_advice(self):
        previous=storage.create_session("Error previo",self.standard)
        storage.add_message("user","Cómo puedo reparar esta falla de red?",
                            previous["id"],self.standard)
        mistaken=storage.add_message("assistant","Apaga toda la seguridad de la red",
                                     previous["id"],self.standard)
        brain.record_feedback(self.standard,mistaken["id"],
            "unhelpful","Este consejo era inseguro y no corrigió el problema")
        next_session=storage.create_session("Nueva consulta",self.standard)
        storage.add_message("user","¿Cómo reparo la falla de red?",
                            next_session["id"],self.standard)
        with patch.object(cognition,"ask_ollama",new_callable=AsyncMock,
            return_value={"role":"assistant","content":"Hay que verificar el origen primero."}) as model:
            asyncio.run(cognition.generate_reply("¿Cómo reparo la falla de red?",
                next_session["id"],"http://ollama","nyx:0.4",self.standard))
        text="\n".join(str(entry["content"]) for entry in model.await_args.args[0])
        self.assertIn("NO fue útil",text)
        self.assertNotIn("Apaga toda la seguridad de la red",text)

    def test_recovery_failure_does_not_block_llm(self):
        s=storage.create_session("Prueba de fallback",self.standard)
        storage.add_memory("La fuente de poder usa protección",self.standard)
        with patch.object(brain,"retrieve",new_callable=AsyncMock,
                          side_effect=RuntimeError("Prueba de error")):
            with patch.object(cognition,"ask_ollama",new_callable=AsyncMock,
                return_value={"role":"assistant","content":"Sigo respondiendo."}) as model:
                reply,_=asyncio.run(cognition.generate_reply(
                    "¿Qué recuerdas de la fuente de poder?",s["id"],
                    "http://ollama","nyx:0.4",self.standard))
        self.assertEqual(reply,"Sigo respondiendo.")
        self.assertTrue(model.awaited)

    def test_evaluated_success_has_traceable_source_in_prompt(self):
        brain.create_case(self.standard,kind="procedure",
            title="Revisar fuente de poder",
            situation="Cortes eléctricos en una impresora 3D",
            action_taken="Medir voltaje bajo carga",
            result="Voltaje estable",outcome="success",
            evidence="Lecturas de multímetro durante 10 minutos")
        data=asyncio.run(brain.retrieve(
            "¿Qué procedimiento funcionó en la impresora?",
            self.standard,"http://ollama",model=""))
        text=brain.prompt_context(data)
        self.assertIn("relato_del_usuario_no_verificado",text)
        self.assertIn("caso #",text)
        self.assertNotIn("confirmado científicamente",text)


if __name__=="__main__":
    unittest.main()
