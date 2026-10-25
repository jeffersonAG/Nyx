"""Brain v1 regression tests: isolation, evidence, feedback and retrieval.

No network, Ollama, remote embeddings or production SQLite are touched.
"""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from app import brain, security, storage


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.old_path=os.environ.get("NYX_DB_PATH")
        self.old_embed=os.environ.get("NYX_BRAIN_EMBED_MODEL")
        os.environ["NYX_DB_PATH"]=str(Path(self.temp.name)/"brain.db")
        os.environ.pop("NYX_BRAIN_EMBED_MODEL",None)
        brain._EMBED_UNAVAILABLE_UNTIL=0.0
        security.init_auth_db()
        brain.init_db()
        security.bootstrap_admin("brain_admin","temporary-strong-pass")
        with storage.connect() as db:
            self.admin=db.execute("SELECT id FROM users WHERE role='admin'").fetchone()[0]
        self.other=security.create_standard("brain_user","TestPassword1234")["id"]

    def tearDown(self):
        if self.old_path is None:
            os.environ.pop("NYX_DB_PATH",None)
        else:
            os.environ["NYX_DB_PATH"]=self.old_path
        if self.old_embed is None:
            os.environ.pop("NYX_BRAIN_EMBED_MODEL",None)
        else:
            os.environ["NYX_BRAIN_EMBED_MODEL"]=self.old_embed
        self.temp.cleanup()

    def _case(self,user_id=None,**overrides):
        payload={
            "kind":"experience",
            "title":"Falla de antena inalámbrica",
            "situation":"El enlace WiFi del laboratorio dejó de funcionar",
            "action_taken":"Revisar alimentación PoE y conectar nuevamente el cable",
            "result":"El enlace fue restablecido",
            "outcome":"success",
            "evidence":"Se observó el enlace operativo después de la prueba",
        }
        payload.update(overrides)
        return brain.create_case(user_id or self.admin,**payload)

    def test_schema_additive_idempotent(self):
        a=self._case()
        brain.init_db()
        self.assertEqual(brain.get_case(self.admin,a["id"])["title"],a["title"])
        self.assertEqual(brain.get_case(self.other,a["id"]),None)

    def test_scoped_crud_and_feedback_reviews(self):
        first=self._case()
        self.assertEqual(brain.list_cases(self.other),[])
        self.assertIsNone(brain.review_case(self.other,first["id"],"failure","Mal"))
        changed=brain.review_case(self.admin,first["id"],"partial",
                                   "Funcionó temporalmente, pero la falla regresó")
        self.assertEqual(changed["outcome"],"partial")
        self.assertFalse(brain.delete_case(self.other,first["id"]))
        self.assertTrue(brain.delete_case(self.admin,first["id"]))
        self.assertIsNone(brain.get_case(self.admin,first["id"]))

    def test_evidence_graph_requires_real_owned_records(self):
        left=self._case()
        right=self._case(kind="hypothesis",title="Hipótesis sobre PoE",
           situation="Pérdidas eléctricas en alimentación inalámbrica",
           action_taken="Medir corriente bajo carga",outcome="unknown")
        with self.assertRaises(ValueError):
            brain.add_relation(self.admin,from_case_id=left["id"],
                to_case_id=right["id"],relation="supports",evidence="corto")
        with self.assertRaises(ValueError):
            brain.add_relation(self.other,from_case_id=left["id"],
                to_case_id=right["id"],relation="supports",
                evidence="Medimos tensión y registramos resultados")
        link=brain.add_relation(self.admin,from_case_id=left["id"],
                to_case_id=right["id"],relation="related_to",
                evidence="Ambos registros corresponden al mismo enlace WiFi")
        self.assertEqual(link["basis"],"user_report")
        self.assertEqual(len(brain.list_relations(self.admin)),1)
        self.assertEqual(brain.list_relations(self.other),[])
        with self.assertRaises(ValueError):
            brain.add_relation(self.admin,from_case_id=left["id"],
                to_case_id=right["id"],relation="related_to",
                evidence="Ambos registros corresponden al mismo enlace WiFi")
        self.assertFalse(brain.delete_relation(self.other,link["id"]))
        self.assertTrue(brain.delete_case(self.admin,left["id"]))
        self.assertEqual(brain.list_relations(self.admin),[])

    def test_suggested_links_not_verified(self):
        first=self._case()
        second=self._case(kind="procedure",title="Problema de enlace inalámbrico",
            situation="El enlace WiFi del laboratorio está caído",
            action_taken="Verificar estado de antena",outcome="unknown")
        links=brain.propose_links(self.admin,first["id"])
        self.assertTrue(any(row["case_id"]==second["id"] for row in links))
        self.assertTrue(all(row["status"]=="propuesta_no_verificada" for row in links))

    def test_rating_targets_only_owned_assistant_message(self):
        session=storage.create_session("Experimento",self.admin)
        storage.add_message("user","¿Cómo diagnostico la fuente de poder?",session["id"],self.admin)
        answer=storage.add_message("assistant","Comprueba la alimentación y el voltaje.",
                                   session["id"],self.admin)
        with self.assertRaises(ValueError):
            brain.record_feedback(self.other,answer["id"],"helpful")
        rated=brain.record_feedback(self.admin,answer["id"],"unhelpful",
                                   "Faltó medir consumo bajo carga")
        self.assertIn("diagnostico",rated["question"])
        self.assertEqual(rated["rating"],"unhelpful")
        changed=brain.record_feedback(self.admin,answer["id"],"helpful",
                                      "Ahora sí explica cómo medir")
        self.assertEqual(changed["rating"],"helpful")
        self.assertEqual(brain.metrics(self.admin)["rated_answers"],1)
        self.assertEqual(brain.metrics(self.other)["rated_answers"],0)
        self.assertEqual(brain.list_feedback(self.admin)[0]["note"],
                         "Ahora sí explica cómo medir")

    def test_tool_execution_does_not_mean_real_world_success(self):
        session=storage.create_session("Tareas",self.admin)
        storage.add_message("user","crea tarea",session["id"],self.admin)
        message=storage.add_message("assistant","Tarea guardada",session["id"],self.admin)
        item=brain.record_tool_episode(self.admin,message["id"],"crea tarea",
            "Tarea guardada",["create_task"])
        self.assertEqual(item["outcome"],"executed")
        self.assertEqual(item["origin"],"tool_execution")
        brain.record_tool_episode(self.admin,message["id"],"crea tarea",
            "Tarea guardada",["create_task"])
        self.assertEqual(len(brain.list_cases(self.admin)),1)
        with self.assertRaises(ValueError):
            brain.review_case(self.admin,item["id"],"success")
        self.assertIsNone(brain.record_tool_episode(self.other,message["id"],
            "crea tarea","Tarea guardada",["create_task"]))

    def test_lexical_retrieval_labels_unverified_report(self):
        own=self._case()
        self._case(self.other,title="Clave privada de otra persona",
            situation="Falla de antena inalámbrica del otro usuario",
            action_taken="No divulgar información",result="Bloqueado")
        recalled=asyncio.run(brain.retrieve(
            "Problema con antena WiFi del laboratorio",self.admin,
            "http://ollama:11434",model=""))
        self.assertEqual(recalled["mode"],"lexical")
        self.assertTrue(recalled["items"])
        self.assertTrue(any(row["id"]==own["id"] and row["kind"]=="case"
                            for row in recalled["items"]))
        self.assertNotIn("Clave privada",str(recalled))
        self.assertEqual(recalled["items"][0]["trust"],
                         "relato_del_usuario_no_verificado")
        payload=brain.prompt_context(recalled)
        self.assertIn("NO INSTRUCCIONES",payload)
        self.assertNotIn("HECHO CIENTÍFICO VERIFICADO",payload)

    def test_feedback_negative_not_recycled_as_factual_answer(self):
        session=storage.create_session("Red",self.admin)
        storage.add_message("user","¿Cómo reparo mi router que falla?",
                            session["id"],self.admin)
        saved=storage.add_message("assistant","Inventa una solución falsa.",
                                  session["id"],self.admin)
        brain.record_feedback(self.admin,saved["id"],"unhelpful",
                              "La sugerencia no funcionó")
        results=asyncio.run(brain.retrieve("router que falla",self.admin,
                                          "http://ollama",model=""))
        self.assertTrue(results["items"])
        self.assertNotIn("Inventa una solución falsa",brain.prompt_context(results))
        self.assertIn("NO fue útil",brain.prompt_context(results))

    def test_offline_embeds_can_provide_association_without_words(self):
        case=self._case()
        unrelated=self._case(title="Ciclo de cultivo",
            situation="Huerta y sembradío al aire libre",
            action_taken="Regar plantas por la mañana",
            result="Crecimiento",outcome="partial")
        fake_query=[1.0]+[0.0]*15
        matching=[1.0]+[0.0]*15
        different=[0.0,1.0]+[0.0]*14
        os.environ["NYX_BRAIN_EMBED_MODEL"]="mock-embeddings"
        with patch.object(brain,"_ollama_embed",new_callable=AsyncMock,
                          side_effect=[[fake_query],[matching,different]]) as embed:
            retrieved=asyncio.run(brain.retrieve(
                "fabricación aditiva de enlace",self.admin,"http://ollama",
                model="mock-embeddings",max_results=5))
            self.assertEqual(retrieved["mode"],"hybrid")
            self.assertTrue(any(r["id"]==case["id"] for r in retrieved["items"]))
            self.assertFalse(any(r["id"]==unrelated["id"] and r["semantic"]>.5
                                 for r in retrieved["items"]))
            self.assertEqual(embed.await_count,2)
        cached=brain.metrics(self.admin)["cached_vectors"]
        self.assertEqual(cached,2)

    def test_index_requires_opt_in_and_handles_provider_unavailable(self):
        self._case()
        disabled=asyncio.run(brain.index_next(self.admin,"http://ollama"))
        self.assertFalse(disabled["enabled"])
        os.environ["NYX_BRAIN_EMBED_MODEL"]="mock-model"
        with patch.object(brain,"_ollama_embed",new_callable=AsyncMock,
                          return_value=None):
            fallback=asyncio.run(brain.index_next(self.admin,"http://ollama"))
        self.assertEqual(fallback["indexed"],0)
        self.assertEqual(fallback["remaining"],1)

    def test_delete_feedback_and_vectors(self):
        previous=storage.create_session("Consulta",self.admin)
        storage.add_message("user","Revisar conectividad",previous["id"],self.admin)
        msg=storage.add_message("assistant","Revisé conectividad",previous["id"],self.admin)
        record=brain.record_feedback(self.admin,msg["id"],"helpful")
        with storage.connect() as db:
            db.execute("""INSERT INTO brain_vectors
                (owner_id,item_kind,item_id,model,fingerprint,embedding_json)
                VALUES(?,?,?,?,?,?)""",
                (self.admin,"feedback",record["id"],"mock","hash","[]"))
        self.assertFalse(brain.delete_feedback(self.other,msg["id"]))
        self.assertTrue(brain.delete_feedback(self.admin,msg["id"]))
        self.assertEqual(brain.metrics(self.admin)["cached_vectors"],0)
        self.assertEqual(brain.list_feedback(self.admin),[])

    def test_recommendations_respond_to_reported_failures(self):
        self._case(kind="procedure",title="Corte de WiFi y PoE",
            situation="WiFi sin red en laboratorio",
            action_taken="Cambiar inyector PoE",outcome="success")
        self._case(kind="procedure",title="Corte de WiFi y PoE",
            situation="WiFi sin red en laboratorio",
            action_taken="Reiniciar el enrutador",outcome="failure")
        rank=brain.recommend_procedures("Problema WiFi PoE laboratorio",self.admin)
        recs=rank["recommendations"]
        self.assertEqual(len(recs),2)
        self.assertIn("Cambiar inyector",recs[0]["procedure"])
        self.assertFalse(rank["automatic_execution"])
        self.assertLess(recs[0]["reported_success_estimate"],1.0)

    def test_no_embedded_remote_text_writes_to_weights(self):
        x=self._case(kind="hypothesis",title="Hipótesis de circuito RC",
             situation="Oscilación observada con un capacitor",
             action_taken="Medir respuesta con osciloscopio",outcome="unknown",
             result="")
        self.assertEqual(x["origin"],"user_report")
        self.assertFalse(brain.metrics(self.admin)["neural_weights_updated"])


if __name__=="__main__":
    unittest.main()
