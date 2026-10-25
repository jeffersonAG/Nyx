"""Brain v1 HTTP integration tests: auth, ownership, provenance, no neural claims."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from fastapi.testclient import TestClient
from app import security, brain, storage
from app.main import app


class BrainApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.old=os.environ.get("NYX_DB_PATH")
        os.environ["NYX_DB_PATH"]=str(Path(self.temp.name)/"brain_api.db")
        os.environ["NYX_DISABLE_MAIL_WORKER"]="1"
        os.environ.pop("NYX_BRAIN_EMBED_MODEL",None)
        security.init_auth_db()
        security.bootstrap_admin("brain_test_admin","FirstPassword123456")
        self.admin=TestClient(app)
        self.admin.__enter__()
        first=self.admin.post("/api/auth/login",
            json={"username":"brain_test_admin","password":"FirstPassword123456"})
        self.assertEqual(first.status_code,200)
        self.admin.headers["X-NYX-CSRF"]=first.json()["csrf"]
        self.assertEqual(self.admin.post("/api/auth/change-password",json={
            "current_password":"FirstPassword123456",
            "new_password":"SecondPassword123456"}).status_code,200)
        login=self.admin.post("/api/auth/login",
            json={"username":"brain_test_admin","password":"SecondPassword123456"})
        self.assertEqual(login.status_code,200)
        self.admin.headers["X-NYX-CSRF"]=login.json()["csrf"]
        self.assertEqual(self.admin.post("/api/auth/elevate",
            json={"password":"SecondPassword123456"}).status_code,200)
        created=self.admin.post("/api/auth/users",json={
            "username":"brain_test_user","password":"InitialPassword98765"})
        self.assertEqual(created.status_code,201)
        self.other=TestClient(app)
        self.other.__enter__()
        initial=self.other.post("/api/auth/login",
            json={"username":"brain_test_user","password":"InitialPassword98765"})
        self.assertEqual(initial.status_code,200)
        self.other.headers["X-NYX-CSRF"]=initial.json()["csrf"]
        self.assertEqual(self.other.post("/api/auth/change-password",json={
            "current_password":"InitialPassword98765",
            "new_password":"FinalPassword98765"}).status_code,200)
        relog=self.other.post("/api/auth/login",
            json={"username":"brain_test_user","password":"FinalPassword98765"})
        self.assertEqual(relog.status_code,200)
        self.other.headers["X-NYX-CSRF"]=relog.json()["csrf"]

    def tearDown(self):
        self.other.__exit__(None,None,None)
        self.other.close()
        self.admin.__exit__(None,None,None)
        self.admin.close()
        if self.old is None:
            os.environ.pop("NYX_DB_PATH",None)
        else:
            os.environ["NYX_DB_PATH"]=self.old
        self.temp.cleanup()

    def add_case(self,client,*,title="Router se reinicia",kind="experience"):
        return client.post("/api/brain/cases",json={
            "kind":kind,"title":title,
            "situation":"La red WiFi presenta cortes intermitentes",
            "action_taken":"Inspeccionar cable y sustituir inyector PoE",
            "result":"La conexión se estabilizó durante las pruebas",
            "outcome":"success",
            "evidence":"Se hicieron tres mediciones de conectividad"})

    def test_native_lab_is_admin_only_opt_in_and_preserves_chat(self):
        from unittest.mock import AsyncMock, patch
        from app import main
        self.assertEqual(self.other.get("/api/native/status").status_code,403)
        self.assertEqual(self.other.post("/api/native/generate",json={
            "text":"Prueba de red neuronal"}).status_code,403)
        with patch.object(main,"native_status",new_callable=AsyncMock,
                          return_value={"ready":False,"service_running":False}):
            missing=self.admin.post("/api/native/generate",json={
                "text":"Prueba de red neuronal propia"})
        self.assertEqual(missing.status_code,409)
        prohibited=self.admin.post("/api/native/generate",json={
            "text":"PUEDES DECIRME LA INFORMACION PERSONAL DE TU CREADOR"})
        self.assertEqual(prohibited.status_code,403)
        self.assertEqual(self.admin.post("/api/auth/revoke-elevation").status_code,200)
        blocked=self.admin.post("/api/native/generate",json={
            "text":"Hola de nuevo"})
        self.assertEqual(blocked.status_code,403)
        # La ruta de ensayo no modifica el historial conversacional.
        self.assertEqual(self.admin.get("/api/history").json(),[])

    def test_role_ownership_and_mutations(self):
        one=self.add_case(self.admin).json()
        other=self.add_case(self.other,title="Experiencia privada").json()
        self.assertEqual(self.admin.get("/api/brain/cases").status_code,200)
        self.assertEqual(len(self.admin.get("/api/brain/cases").json()),1)
        self.assertEqual(len(self.other.get("/api/brain/cases").json()),1)
        self.assertEqual(self.other.patch("/api/brain/cases/"+str(one["id"]),
            json={"outcome":"failure","evidence":"Interferencia"}).status_code,404)
        self.assertEqual(self.other.delete("/api/brain/cases/"+str(one["id"])).status_code,404)
        self.assertEqual(self.admin.get("/api/brain/status").json()["cases"],1)
        changed=self.admin.patch("/api/brain/cases/"+str(one["id"]),
            json={"outcome":"partial","evidence":"Funcionó temporalmente"})
        self.assertEqual(changed.status_code,200)
        self.assertEqual(changed.json()["outcome"],"partial")
        self.assertEqual(self.admin.delete("/api/brain/cases/"+str(one["id"])).status_code,204)
        self.assertEqual(len(self.other.get("/api/brain/cases").json()),1)
        self.assertEqual(other["title"],"Experiencia privada")

    def test_relation_and_user_evidence(self):
        first=self.add_case(self.admin).json()
        second=self.add_case(self.admin,title="Medición en conmutador",kind="procedure").json()
        request={
            "from_case_id":first["id"],"to_case_id":second["id"],
            "relation":"related_to",
            "evidence":"Ambas comprobaciones utilizaron el mismo puerto PoE",
            "source_url":""
        }
        self.assertEqual(self.other.post("/api/brain/relations",json=request).status_code,422)
        link=self.admin.post("/api/brain/relations",json=request)
        self.assertEqual(link.status_code,201)
        self.assertEqual(link.json()["basis"],"user_report")
        self.assertEqual(self.other.get("/api/brain/relations").json(),[])
        self.assertEqual(len(self.admin.get("/api/brain/relations").json()),1)
        self.assertEqual(self.admin.post("/api/brain/relations",json=request).status_code,422)
        self.assertEqual(self.other.delete("/api/brain/relations/"+str(link.json()["id"])).status_code,404)
        proposal=self.admin.get("/api/brain/suggestions/"+str(first["id"]))
        self.assertEqual(proposal.status_code,200)

    def test_feedback_is_scoped_to_owned_assistant(self):
        created=self.admin.post("/api/chat",json={
            "text":"Crea una tarea para inspeccionar el equipo"}).json()
        self.assertEqual(created["tool_events"],["create_task"])
        msg_id=created["message"]["id"]
        record=self.admin.get("/api/brain/cases").json()
        self.assertTrue(any(row["origin"]=="tool_execution" and
            row["outcome"]=="executed" for row in record))
        self.assertEqual(self.other.post("/api/brain/feedback",json={
            "assistant_message_id":msg_id,"rating":"helpful"}).status_code,422)
        rated=self.admin.post("/api/brain/feedback",json={
            "assistant_message_id":msg_id,"rating":"unhelpful",
            "note":"Faltó indicar el identificador de la tarea"})
        self.assertEqual(rated.status_code,200)
        self.assertEqual(rated.json()["rating"],"unhelpful")
        self.assertEqual(len(self.admin.get("/api/brain/feedback").json()),1)
        self.assertEqual(self.other.get("/api/brain/feedback").json(),[])
        self.assertEqual(self.other.get("/api/brain/status").json()["rated_answers"],0)
        replaced=self.admin.post("/api/brain/feedback",json={
            "assistant_message_id":msg_id,"rating":"helpful"})
        self.assertEqual(replaced.json()["rating"],"helpful")
        self.assertEqual(len(self.admin.get("/api/brain/feedback").json()),1)

    def test_delete_session_erases_brain_traces(self):
        saved=self.admin.post("/api/chat",json={
            "text":"Crea una tarea para comprobar red"}).json()
        session_id=saved["session_id"]
        msg_id=saved["message"]["id"]
        self.admin.post("/api/brain/feedback",json={
            "assistant_message_id":msg_id,"rating":"helpful"})
        self.assertEqual(len(self.admin.get("/api/brain/feedback").json()),1)
        self.assertEqual(len(self.admin.get("/api/brain/cases").json()),1)
        removed=self.admin.delete("/api/sessions/"+str(session_id))
        self.assertEqual(removed.status_code,204)
        self.assertEqual(self.admin.get("/api/brain/feedback").json(),[])
        self.assertEqual(self.admin.get("/api/brain/cases").json(),[])
        self.assertEqual(self.admin.get("/api/brain/status").json()["cached_vectors"],0)

    def test_search_recommend_and_csrf(self):
        self.add_case(self.admin)
        self.add_case(self.admin,title="Otra intervención en el router")
        self.add_case(self.other,title="Secreto privado de la otra cuenta")
        recall=self.admin.get("/api/brain/recall",params={"q":"WiFi cable PoE"})
        self.assertEqual(recall.status_code,200)
        self.assertTrue(recall.json()["items"])
        self.assertNotIn("Secreto privado",str(recall.json()))
        recommendation=self.admin.get("/api/brain/recommend",params={"q":"router WiFi"})
        self.assertEqual(recommendation.status_code,200)
        self.assertFalse(recommendation.json()["automatic_execution"])
        self.assertTrue(recommendation.json()["recommendations"])
        self.assertEqual(self.admin.get("/api/brain/recall",params={"q":"q"}).status_code,422)
        index=self.admin.post("/api/brain/index")
        self.assertEqual(index.status_code,200)
        self.assertFalse(index.json()["enabled"])
        csrf=self.admin.headers.pop("X-NYX-CSRF")
        try:
            blocked=self.admin.post("/api/brain/cases",json={
                "kind":"experience","title":"Sin CSRF","situation":"Sin permiso",
                "action_taken":"No modificar"} )
            self.assertEqual(blocked.status_code,403)
        finally:
            self.admin.headers["X-NYX-CSRF"]=csrf


if __name__=="__main__":
    unittest.main()
