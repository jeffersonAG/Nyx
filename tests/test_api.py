"""Tests HTTP locales (sin Ollama)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from fastapi.testclient import TestClient
from app.main import app

class APITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev = os.environ.get('NYX_DB_PATH')
        os.environ['NYX_DB_PATH'] = str(Path(self.tmp.name) / 'nyx.db')
        self.client = TestClient(app)
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None, None, None)
        if self.prev is None: os.environ.pop('NYX_DB_PATH',None)
        else: os.environ['NYX_DB_PATH'] = self.prev
        self.tmp.cleanup()
    def test_tasks(self):
        t = self.client.post('/api/tasks',json={'text':'Estudiar'})
        self.assertEqual(t.status_code,201)
        tid = t.json()['id']
        self.assertEqual(self.client.patch('/api/tasks/'+str(tid),json={'completed':True}).status_code,200)
        self.assertTrue(self.client.get('/api/tasks').json()[0]['completed'])
        self.assertEqual(self.client.delete('/api/tasks/'+str(tid)).status_code,204)
    def test_memory(self):
        r = self.client.post('/api/memory',json={'text':'Mi proyecto'})
        self.assertEqual(r.status_code,201)
        self.assertEqual(len(self.client.get('/api/memory').json()),1)
        self.assertEqual(self.client.delete('/api/memory/'+str(r.json()['id'])).status_code,204)
    def test_validation(self):
        self.assertEqual(self.client.post('/api/tasks',json={'text':''}).status_code,422)
        self.assertEqual(self.client.post('/api/chat',json={'text':'x'*4001}).status_code,422)

if __name__ == '__main__': unittest.main()
