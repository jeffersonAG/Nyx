"""Tests de persistencia (sin Docker)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from app import storage

class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prev = os.environ.get('NYX_DB_PATH')
        os.environ['NYX_DB_PATH'] = str(Path(self.tmp.name) / 'nyx.db')
        storage.init_db()
    def tearDown(self):
        if self.prev is None: os.environ.pop('NYX_DB_PATH', None)
        else: os.environ['NYX_DB_PATH'] = self.prev
        self.tmp.cleanup()
    def test_messages(self):
        storage.add_message('user','Hola')
        storage.add_message('assistant','Saludos')
        self.assertEqual([m['role'] for m in storage.list_messages()], ['user','assistant'])
    def test_tasks(self):
        t = storage.add_task('Estudiar')
        self.assertTrue(storage.update_task(t['id'], True)['completed'])
        self.assertTrue(storage.delete_task(t['id']))
        self.assertEqual(storage.list_tasks(), [])
    def test_memories(self):
        m = storage.add_memory('Aprender FastAPI')
        self.assertEqual(storage.list_memories()[0]['content'],'Aprender FastAPI')
        self.assertTrue(storage.delete_memory(m['id']))
    def test_safe_sql(self):
        storage.add_memory("'); DROP TABLE tasks; --")
        storage.add_task('La tabla sigue')
        self.assertEqual(len(storage.list_tasks()),1)

if __name__ == '__main__': unittest.main()
