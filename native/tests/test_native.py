"""Pruebas REALES de entrenamiento PyTorch, sin pesos Qwen ni datos privados."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from nyx_native.checkpoint import restore_model, load_checkpoint
from nyx_native.curate import curate
from nyx_native.dataset import load_corpus, sample_batch
from nyx_native.inference import generate_text
from nyx_native.model import ModelConfig, NativeTransformer
from nyx_native.tokenizer import BOS, EOS, PAD, VOCAB_SIZE, encode, decode
from nyx_native.train import _arguments, train

SAMPLES = Path(__file__).resolve().parents[1] / "sample_corpus"


class NeuralNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tokenizer_bytes_utf8_and_special_tokens(self):
        text = "¿Cómo aprende NYX? ¡Así!"
        tokens = encode(text, add_bos=True, add_eos=True)
        self.assertEqual(tokens[0], BOS)
        self.assertEqual(tokens[-1], EOS)
        self.assertEqual(decode(tokens), text)
        self.assertEqual(VOCAB_SIZE, 259)
        self.assertEqual(decode([PAD, BOS, 65, EOS, 66]), "A")

    def test_dataset_train_validation_document_split(self):
        corpus = load_corpus(SAMPLES, context=32)
        self.assertEqual(corpus.documents, 7)
        self.assertGreater(corpus.train.shape[0], corpus.validation.shape[0])
        rng = torch.Generator().manual_seed(2)
        x,y = sample_batch(corpus.train, batch_size=3, context=32, generator=rng)
        self.assertEqual(x.shape, (3,32))
        self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))

    def test_forward_gradient_updates_weights(self):
        model = NativeTransformer(ModelConfig(context=32, d_model=48,
                                             n_heads=4, n_layers=1, dropout=0))
        before = model.tokens.weight.detach().clone()
        x = torch.randint(0, 255, (2, 32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        _,loss = model(x, x.roll(-1, dims=1))
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        self.assertIsNotNone(model.tokens.weight.grad)
        self.assertGreater(model.tokens.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before,model.tokens.weight.detach()))

    def test_causal_attention_blocks_future_tokens(self):
        model = NativeTransformer(ModelConfig(context=16, d_model=48,
                                             n_heads=4, n_layers=2, dropout=0))
        model.eval()
        x=torch.tensor([[4,5,6,7,8,9,10,11]])
        y=torch.tensor([[4,5,6,7,8,9,10,99]])
        with torch.no_grad():
            a,_=model(x)
            b,_=model(y)
        self.assertTrue(torch.allclose(a[:,:-1],b[:,:-1],rtol=1e-5,atol=1e-6))

    def test_training_checkpoint_resume_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/"weights.pt"
            common = [
                "--data-dir",str(SAMPLES),"--checkpoint",str(checkpoint),
                "--batch-size","2","--context","24","--d-model","48",
                "--layers","1","--heads","4","--eval-every","2",
                "--eval-batches","2","--cpu-threads","2","--seed","54",
            ]
            result=train(_arguments(common+["--steps","4"]))
            self.assertTrue(checkpoint.exists())
            self.assertEqual(result["step"],4)
            self.assertTrue(result["trained_from_scratch"])
            self.assertLess(result["parameter_count"], 500_000)
            model,saved = restore_model(checkpoint)
            self.assertEqual(saved["step"],4)
            self.assertEqual(model.parameter_count,result["parameter_count"])
            generated=generate_text(checkpoint,"Hola",max_new_tokens=10,
                                    seed=5,top_k=10)
            self.assertTrue(generated["experiment"])
            self.assertEqual(generated["step"],4)
            self.assertLessEqual(len(generated["text"].encode("utf8")),30)
            after=train(_arguments(common+["--steps","6","--resume"]))
            self.assertEqual(after["step"],6)
            self.assertEqual(load_checkpoint(checkpoint)["step"],6)

    def test_continued_training_updates_weights_without_overwriting_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)/"original.pt"
            adapted = Path(tmp)/"adapted.pt"
            base = ["--data-dir",str(SAMPLES),
                "--batch-size","2","--context","24","--d-model","48",
                "--layers","1","--heads","4","--eval-every","2",
                "--eval-batches","2","--cpu-threads","2","--seed","18"]
            train(_arguments(base+["--checkpoint",str(parent),"--steps","3"]))
            before=load_checkpoint(parent)
            original=before["model"]["tokens.weight"].clone()
            result=train(_arguments(base+["--checkpoint",str(adapted),
                "--continue-from",str(parent),"--steps","3"]))
            updated=load_checkpoint(adapted)
            self.assertEqual(result["training_mode"],"adaptation")
            self.assertEqual(updated["training_mode"],"adaptation")
            self.assertNotEqual(updated["parent_sha256"],None)
            self.assertFalse(torch.equal(original,updated["model"]["tokens.weight"]))
            self.assertTrue(torch.equal(
                original,load_checkpoint(parent)["model"]["tokens.weight"]))
            self.assertTrue(torch.isfinite(
                torch.tensor(updated["validation_baseline"])))
            with self.assertRaises(ValueError):
                train(_arguments(base+["--checkpoint",str(parent),
                    "--continue-from",str(parent),"--steps","4"]))

    def test_approved_corpus_builder_deduplicates_and_isolates_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"data.jsonl"
            data=[
                {"approved":True,"question":f"Pregunta sobre procedimiento {n}",
                 "answer":"Verificar datos y documentar resultados observables",
                 "evidence":"Prueba registrada por el operador"}
                for n in range(12)
            ]
            data.append(data[0])
            data.append({"approved":False,"question":"No usar datos no aprobados",
                         "answer":"Este ejemplo no está autorizado"})
            source.write_text("\n".join(json.dumps(row,ensure_ascii=False)
                                        for row in data),encoding="utf-8")
            output=Path(tmp)/"prepared"
            result=curate(source,output,shards=3)
            self.assertEqual(result["approved_examples"],12)
            self.assertEqual(result["skipped_unapproved_or_duplicates"],2)
            corpus=load_corpus(output,context=24)
            self.assertEqual(corpus.documents,3)
            self.assertTrue((output/"03_aprobados.txt").is_file())
            self.assertNotIn("No usar datos no aprobados",
                              "\n".join(p.read_text(encoding="utf-8")
                                for p in output.glob("*.txt")))
            with self.assertRaises(ValueError):
                curate(source,output,shards=3)

    def test_validate_prevents_invalid_large_config_and_bad_corpus(self):
        with self.assertRaises(ValueError):
            ModelConfig(context=5000).validate()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/"one.txt").write_text("solo una prueba "*25,encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corpus(path,context=16)
            bad=NativeTransformer(ModelConfig(context=16,d_model=48,
                                               n_heads=4,n_layers=1))
            with self.assertRaises(ValueError):
                bad(torch.randint(0,20,(1,17)))
            with self.assertRaises(ValueError):
                bad(torch.randint(0,20,(1,16)),torch.randint(0,20,(1,12)))

    def test_seeded_sampling_repeats_in_evaluation(self):
        model=NativeTransformer(ModelConfig(context=20,d_model=48,
                                            n_heads=4,n_layers=1,dropout=0))
        prompt=torch.tensor([[BOS,72,105]])
        first=model.generate(prompt,max_new_tokens=9,seed=7,top_k=10)
        second=model.generate(prompt,max_new_tokens=9,seed=7,top_k=10)
        self.assertTrue(torch.equal(first,second))


if __name__ == "__main__":
    unittest.main()
