"""Exercise the actual inference loop using deterministic model responses."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import pathagent
from models.runtime import model_load_kwargs, runtime_metadata
from models.input_validation import checkpoint_report, validate_inputs


ROOT = Path(__file__).resolve().parents[1]


class FakePLIP:
    device = "cpu"

    def encode_text(self, texts, **kwargs):
        return np.array([[1., 0.]] * len(texts))

    def encode_images(self, images, **kwargs):
        # Select the bottom-right crop to test a nonzero origin offset.
        return np.array([[1., float(len(images) - i - 1)] for i in range(len(images))])


class ConsolidationTests(unittest.TestCase):
    def test_runner_allocation_rejects_three_gpus(self):
        import argparse
        from tools.run_wsi_vqa import parse_gpu_ids
        self.assertEqual(parse_gpu_ids('2'),[2])
        self.assertEqual(parse_gpu_ids('0,2'),[0,2])
        for value in ('0,1,2','0,0','-1','auto',''):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_gpu_ids(value)

    def test_server_profiles_use_at_most_two_gpus(self):
        for name in ('server_original.json','wsi_vqa_paper.json'):
            config=json.loads((ROOT/'configs'/name).read_text())
            devices={config[k] for k in ('executor_device_map','perceptor_device_map','plip_device')}
            self.assertLessEqual(len(devices),2)

    def args(self, *extra):
        args = pathagent.parse_args([
            "--config", str(ROOT / "configs/local_8gb.json"),
            "--descriptions_file", "unused", "--questions_file", "unused",
            "--feature_dir", "unused", "--patch_root", "unused", "--save_dir", "unused", *extra])
        # File preflight uses synthetic metadata only; no local models or data required.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        library = root / "plip_source"
        library.mkdir()
        (library / "plip.py").write_text("# Synthetic preflight fixture\n")
        args.plip_lib_path = str(library)
        for role in ("qwen_ckpt", "patho_r1_ckpt", "plip_ckpt"):
            model = root / role
            model.mkdir()
            original = args.profile_name == "server_original"
            config = {"model_type": "qwen2_5_vl", "hidden_size": 3584 if original else 2048,
                      "num_hidden_layers": 28 if original else 36}
            (model / "config.json").write_text(json.dumps(config))
            (model / "model.safetensors").write_bytes(b"fixture-not-loadable")
            for filename in ("tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
                (model / filename).write_text("{}")
            setattr(args, role, str(model))
        for name in ("feature_dir", "patch_root"):
            folder = root / name
            folder.mkdir()
            setattr(args, name, str(folder))
        return args

    def run_case(self, decisions, count=3, minimum=1, attempts=3):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        patches, features = root / "patches" / "DX1_CASE", root / "features" / "DX1_CASE"
        patches.mkdir(parents=True)
        features.mkdir(parents=True)
        descriptions = {}
        for i in range(count):
            name = f"{100 + i * 4096}_200.png"
            Image.new("RGB", (512, 512), "pink").save(patches / name)
            np.save(features / (Path(name).stem + ".npy"), np.array([1., float(i)]))
            descriptions[name] = "generic description"
        for filename, data in {
            "descriptions.json": {"DX1_CASE": descriptions},
            "questions.json": [{"Id": "DX1_CASE", "Question": "Test question?", "Choice": ["A", "B"], "Answer": "B"}],
            "geometry.json": {"DX1_CASE": {"coordinate_scale": 4, "base_magnification": 10,
                                           "coordinate_space": "CAMELYON16_level0", "level0_verified": True}},
        }.items():
            (root / filename).write_text(json.dumps(data), encoding="utf-8")
        args = self.args("--initial_sample_count", "1", "--max_attempts", str(attempts),
                         "--min_observation_rounds", str(minimum), "--perceptor_max_new_tokens", "37")
        args.patch_root, args.feature_dir = str(patches.parent), str(features.parent)
        args.save_dir = str(root / "results")
        args.descriptions_file = str(root / "descriptions.json")
        args.questions_file = str(root / "questions.json")
        args.geometry_file = str(root / "geometry.json")
        # Keep data validation real; skip only unavailable test-model checkpoint files.
        fake = SimpleNamespace(hf_device_map={"": "cpu"}, dtype="test")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("models.input_validation.checkpoint_report", return_value={}))
            # Temporary tokenizer/processor metadata above keeps preflight real.
            stack.enter_context(patch.object(pathagent, "load_models", return_value=(None, fake, FakePLIP(), fake, None)))
            evaluate = stack.enter_context(patch.object(pathagent, "evaluate_with_llm_chain", side_effect=decisions))
            describe = stack.enter_context(patch.object(pathagent, "patho_r1_describe", return_value="observed tissue"))
            stack.enter_context(patch.object(pathagent, "summarize_patches_in_chunks", return_value="summary"))
            answer = stack.enter_context(patch.object(pathagent, "slide_llm_answer", return_value={"answer": "A", "explanation": "test"}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            results = pathagent.main(args)
        result = results[0]
        saved = list((root / "results").glob("*_trajectory.json"))
        self.assertEqual(len(saved), 1)
        self.assertTrue(Path(result["trajectory_files"]["image"]).is_file())
        events = [json.loads(line) for line in Path(result["trajectory_files"]["events"]).read_text().splitlines()]
        self.assertEqual(events[-1]["event"], "run_completed")
        self.assertEqual([p["step"] for p in result["trajectory"]], list(range(1, len(result["trajectory"]) + 1)))
        self.assertTrue(all(p["status"] == "completed" for p in result["trajectory"]))
        self.assertEqual(result["runtime"]["quantization"], "nf4")
        for call in evaluate.call_args_list:
            self.assertNotIn("ground_truth", call.kwargs)
        return result, json.loads(saved[0].read_text()), events, describe, answer

    def test_single_candidate_stops_and_exports_level0_center(self):
        result, trace, _, _, _ = self.run_case([{"sufficient": "Yes"}], count=1)
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(result["process"][0]["mode"], "can_answer_now")
        self.assertEqual((trace["trajectory"][0]["x_px"], trace["trajectory"][0]["y_px"]), (1124, 1224))

    def test_missing_information_keeps_history(self):
        result, _, _, _, _ = self.run_case([
            {"sufficient": "No", "zoom_recommendation": "No", "missing_info": "tumor cells"},
            {"sufficient": "Yes"}])
        self.assertEqual([r["mode"] for r in result["process"]], ["find_missing", "can_answer_now"])
        self.assertEqual(len({p["patch_name"] for p in result["trajectory"]}), 2)

    def test_zoom_uses_offset_scale_and_token_limit(self):
        result, trace, _, describe, _ = self.run_case([
            {"sufficient": "No", "zoom_recommendation": "Yes", "zoom_level": 20, "zoom_reason": "detail"}], count=1)
        zoom = trace["trajectory"][-1]
        self.assertEqual((zoom["x"], zoom["y"], zoom["x_px"], zoom["y_px"]), (1124, 1224, 1636, 1736))
        self.assertEqual((zoom["magnification"], zoom["width"]), (20, 256))
        self.assertEqual(describe.call_args.kwargs["max_new_tokens"], 37)
        self.assertEqual(result["process"][0]["mode"], "zoom_then_select")

    def test_explicit_minimum_round_guard_is_logged(self):
        result, _, events, _, _ = self.run_case([{"sufficient": "Yes"}, {"sufficient": "Yes"}], minimum=2)
        self.assertEqual(len(result["trajectory"]), 2)
        self.assertEqual(sum(e["event"] == "minimum_rounds_guard" for e in events), 1)

    def test_budget_exhaustion_keeps_fallback(self):
        result, _, _, _, _ = self.run_case([
            {"sufficient": "No", "zoom_recommendation": "No", "missing_info": "details"}], count=1, attempts=1)
        self.assertEqual([r["mode"] for r in result["process"]], ["find_missing", "fallback_final_attempt"])

    def test_server_has_no_quantization_and_cli_overrides_config(self):
        args = self.args("--config", str(ROOT / "configs/server_original.json"), "--perceptor-device-map", "cuda:2")
        kwargs = model_load_kwargs(args, "perceptor")
        self.assertNotIn("quantization_config", kwargs)
        self.assertNotIn("max_memory", kwargs)
        self.assertEqual(kwargs["device_map"], "cuda:2")
        self.assertEqual(str(kwargs["torch_dtype"]), "torch.bfloat16")
        self.assertTrue(Path(args.patho_r1_ckpt).is_absolute())
        self.assertFalse(args.allow_empty_descriptions)

    def test_runtime_metadata_serializes_actual_device_objects(self):
        import torch

        executor = SimpleNamespace(hf_device_map={"": torch.device("cuda:0")}, dtype=torch.bfloat16)
        perceptor = SimpleNamespace(hf_device_map={"layer": torch.device("cuda:1"), "other": 1}, dtype=torch.bfloat16)
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.device_count", return_value=0):
            metadata = runtime_metadata(self.args(), executor, perceptor, FakePLIP())
        saved = json.loads(json.dumps(metadata))
        self.assertEqual(saved["executor_device_map"], {"": "cuda:0"})
        self.assertEqual(saved["perceptor_device_map"], {"layer": "cuda:1", "other": 1})
        self.assertEqual(saved["executor_dtype"], "torch.bfloat16")

    def test_preflight_only_does_not_load_models(self):
        args = self.args("--check-inputs")
        with patch.object(pathagent, "validate_inputs", return_value={"ok": True}), patch.object(pathagent, "load_models") as loader:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pathagent.main(args), {"ok": True})
            loader.assert_not_called()

    def test_server_preflight_rejects_substitute(self):
        args = self.args()
        args.profile_name = "server_original"
        with self.assertRaisesRegex(ValueError, "3B substitute"):
            validate_inputs(args)

    def test_empty_descriptions_require_explicit_smoke_permission(self):
        args = self.args("--no-allow-empty-descriptions")
        base = Path(args.patch_root).parent
        args.questions_file = str(base / "questions.json")
        args.descriptions_file = str(base / "descriptions.json")
        (Path(args.patch_root) / "DX1_FIXTURE").mkdir()
        Path(args.questions_file).write_text(json.dumps([
            {"Id": "DX1_FIXTURE", "Question": "Test?", "Choice": ["A", "B"], "Answer": "B"}]))
        Path(args.descriptions_file).write_text(json.dumps({"DX1_FIXTURE": {"0_0.png": ""}}))
        with self.assertRaisesRegex(ValueError, "empty generic descriptions"):
            validate_inputs(args)


if __name__ == "__main__":
    unittest.main()
