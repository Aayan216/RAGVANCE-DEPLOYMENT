"""Export the sentence-transformers backbone to ONNX (build-time only).

Runs during the Render build, after the HF cache verification step, while
torch and sentence-transformers are still installed. Writes ``model.onnx``
next to the other snapshot files, then verifies the exported graph against
torch with onnxruntime before the build may continue.

The runtime embedder (EMBEDDING_BACKEND=onnx) loads this file with only
tokenizers + onnxruntime, so the ~450MB torch/sklearn/pandas stack never
enters the gunicorn worker (Render free instance limit: 512MB).

Usage:  python scripts/export_onnx.py
Exit codes: 0 = exported/verified, non-zero = FATAL (fails the Render build).
"""

import glob
import os
import sys

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OPSET = 17
MIN_ONNX_BYTES = 40_000_000
MAX_ABS_DIFF_TOLERANCE = 1e-3


def hub_candidates():
    hubs = []
    for value in (
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ):
        if value and value not in hubs and os.path.isdir(value):
            hubs.append(value)
    return hubs


def find_snapshot():
    bare = MODEL_NAME.split("/")[-1]
    dir_names = []
    if "/" in MODEL_NAME:
        dir_names.append("models--" + MODEL_NAME.replace("/", "--"))
    dir_names.append("models--sentence-transformers--" + bare)

    found = []
    for hub in hub_candidates():
        for dir_name in dir_names:
            for snap in glob.glob(os.path.join(hub, dir_name, "snapshots", "*")):
                has_weights = any(
                    os.path.isfile(os.path.join(snap, name))
                    for name in ("model.safetensors", "pytorch_model.bin")
                )
                if has_weights and os.path.isfile(os.path.join(snap, "config.json")):
                    found.append(snap)
    if not found:
        sys.exit(
            "FATAL: snapshot with model weights not found for "
            f"{MODEL_NAME}; hubs searched: {hub_candidates() or 'none'}"
        )
    found.sort(key=lambda path: os.path.getmtime(os.path.join(path, "config.json")), reverse=True)
    return found[0]


def load_backbone(snapshot):
    import torch
    from transformers import AutoModel

    class Wrapped(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.m = module

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state

    backbone = AutoModel.from_pretrained(snapshot)
    backbone.eval()
    wrapped = Wrapped(backbone)
    wrapped.eval()
    return wrapped


def tokenize(snapshot, texts):
    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    tok.enable_truncation(max_length=256)
    tok.enable_padding(pad_id=0, pad_token="[PAD]", pad_type_id=0)
    encs = tok.encode_batch(texts)
    return {
        "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
    }


def verify(model_path, snapshot):
    import numpy as np
    import onnxruntime as ort
    import torch

    inputs = tokenize(
        snapshot,
        [
            "The Zephyr protocol review checklist requires operators to verify relay nodes.",
            "short",
        ],
    )

    wrapped = load_backbone(snapshot)
    with torch.no_grad():
        ref = wrapped(
            torch.tensor(inputs["input_ids"]),
            torch.tensor(inputs["attention_mask"]),
            torch.tensor(inputs["token_type_ids"]),
        ).numpy()

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, inputs)[0]
    max_diff = float(np.abs(ref - got).max())
    if not np.isfinite(max_diff) or max_diff > MAX_ABS_DIFF_TOLERANCE:
        sys.exit(
            f"FATAL: ONNX verification failed - max abs diff {max_diff:.6f} "
            f"> tolerance {MAX_ABS_DIFF_TOLERANCE}"
        )
    return max_diff, got.shape


def export(snapshot):
    import torch

    wrapped = load_backbone(snapshot)
    dummy_ids = torch.tensor([[101, 2054, 2003, 102]])
    dummy_mask = torch.tensor([[1, 1, 1, 1]])
    dummy_tti = torch.tensor([[0, 0, 0, 0]])
    out_path = os.path.join(snapshot, "model.onnx")
    torch.onnx.export(
        wrapped,
        (dummy_ids, dummy_mask, dummy_tti),
        out_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    return out_path


def main():
    snapshot = find_snapshot()
    print(f"[export_onnx] snapshot: {snapshot}", flush=True)
    model_path = os.path.join(snapshot, "model.onnx")

    needs_export = True
    if os.path.isfile(model_path) and os.path.getsize(model_path) >= MIN_ONNX_BYTES:
        try:
            max_diff, shape = verify(model_path, snapshot)
            needs_export = False
            print(
                f"[export_onnx] existing model.onnx verified: max_abs_diff={max_diff:.6f} "
                f"shape={shape}",
                flush=True,
            )
        except SystemExit as exc:
            print(f"[export_onnx] existing file failed verification ({exc}); re-exporting", flush=True)
    if needs_export:
        out_path = export(snapshot)
        size = os.path.getsize(out_path)
        if size < MIN_ONNX_BYTES:
            sys.exit(f"FATAL: exported model.onnx too small ({size} bytes)")
        max_diff, shape = verify(out_path, snapshot)
        print(
            f"[export_onnx] exported {out_path} ({size} bytes); "
            f"verified max_abs_diff={max_diff:.6f} shape={shape}",
            flush=True,
        )
    print("[export_onnx] OK", flush=True)


if __name__ == "__main__":
    main()
