"""
GNN gap diagnostic 3 — layer order verification.

Static claim (from directly reading src/polyllm/models/gnn.py:96-109,
GNNEncoder.forward): conv1 -> bn1 -> leaky_relu -> dropout1 -> conv2 -> bn2
-> leaky_relu -> dropout2 -> conv3 -> lin1, with nothing (no BN, activation,
or dropout) after conv3/lin1. This matches the authors' GNN.py exactly
(bn3/bn4 and a dropout_rate constructor arg are declared in the authors'
code but never referenced in forward() — confirmed dead code, intentionally
omitted here rather than reproduced as inert clutter).

This script re-confirms that claim at runtime rather than by source reading
alone:
  1. named_modules() contains exactly conv1/bn1/dropout1/conv2/bn2/dropout2/
     conv3/lin1 -- no bn3/bn4/dropout3/etc secretly present.
  2. forward hooks record the actual module call order during a real
     forward pass.
  3. With the model in eval() mode (dropout becomes identity, so its input
     and output are equal and directly comparable), dropout1/dropout2's
     hooked INPUT is checked to equal F.leaky_relu(bn_output) elementwise --
     proving leaky_relu sits exactly between each BN and dropout, not
     functionally invisible to a pure module-hook trace.
  4. lin1's hooked INPUT is checked to equal conv3's hooked OUTPUT exactly
     (no activation/dropout in between), and the model's final return value
     is checked to equal lin1's hooked OUTPUT exactly (nothing after it).

No model is trained; this uses a freshly-initialized GNNEncoder with random
input, since layer order is a structural property independent of trained
weights.

Run from the project root:
    .venv-polyllm/Scripts/python.exe src/polyllm/diagnostics/verify_layer_order.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_src = Path(__file__).resolve().parents[2]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from polyllm.models.gnn import GNNEncoder  # noqa: E402

OUTPUT_PATH = Path("outputs/polyllm/gnn_diagnostics/layer_order_audit.json")

EXPECTED_MODULE_NAMES = {"conv1", "bn1", "dropout1", "conv2", "bn2", "dropout2", "conv3", "lin1"}
EXPECTED_CALL_ORDER = ["conv1", "bn1", "dropout1", "conv2", "bn2", "dropout2", "conv3", "lin1"]

STATIC_SOURCE_EXCERPT = (
    "def forward(self, x, edge_index):\n"
    "    x = self.conv1(x, edge_index)\n"
    "    x = self.bn1(x)\n"
    "    x = F.leaky_relu(x)\n"
    "    x = self.dropout1(x)\n"
    "\n"
    "    x = self.conv2(x, edge_index)\n"
    "    x = self.bn2(x)\n"
    "    x = F.leaky_relu(x)\n"
    "    x = self.dropout2(x)\n"
    "\n"
    "    x = self.conv3(x, edge_index)\n"
    "    x = self.lin1(x)\n"
    "    return x"
)


def main() -> None:
    torch.manual_seed(0)
    hidden_channels = 64
    model = GNNEncoder(hidden_channels)
    model.eval()  # dropout -> identity, so hooked input == output, directly comparable

    # --- Check 1: no extra/dead modules present at runtime ---------------
    module_names = {name for name, _ in model.named_children()}
    check_no_extra_modules = module_names == EXPECTED_MODULE_NAMES
    print(f"named_children(): {sorted(module_names)}")
    print(f"Expected:          {sorted(EXPECTED_MODULE_NAMES)}")
    print(f"[{'OK' if check_no_extra_modules else 'FAIL'}] no extra/dead modules (bn3/bn4/dropout_rate confirmed absent)")

    # --- Checks 2-4: hook-based runtime trace -----------------------------
    call_order: list[str] = []
    captured_inputs: dict[str, torch.Tensor] = {}
    captured_outputs: dict[str, torch.Tensor] = {}

    def make_forward_hook(name: str):
        def hook(module, inputs, output):
            call_order.append(name)
            captured_outputs[name] = output.detach().clone()
        return hook

    def make_pre_hook(name: str):
        def hook(module, inputs):
            captured_inputs[name] = inputs[0].detach().clone()
        return hook

    handles = []
    for name in EXPECTED_CALL_ORDER:
        submodule = getattr(model, name)
        handles.append(submodule.register_forward_hook(make_forward_hook(name)))
        handles.append(submodule.register_forward_pre_hook(make_pre_hook(name)))

    n_nodes = 200
    num_edges = 800
    x = torch.randn(n_nodes, hidden_channels)
    edge_index = torch.randint(0, n_nodes, (2, num_edges))

    with torch.no_grad():
        final_output = model(x, edge_index)

    for h in handles:
        h.remove()

    check_call_order = call_order == EXPECTED_CALL_ORDER
    print(f"\nRecorded call order: {call_order}")
    print(f"Expected call order: {EXPECTED_CALL_ORDER}")
    print(f"[{'OK' if check_call_order else 'FAIL'}] call order matches expected sequence")

    # leaky_relu sits between bn1 and dropout1 (and bn2/dropout2): the value
    # fed INTO dropout must equal leaky_relu applied to bn's OUTPUT.
    leaky_relu_checks = {}
    for bn_name, dropout_name in [("bn1", "dropout1"), ("bn2", "dropout2")]:
        expected = F.leaky_relu(captured_outputs[bn_name])
        actual = captured_inputs[dropout_name]
        matches = torch.allclose(expected, actual, atol=1e-6)
        leaky_relu_checks[f"leaky_relu_between_{bn_name}_and_{dropout_name}"] = bool(matches)
        print(f"[{'OK' if matches else 'FAIL'}] leaky_relu applied between {bn_name} and {dropout_name}")

    # conv3's output feeds lin1 directly -- no transform in between.
    conv3_to_lin1_direct = torch.equal(captured_outputs["conv3"], captured_inputs["lin1"])
    print(f"[{'OK' if conv3_to_lin1_direct else 'FAIL'}] conv3 output feeds lin1 input directly (no BN/activation/dropout)")

    # Nothing after lin1 -- final return value IS lin1's output.
    nothing_after_lin1 = torch.equal(captured_outputs["lin1"], final_output)
    print(f"[{'OK' if nothing_after_lin1 else 'FAIL'}] final output equals lin1 output directly (nothing after lin1)")

    all_checks = {
        "no_extra_or_dead_modules": check_no_extra_modules,
        "call_order_matches_expected": check_call_order,
        **leaky_relu_checks,
        "conv3_output_feeds_lin1_input_directly": conv3_to_lin1_direct,
        "nothing_after_lin1": nothing_after_lin1,
    }
    all_passed = all(all_checks.values())

    result = {
        "static_source_excerpt": STATIC_SOURCE_EXCERPT,
        "expected_call_order": EXPECTED_CALL_ORDER,
        "recorded_call_order": call_order,
        "runtime_module_names": sorted(module_names),
        "checks": all_checks,
        "all_passed": all_passed,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if not all_passed:
        raise AssertionError(f"Layer order verification FAILED: {all_checks}")

    print(f"\nAll layer-order checks passed. Results saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
