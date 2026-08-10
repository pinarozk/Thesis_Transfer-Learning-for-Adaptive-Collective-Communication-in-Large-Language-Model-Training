import json

REPO_CONFIG_DIR = "/root/SyCCL/config"
OUT_DIR = "/mnt/c/Users/Pınar/Desktop/Thesis/AgentGNN/teacher_inputs/syccl"
RESULT_DIR = "/mnt/c/Users/Pınar/Desktop/Thesis/AgentGNN/data/raw/syccl_runs"

# Use the PAPER'S OWN proven leaf-spine/clos and rail configs as
# templates (do not hand-generate topology params -- ConfigGen's
# clos_topo(nleaf=1) was degenerate (0 sketches) and nleaf=2 crashed
# the solver; the repo's own shipped configs are known-correct).
# Only message size is varied, same pattern as default.json.
TEMPLATES = [
    ("a100-8gpu-4nic-clos-ag.json", "clos_ag"),
    ("a100-8gpu-4nic-clos-a2a.json", "clos_a2a"),
    ("h800-8gpu-8nic-rail.json", "rail_ag"),
]
SIZES = [
    (4096,    "4KB"),
    (32768,   "32KB"),
    (262144,  "256KB"),
    (1048576, "1MB"),
]

for fname, tag in TEMPLATES:
    base = json.load(open(f"{REPO_CONFIG_DIR}/{fname}"))
    for byte, size_tag in SIZES:
        conf = json.loads(json.dumps(base))
        conf["coll"]["byte"] = byte
        out_name = f"syccl_{tag}_{size_tag}"
        conf.setdefault("algo_solve", {})["solve_output"] = \
            f"{RESULT_DIR}/{out_name}_result.json"
        if "sketch" in conf:
            conf["sketch"]["sketch_path"] = f"{RESULT_DIR}/{out_name}_sketch.json"
            conf["sketch"]["save_sketch"] = False
        path = f"{OUT_DIR}/{out_name}.json"
        with open(path, "w") as f:
            f.write(json.dumps(conf, indent=2))
        print(f"wrote {path}")
