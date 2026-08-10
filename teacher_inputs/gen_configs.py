import json

BASE = "/mnt/c/Users/Pınar/Desktop/Thesis/AgentGNN/teacher_inputs/syccl/default.json"
OUT_DIR = "/mnt/c/Users/Pınar/Desktop/Thesis/AgentGNN/teacher_inputs/syccl"

# clos_topo(nleaf=1, nspine=1) produced 0 sketches at this scale (a
# real degenerate-parameter bug in that generator, verified: same
# "0 generated" at both 4 and 8 GPUs/host, unrelated to scale). The
# safe path is default.json's OWN topology (single-level "pod"
# switch, proven working), varying only message size.
SIZES = [
    (4096,    "syccl_default_4KB"),
    (32768,   "syccl_default_32KB"),   # == original default.json
    (262144,  "syccl_default_256KB"),
    (1048576, "syccl_default_1MB"),
    (4194304, "syccl_default_4MB"),
]

base = json.load(open(BASE))

for byte, tag in SIZES:
    conf = json.loads(json.dumps(base))   # deep copy
    conf["coll"]["byte"] = byte
    path = f"{OUT_DIR}/{tag}.json"
    with open(path, "w") as f:
        f.write(json.dumps(conf, indent=2))
    print(f"wrote {path}")
