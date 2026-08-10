import json
import sys
sys.path.insert(0, "/root/SyCCL/scripts")
from config_gen import ConfigGen

OUT_DIR = "/mnt/c/Users/Pınar/Desktop/Thesis/AgentGNN/data/raw/syccl_runs"

cfg_gen = ConfigGen()
cfg_gen.set_solve_br(0.2)
conf = cfg_gen.a100conf(
    32768, 2, 8, 4,
    nleaf=2, nspine=1, prune_type="small",
    solve_output=f"{OUT_DIR}/test_clos_result.json",
    sketch_output=f"{OUT_DIR}/test_clos_sketch.json",
    a2a=False,
)
conf["sketch"]["save_sketch"] = False
path = f"{OUT_DIR}/test_clos.json"
with open(path, "w") as f:
    f.write(json.dumps(conf, indent=2))
print("wrote", path)
