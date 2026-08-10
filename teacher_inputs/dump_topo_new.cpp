#include "common/json.hpp"
#include "PerfModel/PerfModel.h"
#include "PerfModel/Topology.h"
#include "PerfModel/Device.h"
#include "PerfModel/Link.h"
#include "Sketch/Topology.h"
#include <fstream>
#include <iostream>


int gid = 0;
int SKETCH_EXPLORATION_THREAD_NUM = MAX_THREAD_NUM;
int PARALLEL_THREAD_NUM = MAX_THREAD_NUM / 2;
int PERFMODEL_POOL_SIZE = PARALLEL_THREAD_NUM;

const char *type_name(node_type_e t) {
  switch (t) {
    case NODE_GPU: return "GPU";
    case NODE_SWITCH_NV: return "SWITCH_NV";
    case NODE_SWITCH_NET: return "SWITCH_NET";
    case NODE_NIC: return "NIC";
    default: return "NONE";
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: dump_topo <config.json> <output.json>" << std::endl;
    return 1;
  }
  std::ifstream input(argv[1]);
  if (!input.is_open()) {
    std::cerr << "Failed to open " << argv[1] << std::endl;
    return 1;
  }
  nlohmann::json config;
  input >> config;

  // PXN (same-host-first redirect before crossing to the network) is
  // an artifact of RAIL topologies specifically: cross-rail traffic
  // routes via a same-host GPU first, and Topology::getNpuRoute's PXN
  // branch computes machine_size from topo_info.layer_info[1] under
  // an assumption that only holds for that layout. Clos/pod fabrics
  // route inter-host traffic via NIC->leaf(->spine) directly and PXN
  // is not just unneeded but actively wrong there -- forcing PXN=true
  // universally (this tool's original, rail-only use case) hits
  // "assert(machine_size == 8)" on Clos configs whose layer_info
  // doesn't match rail's. Detect from the config itself instead of
  // hardcoding: only rail configs carry a "switch_topo": "rail" layer.
  bool use_pxn = false;
  for (auto &layer : config.value("topo", nlohmann::json::array())) {
    if (layer.value("switch_topo", "") == "rail") {
      use_pxn = true;
      break;
    }
  }

  auto topo = Simulation::constructTopology(config);

  nlohmann::json out;
  out["num_devices"] = topo->get_devices_count();
  out["num_npus"] = topo->get_npus_count();
  out["devices"] = nlohmann::json::array();
  out["edges"] = nlohmann::json::array();

  for (auto &dev : topo->devices) {
    nlohmann::json d;
    d["id"] = dev->get_id();
    d["type"] = type_name(dev->type);
    out["devices"].push_back(d);
    for (auto &[nbr_id, link] : dev->links) {
      nlohmann::json e;
      e["src"] = dev->get_id();
      e["dst"] = nbr_id;
      e["bandwidth_GBps"] = link->getBandwidthBpns();
      e["latency_ns"] = link->getLatencyns();
      out["edges"].push_back(e);
    }
  }

  // NPU-logical id -> real (sparse) device id. Events/"src_chunk" use
  // the NPU-logical numbering (0..num_npus-1, Topology::npus index),
  // NOT the same as Device::device_id once NICs/switches are
  // interposed (confirmed: host1's GPUs have npu_id 8..15 but real
  // device_id 13..20 in the default 2-host config). Read directly
  // from topo->npus so nothing here is guessed.
  out["npu_to_device"] = nlohmann::json::array();
  for (auto &npu : topo->npus) {
    out["npu_to_device"].push_back(npu.device->get_id());
  }

  // GPU-to-GPU logical routes, via the topology's OWN routing logic
  // (Topology::getNpuRoute), not reimplemented. SyCCL's solved
  // schedules log logical "src_gpu -> dst_gpu" sends that are NOT
  // direct physical edges in general (real hops go through NIC/
  // NVSwitch/network-switch) -- this table lets the Python adapter
  // expand each logical send into its real physical hop chain.
  // use_pxn (see above, detected from config): true only for rail
  // topologies (PXN=false there hits "assert(i + 1 < sw_layers)" for
  // GPU pairs with no shared switch at any layer -- rails route
  // cross-rail traffic via a same-host GPU first, which is exactly
  // what PXN enables); false for everything else, notably Clos/pod,
  // where forcing PXN=true instead hits "assert(machine_size == 8)".
  // layer=0 lets same-machine sends take the NVSwitch fast path where
  // present, independent of use_pxn.
  out["gpu_routes"] = nlohmann::json::array();
  int n = topo->get_npus_count();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      auto route = topo->getNpuRoute(i, j, 0, use_pxn);
      nlohmann::json r;
      r["src_gpu"] = i;
      r["dst_gpu"] = j;
      r["path"] = nlohmann::json::array();
      for (auto &dev : route) {
        r["path"].push_back(dev->get_id());
      }
      out["gpu_routes"].push_back(r);
    }
  }

  std::ofstream o(argv[2]);
  o << out.dump(2);
  std::cerr << "Wrote " << argv[2] << " with " << out["devices"].size()
            << " devices, " << out["edges"].size() << " edges, "
            << out["gpu_routes"].size() << " gpu routes." << std::endl;
  return 0;
}
