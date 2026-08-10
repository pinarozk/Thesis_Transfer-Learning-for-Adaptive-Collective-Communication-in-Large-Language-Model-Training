# CCL Simulator

## What it is
- A packet-level, event-driven network simulator specialized for Collective Communication Library (CCL) workloads.
- Designed to model CCL execution semantics (multi-stage propagation, fan-out/fan-in, dependency-triggered scheduling) under controlled network conditions.

## Why it exists
- General-purpose network simulators often focus on transport protocols and congestion dynamics.
- This simulator focuses on CCL scheduling and dependencies: when chunks become available and how they are propagated according to a user-defined policy.

## Input format (policy)
A simulation is driven by a list of policy entries.

    [chunk_id, src, dst, qpid, rate, chunk_size_bytes, path, time, dependency]

### Field breakdown
- chunk_id: identifier of the chunk (int or str)
- src, dst: source and destination node IDs (typically GPUs)
- qpid: which QP to enqueue packets into at the source port
- rate: numeric value in bps, or "Max" (use link line rate)
- chunk_size_bytes: payload size of the chunk in bytes
- path: explicit forwarding path as a list of node IDs, starting with src and ending with dst
- time: trigger time
- dependency: Received chunks before trigger

### Policy semantics
- Each policy entry is installed at (chunk_id, src).
- A policy entry is triggered only when src fully owns the corresponding chunk.
- Multiple entries sharing the same (chunk_id, src) are triggered independently, enabling parallel fan-out and DAG-style execution.

## Execution model
- Event-driven simulation based on SimPy.
- Time advances only through discrete events such as packet transmission, propagation, arrival, and completion.
- All data transfers are packetized and forwarded hop by hop.

## Network modeling
- Explicit packetization with configurable payload size.
- Per-packet modeling includes:
  - serialization delay
  - propagation delay
  - store-and-forward switching behavior
- Switches forward packets at packet granularity without cut-through.

## Port and queue model
- Each directed link is modeled as a single-server output port.
- Ports maintain multiple QP (Queue Pair) queues.
- Scheduling across QPs:
  - round-robin arbitration
  - configurable quantum (number of packets per visit)
- Within a single QP, packets are strictly serialized.

## Modeling assumptions
- No congestion control: transport-layer congestion control mechanisms are not modeled.
- No packet loss: all packets are delivered reliably, without retransmissions.

These assumptions isolate the effects of CCL scheduling and data dependencies without confounding transport dynamics.

## Dependencies
- Python >= 3.11
- simpy
- networkx

Install dependencies:

    pip install simpy networkx

## Status
- Research-oriented simulator under active development.
- Packet loss / congestion control can be added later, but are currently outside the scope.