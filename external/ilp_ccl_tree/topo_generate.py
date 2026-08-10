import numpy as np
def generate_delay_and_capacity(topo_demo):
    """
    根据给定的拓扑矩阵生成 transfer_delay 和 capacity 矩阵。
    - topo_demo: N x N，1表示有连接，0表示无连接，自己和自己无连接
    - transfer_delay: 有连接为2，无连接为30，自己到自己为0
    - capacity: 
        - 前4个节点之间或后4个节点之间有连接时为0.5
        - 前4个和后4个之间有连接时为0.2
        - 其他无连接为0
    """
    N = len(topo_demo)
    transfer_delay = np.full((N, N), 30, dtype=float)
    capacity = np.zeros((N, N), dtype=float)
    fastest_linkrate = 50e9 # GBps/s
    # 前4个节点和后4个节点的capacity矩阵
    cap_4 = np.array([
        [0, 25e9, 50e9, 25e9],
        [25e9, 0, 50e9, 50e9],
        [50e9, 50e9, 0, 25e9],
        [25e9, 50e9, 25e9, 0]
    ], dtype=float)
    # cap_4 = np.array([
    #     [0, 50e9, 50e9, 50e9],
    #     [50e9, 0, 50e9, 50e9],
    #     [50e9, 50e9, 0, 50e9],
    #     [50e9, 50e9, 50e9, 0]
    # ], dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                transfer_delay[i, j] = 0
                capacity[i, j] = 0
            elif topo_demo[i][j] == 1:
                # 特殊跨组连接
                if (i, j) == (0, 4) or (i, j) == (4, 0):
                    capacity[i, j] = fastest_linkrate
                elif (i, j) == (1, 5) or (i, j) == (5, 1):
                    capacity[i, j] = 25e9
                elif (i, j) == (2, 6) or (i, j) == (6, 2):
                    capacity[i, j] = 25e9
                elif (i, j) == (3, 7) or (i, j) == (7, 3):
                    capacity[i, j] = fastest_linkrate
                # 前4个节点内部
                elif i < 4 and j < 4:
                    capacity[i, j] = cap_4[i, j]
                # 后4个节点内部
                elif i >= 4 and j >= 4:
                    capacity[i, j] = cap_4[i-4, j-4]
                # 其他有连接
                transfer_delay[i, j] = 0.7e-6
            else:
                transfer_delay[i, j] = np.inf
                capacity[i, j] = 0
    return transfer_delay, capacity





def generate_ndv2_two_chassis_topo(single_chassis_topo):
    """
    将单 chassis 的 topo 扩展为 2 chassis + 1 switch 的拓扑。
    输入:
      single_chassis_topo: MxM 二值邻接矩阵 (list 或 np.ndarray)，表示单个 chassis 内部的连接（0/1），
                           自环应为0。
    输出:
      topo: NxN 二值邻接矩阵 (list)，N = 2*M + 1
      mapping: dict，键为 ('chassis', c, local_idx) 或 ('switch',), 值为全局节点索引
               例如 mapping[('chassis',0,0)] = 0, mapping[('chassis',1,0)] = M, mapping[('switch',)] = 2*M
    行为:
      - 保持每个 chassis 内部连接与 single_chassis_topo 相同（复制到两个 block）。
      - 插入一个 switch 节点，index = 2*M。
      - 添加连接: chassis0 node0 -> switch, chassis1 node0 -> switch
                 switch -> chassis0 node1, switch -> chassis1 node1
    """
    import numpy as np

    base = np.array(single_chassis_topo, dtype=int)
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("single_chassis_topo must be a square matrix")
    M = base.shape[0]
    N = 2 * M + 1
    topo = np.zeros((N, N), dtype=int)

    # 复制两个 chassis 的内部拓扑
    topo[0:M, 0:M] = base
    topo[M:2*M, M:2*M] = base

    # 插入 switch 节点（索引 sw = 2*M）
    sw = 2 * M

    # 从每个 chassis 的 node0 -> switch
    topo[0, sw] = 1            # chassis 0, local 0 -> switch
    topo[M + 0, sw] = 1        # chassis 1, local 0 -> switch

    # 从 switch -> 每个 chassis 的 node1
    topo[sw, 1] = 1            # switch -> chassis 0, local 1
    topo[sw, M + 1] = 1        # switch -> chassis 1, local 1

    # 确保自环为0（防御性）
    np.fill_diagonal(topo, 0)

    # 返回 list-of-lists 形式以兼容现有代码
    mapping = {}
    for c in (0, 1):
        for local in range(M):
            mapping[('chassis', c, local)] = c * M + local
    mapping[('switch',)] = sw

    return topo.tolist(), mapping

# ...existing code...
def generate_delay_and_capacity_two_chassis(single_transfer_delay, single_capacity,
                                           switch_delay=1.3e-6, switch_capacity=12.5e9):
    """
    将单 chassis 的 transfer_delay 和 capacity 扩展为 2 chassis + 1 switch 的矩阵。
    输入:
      single_transfer_delay: MxM numpy 数组，单 chassis 的传输延迟（无链路用 np.inf）
      single_capacity: MxM numpy 数组，单 chassis 的容量（无链路用 0）
      switch_delay: switch 相关链路的 delay（默认 1.3e-6）
      switch_capacity: switch 相关链路的 capacity（默认 12.5e9）
    输出:
      transfer_delay_2ch: NxN numpy 数组，N = 2*M + 1
      capacity_2ch: NxN numpy 数组
    行为:
      - 保持两个 chassis 内部的 delay/capacity 与输入一致（分别复制到两个 block）
      - 其它未连接的位置保持为 np.inf（delay）和 0（capacity）
      - 增加 switch（index = 2*M）与两个 chassis 的指定连接：
          chassis0 node0 -> switch
          chassis1 node0 -> switch
          switch -> chassis0 node1
          switch -> chassis1 node1
        上述四条链路使用 switch_delay 和 switch_capacity
    """
    import numpy as np

    td = np.array(single_transfer_delay, dtype=float)
    cap = np.array(single_capacity, dtype=float)
    if td.ndim != 2 or td.shape[0] != td.shape[1]:
        raise ValueError("single_transfer_delay must be a square matrix")
    if cap.shape != td.shape:
        raise ValueError("single_capacity must have same shape as single_transfer_delay")

    M = td.shape[0]
    N = 2 * M + 1
    td2 = np.full((N, N), np.inf, dtype=float)
    cap2 = np.zeros((N, N), dtype=float)

    # 复制两个 chassis 的内部矩阵
    td2[0:M, 0:M] = td
    cap2[0:M, 0:M] = cap
    td2[M:2*M, M:2*M] = td
    cap2[M:2*M, M:2*M] = cap

    # switch 索引
    sw = 2 * M

    # 添加 switch 相关的四条指定链路
    td2[0, sw] = switch_delay
    cap2[0, sw] = switch_capacity

    td2[M + 0, sw] = switch_delay
    cap2[M + 0, sw] = switch_capacity

    td2[sw, 1] = switch_delay
    cap2[sw, 1] = switch_capacity

    td2[sw, M + 1] = switch_delay
    cap2[sw, M + 1] = switch_capacity

    # 自环为 0 / 0
    np.fill_diagonal(td2, 0.0)
    np.fill_diagonal(cap2, 0.0)

    return td2, cap2
# ...existing code...