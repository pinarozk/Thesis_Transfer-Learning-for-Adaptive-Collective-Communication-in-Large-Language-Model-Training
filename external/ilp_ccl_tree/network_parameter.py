import numpy as np
import random
from networkx.algorithms.tree.branchings import Edmonds
import random
import networkx as nx
from networkx import DiGraph
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for



def plot_all_candidate_trees(tree_lib, s_range, figsize_per_tree=(4,4)):
    """
    每个源节点画出其所有候选树（每个树一个子图）。
    可支持每个 source 的候选树数量不同。
    """
    import matplotlib.pyplot as plt

    # 固定节点坐标（四节点布局）
    pos = {
        0: (0, 1),   # 左上
        1: (1, 1),   # 右上
        2: (0, 0),   # 左下
        3: (1, 0),   # 右下
    }

    # 统计每个源的树数量
    trees_per_source = {s: sorted([n for (ss, n) in tree_lib.keys() if ss == s]) for s in s_range}
    max_trees = max((len(v) for v in trees_per_source.values()), default=0)
    rows = len(s_range)
    cols = max_trees if max_trees > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_tree[0], rows * figsize_per_tree[1]))
    # 规范 axes 形状为二维列表
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i, s in enumerate(s_range):
        n_list = trees_per_source.get(s, [])
        for j in range(cols):
            ax = axes[i][j]
            if j < len(n_list):
                n = n_list[j]
                edge_list = tree_lib.get((s, n), [])
                tree = nx.DiGraph()
                tree.add_edges_from(edge_list)
                nx.draw(tree, pos, with_labels=True, node_color='lightblue', arrows=True, ax=ax)
                ax.set_title(f"src {s} tree {n}")
            else:
                ax.axis('off')
    plt.tight_layout()
    plt.show()

def generate_all_gather_single_chunk(topo,
                                     include_self: bool = True,
                                     dtype=np.int8):
    """
    生成形状 [S, S, 1] 的 All-Gather 需求张量。
    demand[s, d, 0] = 1 表示源 s 的那唯一 chunk 需要发给 d。

    参数:
        topo: 方阵 (S×S) 或可转为方阵的列表，仅用来取得 S
        include_self: True 时 demand[s, s, 0] = 1；False 时对角置 0
        dtype: 返回数组类型

    返回:
        demand: ndarray, shape = (S, S, 1)
        meta: dict 说明
    """
    topo_arr = np.asarray(topo)
    if topo_arr.ndim != 2 or topo_arr.shape[0] != topo_arr.shape[1]:
        raise ValueError("topo 必须是方阵")
    S = topo_arr.shape[0]

    demand = np.ones((S, S, 1), dtype=dtype)
    if not include_self:
        for s in range(S):
            demand[s, s, 0] = 0

    meta = {
        "S": S,
        "chunk_dim": 1,
        "include_self": include_self,
        "semantics": "demand[s,d,0]=1 表示源 s 的唯一 chunk 需要被 d 接收"
    }
    return demand, meta


def plot_trees(tree_lib, s_range, n=0, show_all_in_one=True, figsize=(16, 4)):
    """
    可视化每个源节点的第 n 棵树，并可选地将所有树画在一张大图上。
    参数:
        tree_lib: {(s, n): edge_list}
        s_range: 源节点编号范围
        n: 画第几棵树（默认0）
        show_all_in_one: 是否画所有树在一张大图
        figsize: 子图总大小
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # 固定节点坐标
    pos = {
        0: (0, 1),   # 左上
        1: (1, 1),   # 右上
        2: (0, 0),   # 左下
        3: (1, 0),   # 右下
    }

    # 画每个源节点的第 n 棵树
    fig, axes = plt.subplots(1, len(s_range), figsize=figsize)
    if len(s_range) == 1:
        axes = [axes]
    for idx, s in enumerate(s_range):
        edge_list = tree_lib.get((s, n), [])
        tree = nx.DiGraph()
        tree.add_edges_from(edge_list)
        ax = axes[idx]
        nx.draw(
            tree, pos, with_labels=True, node_color='lightblue', arrows=True, ax=ax
        )
        ax.set_title(f"Tree for source {s}, tree #{n}")
    plt.tight_layout()
    plt.show()

    # 画所有树在一张大图上
    if show_all_in_one:
        plt.figure(figsize=(6, 6))
        color_map = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        for idx, s in enumerate(s_range):
            edge_list = tree_lib.get((s, n), [])
            tree = nx.DiGraph()
            tree.add_edges_from(edge_list)
            nx.draw(
                tree, pos, with_labels=True, node_color='lightblue',
                arrows=True, edge_color=color_map[idx % len(color_map)], width=2, alpha=0.7
            )
        plt.title(f"All source trees (tree #{n}) in one plot")
        plt.show()



def generate_all_gather_single_chunk_with_switch(topo, include_self=True, exclude_nodes=None, chunk_number=1):
    """
    生成 all-gather 形式的 demand 矩阵 D 和一些元信息 meta_inc。
    设计要求：
      - 默认每个 source s 的 chunk 需要被所有节点接收（D[s,d,c]=1），
        include_self 控制是否包含 d==s 的情形（默认包含）。
      - 支持 exclude_nodes 列表：这些节点不会作为 source 也不会作为 destination（既不发也不收）。
      - 返回 D (N x N x chunk_number) 和 meta_inc（包含排除节点等信息）。
    参数：
      topo: 邻接矩阵（list 或 np.ndarray），仅用于确定节点数 N。
      include_self: 是否把 source 自己包含在需求内（默认 True）。
      exclude_nodes: 可选 list/iterable，指定不参与 all-gather 的节点索引（例如 switch）。
      chunk_number: chunk 数量（默认 1）。
    返回：
      D: numpy array，形状 (N, N, chunk_number)
      meta_inc: dict，包含 'exclude_nodes' 等信息
    """
    import numpy as np

    topo_arr = np.array(topo)
    if topo_arr.ndim != 2 or topo_arr.shape[0] != topo_arr.shape[1]:
        raise ValueError("topo must be a square matrix")
    N = topo_arr.shape[0]

    # 处理 exclude_nodes：如果未指定，尝试自动检测单个 switch（简单启发式）
    if exclude_nodes is None:
        indeg = topo_arr.sum(axis=0)
        outdeg = topo_arr.sum(axis=1)
        # NDv2 的 switch 在典型生成函数中会有 in_degree==2 && out_degree==2 且节点数为奇数
        candidates = [int(i) for i in range(N) if indeg[i] == 2 and outdeg[i] == 2]
        if len(candidates) == 1 and (N % 2 == 1):
            exclude_nodes = candidates
        else:
            exclude_nodes = []
    else:
        exclude_nodes = list(exclude_nodes)

    # 默认每个 source 的 chunk 要被所有节点接收（包括或不包括 self）
    D = np.ones((N, N, chunk_number), dtype=int)

    if not include_self:
        # 将对角线设为 0（source 自己不需要）
        for c in range(chunk_number):
            np.fill_diagonal(D[:, :, c], 0)

    # 将排除的节点既设为不发（对应 source 行全部置0），也设为不收（对应 destination 列全部置0）
    for ex in exclude_nodes:
        if 0 <= ex < N:
            D[ex, :, :] = 0    # ex 作为 source 不发
            D[:, ex, :] = 0    # ex 作为 destination 不收

    meta_inc = {
        "N": N,
        "chunk_number": chunk_number,
        "exclude_nodes": exclude_nodes,
        "include_self": include_self
    }

    return D, meta_inc


def generate_all_gather_uniform_chunks(topo, chunk_number, include_self=True, dtype=np.int8):
    """
    生成形状 [S, S, chunk_number] 的 All-Gather 需求张量。
    每个节点的 chunk 数量相同。

    参数:
        topo: 方阵 (S×S) 或可转为方阵的列表，仅用来取得 S
        chunk_number: 每个节点的 chunk 数量（相同）
        include_self: True 时 demand[s, s, c] = 1；False 时对角置 0
        dtype: 返回数组类型

    返回:
        demand: ndarray, shape = (S, S, chunk_number)
        meta: dict 说明
    """
    topo_arr = np.asarray(topo)
    if topo_arr.ndim != 2 or topo_arr.shape[0] != topo_arr.shape[1]:
        raise ValueError("topo 必须是方阵")
    S = topo_arr.shape[0]

    # 初始化需求张量
    demand = np.ones((S, S, chunk_number), dtype=dtype)

    if not include_self:
        # 将对角线设为 0（source 自己不需要）
        for c in range(chunk_number):
            np.fill_diagonal(demand[:, :, c], 0)

    meta = {
        "S": S,
        "chunk_number": chunk_number,
        "include_self": include_self,
        "semantics": "demand[s,d,c]=1 表示源 s 的第 c 个 chunk 需要被 d 接收"
    }

    return demand, meta

def generate_all_to_all_uniform_chunks(topo, chunk_number=None, include_self=True, dtype=np.int8):
    """
    生成形状 [S, S, S] 的 All-to-All 需求张量。
    每个节点的数据被均分成 S 份，生成 S 个 chunk，
    按顺序分配：第0份给node0，第1份给node1，第2份给node2，等等。
    需求矩阵中只包含1或0，表示是否有传输需求。

    参数:
        topo: 方阵 (S×S) 或可转为方阵的列表，仅用来取得 S
        chunk_number: 保留参数，实际使用 S 作为 chunk 数量
        include_self: True 时包含自发送（节点给自己发送）；False 时对角置 0
        dtype: 返回数组类型，建议使用整数类型

    返回:
        demand: ndarray, shape = (S, S, S)
        meta: dict 说明
    """
    topo_arr = np.asarray(topo)
    if topo_arr.ndim != 2 or topo_arr.shape[0] != topo_arr.shape[1]:
        raise ValueError("topo 必须是方阵")
    S = topo_arr.shape[0]

    # All-to-All: chunk数量等于节点数
    actual_chunk_number = S
    
    # 初始化需求张量 [S, S, S]
    demand = np.zeros((S, S, actual_chunk_number), dtype=dtype)

    # 填充需求矩阵
    for s in range(S):  # 源节点
        for c in range(actual_chunk_number):  # chunk 索引 (0 到 S-1)
            # 源节点 s 的第 c 个 chunk 发送给节点 c
            # 即：s的第0个chunk给node0，第1个chunk给node1，...
            d = c  # 目标节点就是 chunk 的索引
            if include_self or s != d:
                demand[s, d, c] = 1  # 直接设置为1，表示有传输需求

    if not include_self:
        # 将对角线设为 0（源节点不给自己发送）
        for c in range(actual_chunk_number):
            demand[c, c, c] = 0  # 源节点c的第c个chunk不发给自己

    meta = {
        "S": S,
        "chunk_number": actual_chunk_number,
        "chunk_fraction": f"1/{S} (隐含)",
        "include_self": include_self,
        "semantics": "demand[s,d,c]=1 表示源节点 s 的第 c 个 chunk 需要发送给节点 d。每个chunk大小为原始数据的1/S"
    }

    return demand, meta

# ===== 示例 =====
if __name__ == "__main__":
    topo = [
        [0,1,1],
        [1,0,1],
        [1,1,0]
    ]

    demand_inc, meta_inc = generate_all_gather_single_chunk(topo, include_self=True)
    print("含自发送 demand_inc.shape =", demand_inc.shape)
    print(demand_inc[:, :, 0])

    demand_exc, meta_exc = generate_all_gather_single_chunk(topo, include_self=False)
    print("不含自发送 demand_exc.shape =", demand_exc.shape)
    print(demand_exc[:, :, 0])

    chunk_number = 2
    demand_uniform, meta_uniform = generate_all_gather_uniform_chunks(topo, chunk_number)
    print(f"均匀分块 All-Gather ({chunk_number} 个块) 的需求张量形状 =", demand_uniform.shape)

    exclude_nodes = [1]
    demand_switch, meta_switch = generate_all_gather_single_chunk_with_switch(topo, exclude_nodes=exclude_nodes)
    print("排除节点的需求张量形状 =", demand_switch.shape)
    print("排除节点列表:", meta_switch["exclude_nodes"])

    # 测试 All-to-All 生成函数
    all_to_all_demand, all_to_all_meta = generate_all_to_all_uniform_chunks(topo, chunk_number)
    print("All-to-All 需求张量形状 =", all_to_all_demand.shape)
    print("All-to-All 元信息:", all_to_all_meta)

    # # 测试带有交换机排除功能的 All-to-All 生成函数
    # all_to_all_switch_demand, all_to_all_switch_meta = generate_all_to_all_with_switch(topo, chunk_number, exclude_nodes=exclude_nodes)
    # print("带交换机的 All-to-All 需求张量形状 =", all_to_all_switch_demand.shape)
    # print("带交换机的 All-to-All 元信息:", all_to_all_switch_meta)
