import numpy as np
import random
#from networkx.algorithms.tree.branchings import Edmonds
import random
import networkx as nx
from networkx import DiGraph
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for
import numpy as _np
from itertools import product as _product
import networkx as nx
from collections import deque


# 添加这一行以定义 `_nx`（修复 NameError）
import networkx as _nx

def count_arborescences(G, root):
    """
    用定向矩阵树（Matrix-Tree theorem for directed graphs）计算以 root 为根的生成树数量。
    适合先查看数量是否太大。
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    # W[i,j] = weight of edge i -> j
    W = _np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        W[idx[u], idx[v]] = w
    # 构造拉普拉斯 L： L[i,i] = sum_k w_{k,i}  (入度权和)
    #                 L[i,j] = - w_{j,i}  (i != j)
    L = _np.zeros((n, n), dtype=float)
    for i in range(n):
        L[i, i] = W[:, i].sum()
        for j in range(n):
            if i != j:
                L[i, j] = -W[j, i]
    r = idx[root]
    Lr = _np.delete(_np.delete(L, r, axis=0), r, axis=1)
    det = _np.linalg.det(Lr)
    return int(round(det))

def enumerate_arborescences_bruteforce(G, root, max_count=None):
    """
    暴力枚举所有以 root 为根的有向生成树（每个非根节点选择一条入边，检查是否为树）。
    返回 list(edge_list)，每个 edge_list 是 (u,v) 的列表。
    max_count 可限制返回数量。
    仅适合节点数很小或入度很小的图。
    """
    nodes = list(G.nodes())
    if root not in nodes:
        return []
    non_roots = [v for v in nodes if v != root]
    # 每个非根节点的候选入边来源
    incoming_lists = []
    for v in non_roots:
        preds = list(G.predecessors(v))
        if len(preds) == 0:
            return []  # 某节点没有入边，不可能有 arborescence
        incoming_lists.append(preds)
    results = []
    for choice in _product(*incoming_lists):
        edges = [(u, v) for u, v in zip(choice, non_roots)]
        H = nx.DiGraph()
        H.add_nodes_from(nodes)
        H.add_edges_from(edges)
        # 条件：边数为 n-1，DAG，并且 root 可到达所有节点
        if H.number_of_edges() != len(nodes) - 1:
            continue
        if not nx.is_directed_acyclic_graph(H):
            continue
        reachable = set(nx.descendants(H, root)) | {root}
        if len(reachable) == len(nodes):
            results.append(edges)
            if max_count and len(results) >= max_count:
                break
    return results

def enumerate_arborescences_backtrack(G, root, max_count=None):
    """
    回溯枚举，能早期检测环/不可达以剪枝（比暴力 product 更省内存）。
    返回 list(edge_list)。
    """
    nodes = list(G.nodes())
    if root not in nodes:
        return []
    non_roots = [v for v in nodes if v != root]
    incoming = {v: list(G.predecessors(v)) for v in non_roots}
    if any(len(lst) == 0 for lst in incoming.values()):
        return []
    results = []
    # 当前部分图用于快速检测环：用临时 DiGraph
    partial = nx.DiGraph()
    partial.add_nodes_from(nodes)

    def dfs_assign(idx):
        if max_count and len(results) >= max_count:
            return True
        if idx >= len(non_roots):
            # 检查可达性
            reachable = set(nx.descendants(partial, root)) | {root}
            if len(reachable) == len(nodes):
                results.append(list(partial.edges()))
            return False
        v = non_roots[idx]
        for u in incoming[v]:
            partial.add_edge(u, v)
            # 早期剪枝：如果形成有向环，则回退
            if nx.is_directed_acyclic_graph(partial):
                # 还可以做进一步剪枝：检查从 root 出发能否在剩余步骤覆盖所有尚未可达的节点
                # 简单版：继续递归
                stop = dfs_assign(idx + 1)
                if stop:
                    partial.remove_edge(u, v)
                    return True
            partial.remove_edge(u, v)
        return False

    dfs_assign(0)
    return results
def generate_unique_directed_trees(G, source, tree_number):
    """
    生成最多 tree_number 个不同的以 source 为根的有向生成树（arborescence）。
    使用两种随机化策略交替尝试：
      - 随机扰动边权 + single_source_dijkstra_path（最短路树）
      - 随机化邻居顺序的 BFS（随机 BFS 树）
    返回实际生成的树列表（每项为 edge_list），可能少于 tree_number。
    """
    trees = []
    used_edgesets = set()
    attempts = 0
    max_attempts = max(200, 100 * tree_number)

    # 确保边有 weight 属性（若没有）
    for u, v in G.edges():
        if 'weight' not in G[u][v]:
            G[u][v]['weight'] = 1.0

    while len(trees) < tree_number and attempts < max_attempts:
        edges = []
        # 交替使用两种策略以提高多样性
        if attempts % 2 == 0:
            # 随机扰动权重并用 Dijkstra 提取单源最短路径树
            G_tmp = G.copy()
            for u, v in G_tmp.edges():
                base = G[u][v].get('weight', 1.0)
                G_tmp[u][v]['weight'] = base + random.random() * 1.0
            try:
                paths = nx.single_source_dijkstra_path(G_tmp, source=source, weight='weight')
                for tgt, path in paths.items():
                    if len(path) < 2:
                        continue
                    for u, v in zip(path[:-1], path[1:]):
                        edges.append((u, v))
            except Exception:
                edges = []
        else:
            # 随机化邻居顺序的 BFS（生成随机 BFS 树）
            visited = set([source])
            queue = [source]
            while queue:
                u = queue.pop(0)
                neigh = list(G.successors(u))
                random.shuffle(neigh)
                for v in neigh:
                    if v not in visited:
                        visited.add(v)
                        edges.append((u, v))
                        queue.append(v)
            # 如果 BFS 未覆盖全部节点，则认为不是完整树，但仍可接受部分覆盖的候选树
        edge_set = tuple(sorted(set(edges)))
        if len(edge_set) == 0:
            attempts += 1
            continue
        if edge_set not in used_edgesets:
            used_edgesets.add(edge_set)
            trees.append(list(edge_set))
        attempts += 1
    return trees
def generate_grid_layout(n_nodes):
    """
    为任意数量的节点生成网格布局
    
    参数:
    - n_nodes: 节点数量
    
    返回:
    - pos: 字典 {node_id: (x,y)}
    """
    import math
    
    # 计算网格的行列数
    n_cols = math.ceil(math.sqrt(n_nodes))
    n_rows = math.ceil(n_nodes / n_cols)
    
    # 生成坐标
    pos = {}
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx < n_nodes:
                # 归一化坐标到 [0,1] 范围
                x = col / max(n_cols - 1, 1)
                y = 1 - row / max(n_rows - 1, 1)  # 上面是y=1
                pos[idx] = (x, y)
                idx += 1
    
    return pos
def plot_all_candidate_trees(tree_lib, s_range, figsize_per_tree=(4,4)):
    """
    每个源节点画出其所有候选树（每个树一个子图）。
    可支持每个 source 的候选树数量不同。
    """
    import matplotlib.pyplot as plt
    
    # 获取节点数量并生成网格布局
    n_nodes = max(s_range) + 1
    pos = generate_grid_layout(n_nodes)

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
                tree.add_nodes_from(range(n_nodes))  # 确保所有节点都在图中
                tree.add_edges_from(edge_list)
                nx.draw(tree, pos, with_labels=True, node_color='lightblue', arrows=True, ax=ax)
                ax.set_title(f"src {s} tree {n}")
            else:
                ax.axis('off')
    plt.tight_layout()
    plt.show()

def generate_and_select_trees(topo, select_n, enumeration_limit=2000, sample_tries=500, verbose=True, do_plot=False):
    """
    输入:
      topo: 二维 0/1 邻接矩阵（list 或 np.array），1 表示有向边 i->j
      select_n: 每个节点要选的树数量，int 或 dict{node: n}
      enumeration_limit: 若某根的树总数 <= 此值，尝试穷举所有生成树
      sample_tries: 随机采样上限（用于补齐）
      verbose: 是否打印过程信息
      do_plot: 是否调用 plot_all_candidate_trees 画图
    返回:
      all_trees: dict[(s, idx)] = edge_list   # 所有找到的候选树（按节点分组索引）
      selected: dict[(s, m)] = edge_list     # 对每个节点随机选出的 select_n 棵树（若不足则返回全部）
    行为:
      - 先把 topo 转为有向图并为每条边设置初始 weight=1.0
      - 对每个节点尝试用矩阵树计数决定是否穷举；穷举成功则返回所有穷举到的树
      - 若穷举不可行或数量太大，用随机化 Dijkstra / 随机 BFS 采样补齐
      - 最后从该节点的候选集合中随机选出指定数量返回（若候选数少则全部返回）
    """


    # 规范 select_n 为 dict
    if isinstance(select_n, int):
        want_per_node = {i: select_n for i in range(len(topo))}
    else:
        want_per_node = dict(select_n)

    # 建图（保证 weight 属性存在）
    G = _nx.DiGraph()
    N = len(topo)
    for i in range(N):
        for j in range(N):
            if int(topo[i][j]) == 1:
                G.add_edge(i, j, weight=1.0)

    all_trees = {}
    selected = {}

    for s in range(N):
        want = want_per_node.get(s, 0)
        if verbose:
            print(f"[tree gen] root={s}, want={want}")

        # 1) 计数（矩阵树）
        try:
            total_count = count_arborescences(G, s)
        except Exception as e:
            if verbose:
                print(f"  count_arborescences 异常: {e}")
            total_count = 0

        candidates = []

        # 2) 若数量合理则穷举（回溯枚举）
        if total_count > 0 and total_count <= enumeration_limit:
            if verbose:
                print(f"  total_count={total_count} <= {enumeration_limit}, 尝试穷举全部")
            try:
                arbs = enumerate_arborescences_backtrack(G, s, max_count=None)
                # 保证 edges 为标准化的有序 tuple/list 形式
                for e in arbs:
                    candidates.append(list(e))
            except Exception as e:
                if verbose:
                    print(f"  枚举出错: {e}")
                candidates = []

        # 3) 若穷举不足或不可穷举，使用采样补齐
        if len(candidates) < want:
            need = max(want - len(candidates), 0)
            if verbose:
                print(f"  采样补齐 need={need}（先用 generate_unique_directed_trees）")
            # 先用 generate_unique_directed_trees 尝试获取不同树（请求较多以提升命中）
            try:
                sampled = generate_unique_directed_trees(G, s, max(want * 5, need))
                for edges in sampled:
                    el = list(edges)
                    if el not in candidates:
                        candidates.append(el)
                    if len(candidates) >= want:
                        break
            except Exception as e:
                if verbose:
                    print(f"  generate_unique_directed_trees 异常: {e}")

            # 进一步用随机 BFS 补齐
            tries = 0
            while len(candidates) < want and tries < sample_tries:
                visited = set([s])
                queue = [s]
                edges = []
                while queue:
                    u = queue.pop(0)
                    neigh = list(G.successors(u))
                    random.shuffle(neigh)
                    for v in neigh:
                        if v not in visited:
                            visited.add(v)
                            edges.append((u, v))
                            queue.append(v)
                if len(edges) > 0 and list(edges) not in candidates:
                    candidates.append(list(edges))
                tries += 1
            if verbose and len(candidates) < want:
                print(f"  Warning: root {s} 仅获得 {len(candidates)} 个候选，目标 {want}")

        # 4) 兜底：若仍为空，用普通 BFS 生成一棵
        if len(candidates) == 0:
            if verbose:
                print(f"  兜底：使用普通 BFS 生成一棵树")
            bfs_edges = list(_nx.bfs_tree(G, source=s).edges())
            candidates.append(list(bfs_edges))

        # 存储所有候选（按源分组）
        # 将 candidates 转为按 source 分组的列表结构，键为 s，值为该 source 的候选树列表
        all_trees[s] = [list(edges) for edges in candidates]

        # 从 candidates 随机或顺序选 want 棵作为返回 selected（保持原接口：selected[(s,idx)] = edges）
        if want <= 0:
            selected_list = []
        elif len(candidates) <= want:
            selected_list = [list(c) for c in candidates]
        else:
            # 随机选择 want 个，无重复
            selected_idx = random.sample(range(len(candidates)), want)
            selected_list = [list(candidates[i]) for i in selected_idx]

        for idx, edges in enumerate(selected_list):
            selected[(s, idx)] = edges

        if verbose:
            print(f"  root {s} candidates={len(candidates)}, selected={len(selected_list)}")

    if do_plot:
        try:
            plot_all_candidate_trees(all_trees, range(N))
        except Exception as e:
            if verbose:
                print(f"  plot_all_candidate_trees 失败: {e}")

    # --- 插入：按深度排序并裁剪到 select_n ---
    from collections import deque

    def _compute_tree_depth(edges, root):
        """
        稳健地计算以 root 为根的树深度。
        - 支持 edges 为 [(parent, child), ...]（首选格式）。
        - 如果 edges 中含非二元元素（如整数），这些条目会被忽略而不会导致错误。
        - 若没有有效边，返回 0。
        """
        from collections import deque
        import networkx as _nx

        if not edges:
            return 0

        # 构建有向图，只添加二元边
        G = _nx.DiGraph()
        G.add_node(root)
        has_edge = False
        for e in edges:
            if isinstance(e, (tuple, list)) and len(e) == 2:
                p, c = e
                G.add_edge(p, c)
                has_edge = True
            else:
                # 忽略非二元条目（避免 unpack 错误）
                continue

        if not has_edge:
            return 0

        # BFS 计算从 root 到所有可达节点的最大距离（树深度）
        visited = {root}
        q = deque([(root, 0)])
        max_depth = 0
        while q:
            u, d = q.popleft()
            max_depth = max(max_depth, d)
            for v in G.successors(u):
                if v in visited:
                    continue
                visited.add(v)
                q.append((v, d + 1))

        return max_depth

    # 对每个源按深度升序排序，深度最小的排在前面；然后取前 select_n 棵（若不足则保留所有）
    for s, tree_list in list(all_trees.items()):
        tree_list.sort(key=lambda edges: _compute_tree_depth(edges, s))
        if len(tree_list) > select_n:
            del tree_list[select_n:]

    # 构建 selected_trees 映射 (s, idx) -> edges，保持向后兼容
    selected_trees = {}
    for s, trees in all_trees.items():
        for idx, edges in enumerate(trees):
            selected_trees[(s, idx)] = edges

    return all_trees,selected_trees


# ...existing code...
def plot_all_candidate_trees_new(tree_lib, s_range, figsize_per_tree=(4,4)):
    """
    在同一张大图上绘制所有 source 的候选树。
    布局：节点按深度分行（depth=0 为 source，depth=1 为下一层，依次类推）。
    子图网格：每一行对应一个 source，每一列对应该 source 的第 k 棵候选树（不足则空白）。
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from collections import deque
    import numpy as np

    def _normalize_edge_list(edge_list):
        edges = []
        if edge_list is None:
            return edges
        if isinstance(edge_list, dict):
            for p, children in edge_list.items():
                if children is None:
                    continue
                if isinstance(children, (list, tuple, set)):
                    for c in children:
                        try:
                            edges.append((int(p), int(c)))
                        except Exception:
                            continue
                else:
                    try:
                        edges.append((int(p), int(children)))
                    except Exception:
                        continue
            return edges
        if isinstance(edge_list, (list, tuple)):
            if all(isinstance(e, (list, tuple)) and len(e) == 2 for e in edge_list):
                for p, c in edge_list:
                    try:
                        edges.append((int(p), int(c)))
                    except Exception:
                        continue
                return edges
            if all(isinstance(e, (int, np.integer)) for e in edge_list) and len(edge_list) >= 2:
                for a, b in zip(edge_list, edge_list[1:]):
                    try:
                        edges.append((int(a), int(b)))
                    except Exception:
                        continue
                return edges
            for e in edge_list:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    try:
                        edges.append((int(e[0]), int(e[1])))
                    except Exception:
                        continue
        return edges

    # 收集每个 source 的树（按 idx 顺序可能不连续）
    per_source = {}
    max_trees = 0
    for (s, idx), edges in tree_lib.items():
        per_source.setdefault(s, []).append((idx, edges))
    for s in list(per_source.keys()):
        # 按 idx 排序，保证列顺序稳定
        per_source[s].sort(key=lambda x: x[0])
        max_trees = max(max_trees, len(per_source[s]))

    sources = list(s_range)
    if len(sources) == 0 or max_trees == 0:
        print("plot_all_candidate_trees_new: no trees to plot")
        return

    fig_w = figsize_per_tree[0] * max_trees
    fig_h = figsize_per_tree[1] * len(sources)
    fig, axes = plt.subplots(len(sources), max_trees, figsize=(fig_w, fig_h))
    # 统一处理 axes 形态为 2D array
    if len(sources) == 1 and max_trees == 1:
        axes = np.array([[axes]])
    elif len(sources) == 1:
        axes = np.array([axes])
    elif max_trees == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, s in enumerate(sources):
        trees = per_source.get(s, [])
        # map col -> edges (若某列缺失填 None)
        col_edges = {idx: edges for idx, edges in trees}
        for col in range(max_trees):
            ax = axes[row_idx, col]
            ax.cla()
            edges = col_edges.get(col, None)
            if not edges:
                ax.set_title(f"s={s} t={col} (empty)")
                ax.set_axis_off()
                continue

            G = nx.DiGraph()
            norm_edges = _normalize_edge_list(edges)
            for p, c in norm_edges:
                G.add_edge(p, c)
            if s not in G.nodes():
                G.add_node(s)

            # BFS from source to compute depth levels
            depth = {}
            q = deque([(s, 0)])
            depth[s] = 0
            while q:
                u, d = q.popleft()
                for v in G.successors(u):
                    if v in depth:
                        continue
                    depth[v] = d + 1
                    q.append((v, d + 1))
            maxd = max(depth.values()) if depth else 0
            level_nodes = {d: [] for d in range(0, maxd + 1)}
            for node, d in depth.items():
                level_nodes[d].append(node)

            pos = {}
            # 为每层分配均匀 x 坐标，y = -depth（行向下）
            for d in range(0, maxd + 1):
                nodes = sorted(level_nodes.get(d, []))
                m = len(nodes)
                if m == 0:
                    continue
                for i, node in enumerate(nodes):
                    x = (i + 1) / (m + 1)
                    y = -d
                    pos[node] = (x, y)

            # 剩余节点放到下一层
            other_nodes = [n_ for n_ in G.nodes() if n_ not in pos]
            if other_nodes:
                layer = maxd + 1
                nodes = sorted(other_nodes)
                m = len(nodes)
                for i, node in enumerate(nodes):
                    x = (i + 1) / (m + 1)
                    y = -layer
                    pos[node] = (x, y)

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=8)
            ax.set_title(f"s={s} t={col}")


def generate_and_select_trees_with_copies(topo, tree_number, copy_number=1, enumeration_limit=2000, sample_tries=500, verbose=True, do_plot=False):
    """
    基于原有的 generate_and_select_trees 函数，但每个生成的tree复制 copy_number 份
    
    参数:
        topo: 拓扑矩阵
        tree_number: 每个源节点需要的树数量（复制后的总数）
        copy_number: 每个原始树的复制份数
        其他参数同原函数
        
    返回:
        all_trees: 所有生成的树（包含复制）
        selected_trees: 选中的树（包含复制）
    """
    # 计算需要生成的原始树数量
    original_tree_number = tree_number // copy_number
    if tree_number % copy_number != 0:
        original_tree_number += 1  # 向上取整，确保有足够的树
    
    # 调用原始函数生成基础树
    original_all_trees, original_selected_trees = generate_and_select_trees(
        topo, original_tree_number, enumeration_limit, sample_tries, verbose, do_plot
    )
    
    # 复制树
    all_trees_with_copies = {}
    selected_trees_with_copies = {}
    
    # 处理 all_trees
    for (s, idx), edges in original_all_trees.items():
        for copy_idx in range(copy_number):
            new_idx = idx * copy_number + copy_idx
            all_trees_with_copies[(s, new_idx)] = edges.copy()
    
    # 处理 selected_trees，确保总数不超过 tree_number
    for (s, idx), edges in original_selected_trees.items():
        for copy_idx in range(copy_number):
            new_idx = idx * copy_number + copy_idx
            if new_idx < tree_number:  # 确保不超过需要的数量
                selected_trees_with_copies[(s, new_idx)] = edges.copy()
    
    if verbose:
        print(f"原始生成了 {len(original_selected_trees)} 个树")
        print(f"复制后共有 {len(selected_trees_with_copies)} 个树")
    
    return all_trees_with_copies, selected_trees_with_copies