import matplotlib.pyplot as plt
import networkx as nx
from tree_generate import generate_grid_layout
from itertools import product
def extract_selected_trees(y_vars, tree_lib, s_range, c_range, n_range=None):
    """
    从 y 变量和 tree_lib 中提取每个 (s,c) 最终选中的 tree。
    支持连续松弛：显示所有有分配的树（不限于>0.5）
    返回:
      selected_by_source: dict[(s,c)] = [(n, fraction, edge_list), ...]  # 修改：支持多棵树
      selected_tree_lib: dict[(s, idx)] = edge_list  # 按 source 连续编号，便于绘图
    """
    selected_by_source = {}
    selected_tree_lib = {}
    
    for s in s_range:
        out_idx = 0
        for c in c_range:
            # 尝试从 y_vars 推断 n_range
            if n_range is None:
                try:
                    inferred_n = range(len(y_vars[s]))
                except Exception:
                    inferred_n = range(0)
            else:
                inferred_n = n_range
            
            # 收集所有有分配的树
            selected_trees = []
            for n in inferred_n:
                try:
                    val = y_vars[s][n][c].x
                except Exception:
                    val = 0
                if val is not None and float(val) > 0.001:  # 降低阈值
                    edges = tree_lib.get((s, n), [])
                    selected_trees.append((n, float(val), edges))
            
            # 按分配比例排序（降序）
            selected_trees.sort(key=lambda x: x[1], reverse=True)
            selected_by_source[(s, c)] = selected_trees
            
            # 为绘图选择主要的树（分配最多的）
            if selected_trees:
                selected_tree_lib[(s, out_idx)] = selected_trees[0][2]  # 选择分配最多的树的边
            else:
                selected_tree_lib[(s, out_idx)] = []
            out_idx += 1
    
    return selected_by_source, selected_tree_lib

def plot_selected_trees(selected_tree_lib, s_range, c_range, pos=None, figsize_per_tree=(4,4), block=False):
    """
    绘制 selected_tree_lib 中每个 source 的被选树。
    selected_tree_lib 格式: {(s, idx): edge_list}
    """
    if pos is None:
        pos = {0:(0,1), 1:(1,1), 2:(0,0), 3:(1,0)}
    
    rows = len(list(s_range))
    cols = max(1, len(list(c_range)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_tree[0], rows * figsize_per_tree[1]))
    
    # 规范 axes
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    
    for i, s in enumerate(s_range):
        for j, c in enumerate(c_range):
            ax = axes[i][j]
            edge_list = selected_tree_lib.get((s, j), [])
            ax.clear()
            if edge_list:
                Gtree = nx.DiGraph()
                Gtree.add_nodes_from(sorted(pos.keys()))
                Gtree.add_edges_from(edge_list)
                nx.draw(Gtree, pos, with_labels=True, node_color='lightblue', arrows=True, ax=ax)
                ax.set_title(f"src {s} chunk {c}")
            else:
                ax.set_title(f"src {s} chunk {c} no tree")
                ax.axis('off')
    
    plt.tight_layout()
    try:
        plt.show(block=block)
    except Exception:
        plt.draw()
    plt.pause(0.1)
    return fig

def process_solution_and_plot(model, y, r, tree_lib, s_range, c_range, n_range=None,
                            K=None, delay=None, pos=None, show_vars=False, do_plot=True, pause_time=0.1):
    """
    主入口：从 model/y/r/tree_lib 中提取选中树、打印可选信息并绘图。
    支持任意节点数量的网络和多chunk场景。
    """
    # 提取选中的树
    selected_by_source, selected_tree_lib = extract_selected_trees(y, tree_lib, s_range, c_range, n_range)
    
    # 打印所有chunk的选中树信息
    print("\n=== 所有Chunk的Tree分配情况 ===")
    for s in s_range:
        for c in c_range:
            selected_trees = selected_by_source.get((s, c), [])
            if selected_trees:
                print(f"source={s} chunk={c}:")
                for n, fraction, edges in selected_trees:
                    print(f"  tree={n}, fraction={fraction:.6f}, edges={edges}")
            else:
                print(f"source={s} chunk={c} -> no tree selected")
    
    # 打印总体统计
    total_chunks = len(s_range) * len(c_range)
    chunks_with_trees = sum(1 for (s, c) in selected_by_source if selected_by_source[(s, c)])
    print(f"\n总计: {chunks_with_trees}/{total_chunks} 个chunk有树分配")
    
    # 如果需要,打印详细的路由变量信息
    if show_vars and r is not None and K is not None:
        print("\n=== 前5个epoch的非零传输 (s,n,i->j,c,k) ===")
        count = 0
        for s in s_range:
            for c in c_range:
                for k in range(min(5, K)):  # 只显示前5个epoch
                    for n in (n_range if n_range is not None else range(len(r[s]))):
                        for i in range(len(r[s][n])):
                            for j in range(len(r[s][n][i])):
                                try:
                                    var = r[s][n][i][j][c][k]
                                    val = var.x
                                except Exception:
                                    val = 0
                                if val is not None and float(val) > 0.001:
                                    print(f"  s={s}, c={c}, k={k}: tree{n} {i}->{j} = {val:.6f}")
                                    count += 1
                                    if count > 20:  # 限制输出数量
                                        print("  ... (更多传输省略)")
                                        break
                            if count > 20:
                                break
                        if count > 20:
                            break
                    if count > 20:
                        break
                if count > 20:
                    break
            if count > 20:
                break
    
    # 绘图部分
    if do_plot:
        # 获取节点数量
        n_nodes = max(s_range) + 1
        
        # 如果没有提供pos,使用网格布局
        if pos is None:
            pos = generate_grid_layout(n_nodes)
        
        # 调整图形大小基于节点数量
        base_size = 4
        size_factor = max(1, n_nodes / 8)  # 8个节点时为标准大小
        figsize_per_tree = (base_size * size_factor, base_size * size_factor)
        
        # 绘制选中的树
        fig = plot_selected_trees(selected_tree_lib, 
                                s_range, 
                                c_range, 
                                pos=pos,
                                figsize_per_tree=figsize_per_tree,
                                block=False)
        
        plt.pause(pause_time)
    
    return selected_by_source, selected_tree_lib

def extract_transmission_schedule(r_vars, s_range, c_range, n_range, K, eps=1e-9):
    """
    从 r 变量中按 epoch 顺序提取每个 (source s, chunk c) 的传输记录并返回 schedule。
    - 支持 r 为连续值（fraction），记录所有 r > eps 的条目并保留 amount。
    返回:
      schedule: dict[(s,c)] = [ { 'epoch': k, 'tree': n, 'from': i, 'to': j, 'amount': val } , ... ]
    """
    schedule = {}
    for s in s_range:
        for c in c_range:
            entries = []
            for k in range(K):
                for n in n_range:
                    # 针对不规则内层长度，保护性取 len
                    try:
                        I = len(r_vars[s][n])
                    except Exception:
                        continue
                    for i in range(I):
                        try:
                            J = len(r_vars[s][n][i])
                        except Exception:
                            continue
                        for j in range(J):
                            try:
                                var = r_vars[s][n][i][j][c][k]
                            except Exception:
                                continue
                            v = getattr(var, "x", var)
                            try:
                                val = float(v)
                            except Exception:
                                continue
                            if val > eps:
                                entries.append({
                                    'epoch': int(k),
                                    'tree': int(n),
                                    'from': int(i),
                                    'to': int(j),
                                    'amount': float(val),
                                    'src': int(s),
                                    'chunk': int(c)
                                })
            # 按 epoch 排序（同 epoch 内保持原有顺序）
            entries.sort(key=lambda e: (e['epoch'], e['tree'], e['from'], e['to']))
            schedule[(s, c)] = entries

    # 打印可读输出（保留，但量化信息改为显示 fraction）
    print("\n=== Transmission schedule (per source, per chunk, fractional) ===")
    for s in s_range:
        for c in c_range:
            print(f"\nSource {s}, Chunk {c}:")
            entries = schedule.get((s, c), [])
            if not entries:
                print("  no transmissions")
                continue
            current_epoch = None
            for e in entries:
                if e['epoch'] != current_epoch:
                    current_epoch = e['epoch']
                    print(f"  epoch {current_epoch}:")
                print(f"    tree{e['tree']}: {e['from']} -> {e['to']}  frac={e['amount']:.6f}")
    return schedule

def extract_completion_times(R_vars, s_range, d_range, c_range, K, tol=1e-6):
    """
    提取每个 (s,d,c) 第一次满足 demand 的 epoch（当 R >= 1 - tol 时认为满足）。
    返回:
      completion: dict[(s,d,c)] = epoch_index (int) or None
    说明:
      - 对于连续 relaxation，R_vars 表示已满足的 fraction，通常当达到 ~1 时视为满足。
    """
    completion = {}
    for s in s_range:
        for c in c_range:
            for d in d_range:
                found = None
                for k in range(K):
                    try:
                        rv = R_vars[s][d][c][k]
                    except Exception:
                        rv = 0.0
                    v = getattr(rv, "x", rv)
                    try:
                        if float(v) >= 1.0 - tol:
                            found = int(k)
                            break
                    except Exception:
                        continue
                completion[(s, d, c)] = found

    # 打印可读结果
    print("\n=== Completion times (when each demand is effectively satisfied, R>=1-tol) ===")
    for s in s_range:
        for c in c_range:
            print(f"\nSource {s}, Chunk {c}:")
            for d in d_range:
                epoch = completion.get((s, d, c))
                if epoch is None:
                    print(f"  dest {d}: not satisfied within K")
                else:
                    print(f"  dest {d}: satisfied at epoch {epoch}")
    return completion

def analyze_link_utilization(r, s_range, c_range, n_range, K, eps=1e-9):
    """
    分析每条链路在每个 epoch 的具体传输情况（支持连续 fraction）。
    返回:
    link_schedule: dict[(i,j)] = { k: [ { 's':s, 'c':c, 'n':n, 'frac':val }, ... ], ... }
    说明:
    - 记录所有 frac > eps 的条目（包含 tree n）
    """
    link_schedule = {}

    # 假设节点索引可由 s_range 推断
    nodes = list(s_range)
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            link_schedule[(i, j)] = {}
            for k in range(K):
                flows = []
                for s, n, c in product(s_range, n_range, c_range):
                    try:
                        var = r[s][n][i][j][c][k]
                    except Exception:
                        continue
                    v = getattr(var, "x", var)
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    if val > eps:
                        flows.append({'s': int(s), 'c': int(c), 'n': int(n), 'frac': float(val)})
                if flows:
                    link_schedule[(i, j)][k] = flows

    # 打印可读的输出
    print("\n=== Link Utilization Schedule (fractional) ===")
    for (i, j), epochs in link_schedule.items():
        if not epochs:
            continue
        print(f"\nLink {i}->{j}:")
        for k in sorted(epochs.keys()):
            flows = epochs[k]
            flow_str = ", ".join([f"(s={f['s']},c={f['c']},n={f['n']},frac={f['frac']:.6f})" for f in flows])
            print(f"  Epoch {k}: {flow_str}")
    return link_schedule

def find_first_all_demand_satisfied_epoch(R, s_range, d_range, c_range, K, tol=1e-6):
    """
    返回最早所有 demand 都被满足的 epoch（即每个 (s,d,c) 首次达到 R >= 1-tol 的 epoch 的最大值）。
    若某个 demand 在 K 内未完全满足，记为 K。
    """
    first_satisfied = []
    for s in s_range:
        for d in d_range:
            for c in c_range:
                found = K
                for k in range(K):
                    try:
                        val = getattr(R[s][d][c][k], "x", R[s][d][c][k])
                        v = float(val)
                    except Exception:
                        v = 0.0
                    if v >= 1.0 - tol:
                        found = k
                        break
                first_satisfied.append(found)
    return max(first_satisfied)
def print_schedule(schedule, capacity, transfer_delay, propagation_delay, epoch_duration, capacity_is_chunks_per_epoch=False):
    """
    按时间顺序打印事件：
      - time = epoch * epoch_duration（epoch 开始时间）
      - Transmission delay 使用 propagation_delay[i][j]
      - Propagation delay 使用 transfer_delay[i][j]
    schedule: dict[(s,c)] -> [ {'epoch':k,'tree':n,'from':i,'to':j}, ... ]
    """
    events = []
    for (s, c), entries in schedule.items():
        for e in entries:
            epoch = int(e.get('epoch', 0))
            t = epoch * float(epoch_duration)
            events.append({
                'time': float(t),
                'epoch': epoch,
                'src': int(s),
                'chunk': int(c),
                'tree': e.get('tree', None),
                'from': int(e['from']),
                'to': int(e['to'])
            })
    events.sort(key=lambda x: (x['time'], x['from'], x['to']))
    for ev in events:
        i = ev['from']; j = ev['to']
        # Transmission delay = propagation_delay[i][j]
        try:
            trans = float(propagation_delay[i][j])
        except Exception:
            trans = 0.0
        # Propagation delay = transfer_delay[i][j]
        try:
            prop = float(transfer_delay[i][j])
        except Exception:
            prop = 0.0
        tree_label = f"{ev['tree']}" if ev['tree'] is not None else ""
        # 输出示例：In time 0.00000, chunk 0 2, send 0->1, Transmission delay is 0.00000, Propagation delay is 7.00e-07
        print(f"In time {ev['time']:.5f}, chunk {ev['chunk']} {tree_label}, send {i}->{j}, "
              f"Transmission delay is {trans:.5f}, Propagation delay is {prop:.2e}")

def print_schedule_from_r(r_vars, capacity, transfer_delay, propagation_delay, epoch_duration, s_range, c_range, n_range, K, capacity_is_chunks_per_epoch=False):
    """
    从 r 变量结构生成 schedule 并打印（封装）。
    参数顺序与之前保持一致：transfer_delay, propagation_delay 均以矩阵形式传入。
    """
    schedule = extract_transmission_schedule(r_vars, s_range, c_range, n_range, K)
    print_schedule(schedule, capacity, transfer_delay, propagation_delay, epoch_duration, capacity_is_chunks_per_epoch=capacity_is_chunks_per_epoch)


__all__ = ["extract_selected_trees", "plot_selected_trees", "process_solution_and_plot","extract_transmission_schedule","extract_completion_times","analyze_link_utilization","find_first_all_demand_satisfied_epoch","print_schedule_from_r"]
