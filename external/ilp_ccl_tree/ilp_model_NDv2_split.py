import gurobipy as gp
print("Gurobi installed successfully!")
import math
from gurobipy import Model,GRB
from gurobipy import LinExpr
from gurobipy import quicksum

from itertools import product
from network_parameter import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tree_generate import *
from topo_generate import *
from results_process import *
import copy
# 启用 matplotlib 交互模式，保证在 VSCode Plot Viewer / 窗口中非阻塞显示
plt.ion()

# 允许事件循环刷新图像（非阻塞）
plt.pause(0.1)
# 创建模型
model = Model("ILP_Model")

N = 8
topo_demo = [[0]*N for _ in range(N)]
# 组 A 内完全连通（0..3）
for i in range(0,4):
    for j in range(0,4):
        if i != j:
            topo_demo[i][j] = 1
# 组 B 内完全连通（4..7）
for i in range(4,8):
    for j in range(4,8):
        if i != j:
            topo_demo[i][j] = 1
# 额外双向连接（如果你希望单向，把下面一行改成 topo_demo[1][4]=1 或 topo_demo[4][1]=1）
topo_demo[0][4] = 1
topo_demo[4][0] = 1
topo_demo[1][5] = 1
topo_demo[5][1] = 1
topo_demo[2][6] = 1
topo_demo[6][2] = 1
topo_demo[3][7] = 1
topo_demo[7][3] = 1

# 传输延迟矩阵 transfer_delay：
transfer_delay, capacity = generate_delay_and_capacity(topo_demo)
resolution= 1
chunk_size = (64e6) # KB.  1MB buffer
buffer_size = chunk_size* N
fastest_linkrate = 50e9 # GBps/s
fastest_link_epoch_duration = chunk_size/fastest_linkrate # s
fastest_link_epoch_duration = fastest_link_epoch_duration*resolution
transfer_delay_chunk_epoch_number = np.ceil(transfer_delay / fastest_link_epoch_duration)
propagation_delay_chunk_epoch_number = (chunk_size / capacity) / fastest_link_epoch_duration
fraction_split= 4

def update_transfer_delay_alpha_only(transfer_delay, capacity, no_link_value=500):
    import numpy as np
    td = np.array(transfer_delay, dtype=float)
    cap = np.array(capacity, dtype=float)
    mask_inf = np.isinf(td)
    # 有连接的地方直接相加
    td[~mask_inf] = td[~mask_inf] # + cap[~mask_inf]
    # 无连接的地方填no_link_value
    td[mask_inf] = no_link_value
    np.fill_diagonal(td, 0.0)
    return td

def update_transfer_delay_alpha_beta_all(transfer_delay, capacity, no_link_value=500):
    import numpy as np
    td = np.array(transfer_delay, dtype=float)
    cap = np.array(capacity, dtype=float)
    mask_inf = np.isinf(td)
    # 有连接的地方直接相加
    td[~mask_inf] = td[~mask_inf] + cap[~mask_inf]
    # 无连接的地方填no_link_value
    td[mask_inf] = no_link_value
    np.fill_diagonal(td, 0.0)
    return td
def update_transfer_delay_alpha_one_forth_beta_all(transfer_delay, capacity, no_link_value=500):
    import numpy as np
    td = np.array(transfer_delay, dtype=float)
    cap = np.array(capacity, dtype=float)
    mask_inf = np.isinf(td)
    # 有连接的地方直接相加
    td[~mask_inf] = td[~mask_inf] + np.ceil(1/fraction_split*cap[~mask_inf])
    # 无连接的地方填no_link_value
    td[mask_inf] = no_link_value
    np.fill_diagonal(td, 0.0)
    return td


# 调用
delay_alpha_only = update_transfer_delay_alpha_only(transfer_delay_chunk_epoch_number, propagation_delay_chunk_epoch_number)
delay_alpha_beta_all = update_transfer_delay_alpha_beta_all(transfer_delay_chunk_epoch_number, propagation_delay_chunk_epoch_number)
delay_alpha_1_4_beta_all = update_transfer_delay_alpha_one_forth_beta_all(transfer_delay_chunk_epoch_number, propagation_delay_chunk_epoch_number)



capacity = copy.deepcopy(1./propagation_delay_chunk_epoch_number) # chunk/s


node_number = np.shape(delay_alpha_only)[0]
s_range = range(0, node_number)  # 源节点范围
tree_number= 8
n_range = range(0, tree_number)  # 树编号范围
i_range = range(0, node_number) # 节点范围
j_range = range(0, node_number) # 节点范围
chunk_number = 1
c_range = range(0, chunk_number) # chunk 编号范围
K = 60
k_range = range(0, K )  # epoch 编号范围
d_range = range(0, node_number) # 需求节点范围 

# 生成并选取每个节点的候选树
all_trees, selected_trees = generate_and_select_trees(topo_demo, tree_number, enumeration_limit=2000, sample_tries=500, verbose=True, do_plot=False)

#复制树的逻辑
copy_number = fraction_split
all_trees_copied = {}
selected_trees_copied = {}
total_tree_number = tree_number * copy_number
for (s, idx), edges in selected_trees.items():
    for copy_idx in range(copy_number):
        new_idx = idx * copy_number + copy_idx
        if new_idx < total_tree_number:
            selected_trees_copied[(s, new_idx)] = edges.copy()
selected_trees = selected_trees_copied
tree_number = total_tree_number
n_range = range(0, tree_number)  # 树编号范围


tree_lib = {}
for (s, idx), edges in selected_trees.items():
    tree_lib[(s, idx)] = edges
# 调用绘图函数显示每个节点的所有候选树
plot_all_candidate_trees_new(tree_lib, s_range, figsize_per_tree=(4,4))
plt.pause(0.1) 
# 更新 T（字典 keyed by (j,i,s,n)）
T = {}
for s in s_range:
    for n in n_range:
        for j in j_range:
            for i in i_range:
                T[j, i, s, n] = 0  # 先全部置0
for (s, n), edge_list in tree_lib.items():
    for (i, j) in edge_list:
        T[i,j, s, n] = 1

# all gather 的形式
# D, meta_inc = generate_all_gather_single_chunk(topo_demo, include_self=True)
# print("含自发送 demand_inc.shape =", D.shape)
# print(D[:, :, 0])

D, meta_inc = generate_all_gather_uniform_chunks(topo_demo, chunk_number, include_self=True)
# D, meta_inc = generate_all_to_all_uniform_chunks(topo_demo, chunk_number, include_self=True)

print("含自发送 demand_inc.shape =", D.shape)
print(D[:, :, 0])
## 初始化变量（全部使用连续变量，允许 chunk 被拆分）
# r_(s,n,i,j,c,k): 在 epoch k, 沿边 i->j 发送的 chunk 份额（连续 0..1）
r = np.zeros((node_number, tree_number, node_number, node_number, chunk_number, K)).tolist()
for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range, c_range, k_range):
    r[s][n][i][j][c][k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                      name='route_frac_%d_%d_%d_%d_%d_%d' % (s,n, i, j, c, k))

# y_(s,n,c): chunk c 分配到树 n 的总份额（连续，0..1），允许拆分到多棵树
y = np.zeros((node_number, tree_number, chunk_number)).tolist()
for s, n, c in product(s_range, n_range, c_range):
    y[s][n][c] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"tree_frac_s{s}_n{n}_c{c}")

# B_(s,i,c,k): 节点 i 在 epoch k 开始时拥有来自 s 的 chunk c 的份额（连续 0..1）
B = np.zeros((node_number, node_number, chunk_number, K)).tolist()
for s, i, c, k in product(s_range, i_range, c_range, k_range):
    B[s][i][c][k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"buffer_{s}_{i}_{c}_{k}")

B_hold = np.zeros((node_number, node_number, chunk_number, K)).tolist()
for s, i, c, k in product(s_range, i_range, c_range, k_range):
    B_hold[s][i][c][k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"buffer_hold_{s}_{i}_{c}_{k}")

# R_(s,d,c,k): 节点 d 在 epoch k 结束时满足来自 s 的 chunk c 的份额（连续 0..1）
R = np.zeros((node_number, node_number, chunk_number, K)).tolist()
for s, d, c, k in product(s_range, d_range, c_range, k_range):
    R[s][d][c][k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"demand_satisfied_{s}_{d}_{c}_{k}")

# 定义二元指示变量 z，用于表示该 (s,n,i,j,c,k) 是否发送（z∈{0,1}）
z = np.zeros((node_number, tree_number, node_number, node_number, chunk_number, K)).tolist()
for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range, c_range, k_range):
    z[s][n][i][j][c][k] = model.addVar(vtype=GRB.BINARY,
                                       name=f"send_ind_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}")


# node constraints
# 初始条件：epoch 0 的 buffer
for s, i, c in product(s_range, i_range, c_range):
    model.addConstr(
        B[s][i][c][0] == (D[s, i, c] if i == s else 0.0),
        name=f"initB_epoch_0_{s}_{i}_{c}")
    model.addConstr(
        B_hold[s][i][c][0] == (D[s, i, c] if i == s else 0.0),
        name=f"initB_hold_epoch_0_{s}_{i}_{c}")

# node  constraints: 
# B_(s,i,c,k-1) + ∑ r_(s,n,i,j,c,k-⌈δ_ji⌉) = B_(s,i,c,k) ∀ c, ∀ s, ∀ i, k≠0, k<K, i≠s
# 收到的等于buffer
for s, c, i, k in product(s_range, c_range, i_range, range(1, K)):
    if i == s:
        continue
    summation = quicksum(
        r[s][n][j][i][c][k - int(math.ceil(delay_alpha_only[j, i])) ]
        for j, n in product(j_range, n_range)
        if T[j, i, s, n] == 1 and (k - int(math.ceil(delay_alpha_only[j, i])) ) >= 0
    )
    model.addConstr(
        B[s][i][c][k - 1] + summation == B[s][i][c][k],
        name=f"buffer_constraint_node_input_s{s}_i{i}_c{c}_k{k}"
    )

for s, c, i, k in product(s_range, c_range, i_range, range(1, K)):
    if i == s:
        continue
    summation = quicksum(
        r[s][n][j][i][c][k - int(math.ceil(delay_alpha_1_4_beta_all[j, i])) ]
        for j, n in product(j_range, n_range)
        if T[j, i, s, n] == 1 and (k - int(math.ceil(delay_alpha_1_4_beta_all[j, i])) ) >= 0
    )
    model.addConstr(
        B_hold[s][i][c][k - 1] + summation == B_hold[s][i][c][k],
        name=f"buffer_hold_constraint_node_input_s{s}_i{i}_c{c}_k{k}"
    )


# node constraints: 缓存必须至少覆盖在该 epoch 发出的份额（连续版本）
for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range, c_range, k_range):
    # 只对树边生效，非树边本来 r=0
    if T.get((i, j, s, n), 0) == 1:  
        model.addConstr(
            r[s][n][i][j][c][k] <= B_hold[s][i][c][k],
            name=f"send_gated_by_buffer_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}"
        )



# 到这里还没检查
# R 与 B 的关系（线性化 min）
for s, d, c, k in product(s_range, d_range, c_range, range(K-1)):
    if D[s][d][c] > 0:
        model.addConstr(R[s][d][c][k] == B_hold[s][d][c][k],
                        name=f"R_le_B_s{s}_d{d}_c{c}_k{k}")
    else:
        model.addConstr(R[s][d][c][k] == 0.0, name=f"R_zero_if_no_demand_s{s}_d{d}_c{c}_k{k}")


# 最后一 epoch 的 R 关系（同原模型）
k_last = K - 1
for s, d, c in product(s_range, d_range, c_range):
    terms = []
    for n, j in product(n_range, j_range):
        if T.get((j, d, s, n), 0) != 1:
            continue
        shift = int(math.ceil(delay_alpha_1_4_beta_all[j, d]))
        t_idx = k_last - shift
        if t_idx < 0:
            continue   # 忽略还没“到达”的流
        terms.append(r[s][n][j][d][c][t_idx])
    incoming_sum = quicksum(terms) if terms else 0.0
    model.addConstr(
        R[s][d][c][k_last] == B_hold[s][d][c][k_last] + incoming_sum,
        name=f"constraint3_final_s{s}_d{d}_c{c}_k{k_last}"
    )

for s, d, c in product(s_range, d_range, c_range):
    model.addConstr(R[s][d][c][K-1] == D[s, d, c], name=f"constraint4_final_2s{s}_d{d}_c{c}")

# tree 分配与 y 绑定：允许 chunk 拆分到多棵树，且在每棵树上同 epoch 的所有边发送份额相同
# 每个 chunk 的 tree 分配和为 1（允许拆分到多棵树）
for s, c in product(s_range, c_range):
    model.addConstr(quicksum(y[s][n][c] for n in n_range) == 1.0, name=f"one_tree_frac_per_chunk_s{s}_c{c}")
for s, n, c in product(s_range, n_range, c_range):
    model.addConstr(y[s][n][c] <=1/copy_number, name=f"y_min_0p1_s{s}_n{n}_c{c}")


# # 每个 chunk 的 tree 分配保持恒定比例

# # 用 z 线性化“r 要么为0 要么为 y”的逻辑：
# # 当 z==0 -> r == 0
# # 当 z==1 -> r == y
for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range, c_range, k_range):
    if T.get((i, j, s, n), 0) == 1:
        # 上界：r <= y
        model.addConstr(r[s][n][i][j][c][k] <= y[s][n][c],
                        name=f"r_le_y_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}")
        # 由 z 控制是否发送：若 z==0 则 r<=0 -> r==0；若 z==1 则允许达到 y
        model.addConstr(r[s][n][i][j][c][k] <= z[s][n][i][j][c][k],
                        name=f"r_le_z_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}")
        # 当 z==1 时强制 r >= y（结合上界可得 r==y）；当 z==0 该约束为 r >= y-1 不限制
        model.addConstr(r[s][n][i][j][c][k] >= y[s][n][c] - (1 - z[s][n][i][j][c][k]),
                        name=f"r_ge_y_minus_1mz_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}")
    else:
        # 非树边强制为0，并且指示变量也为0
        model.addConstr(r[s][n][i][j][c][k] == 0.0,
                        name=f"r_zero_non_treeedge_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}")
        model.addConstr(z[s][n][i][j][c][k] == 0,
                        name=f"z_zero_non_treeedge_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}")

for s, n, i, j, c in product(s_range, n_range, i_range, j_range, c_range):
    if T.get((i, j, s, n), 0) == 1:
        model.addConstr(quicksum(r[s][n][i][j][c][k] for k in k_range) == y[s][n][c],
                    name=f"r_once_per_edge_s{s}_n{n}_i{i}_j{j}_c{c}")

# 传输窗口/容量约束（使用按 epoch 的容量约束：sum_{s,n,c} r[s,n,i,j,c,k] <= capacity[i,j] * epoch_duration）
###
epoch_duration = fastest_link_epoch_duration  # 使用定义好的 epoch 时长（秒）
for i, j, k in product(i_range, j_range, k_range):
    try:
        cap_ij = float(capacity[i, j])
    except Exception:
        continue
    if cap_ij <= 0:
        continue
    beta_here = (np.ceil(1./capacity[i, j])).astype(int)# chunk/s
    expr = gp.LinExpr(0.0)
    for l in range (beta_here):
        for s, n, c in product(s_range, n_range, c_range):
            if (k - l) >= 0:
                expr.add(r[s][n][i][j][c][k-l])
    # capacity[i,j] 单位为 chunk/s，乘以 epoch_duration 得到每个 epoch 可传输的 chunk 数
    model.addConstr(expr <= 1,
                    name=f"cap_constr_link_{i}-{j}-{k}")
# ### 


# 优化目标
objective_opt = gp.LinExpr(0.0)
for s, d, c,k in product(s_range, d_range, c_range, k_range):
    if D[s, d, c] > 0:
        objective_opt.add(1./(k+1)*R[s][d][c][k], -1)

model.setObjective(objective_opt, GRB.MINIMIZE)

# 设置模型优化
import time
start_time = time.time()
print("开始模型优化...")
model.optimize()
end_time = time.time()
optimization_time = end_time - start_time
print(f"模型优化完成！耗时: {optimization_time:.2f} 秒")

# 输出结果
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    
    # 打印所有chunk的y值分配情况
    print("\n=== 所有Chunk的Tree分配情况 ===")
    for s in s_range:
        print(f"\n源节点 {s} 的所有chunk分配:")
        for c in c_range:
            print(f"  Chunk {c}:")
            for n in n_range:
                y_val = y[s][n][c].X
                if y_val > 0.01:  # 只显示有意义的分配
                    print(f"    Tree {n}: {y_val:.4f}")
    
    # 调用处理函数，传入所有参数确保处理所有chunk
    selected_by_source, selected_tree_lib = process_solution_and_plot(
        model=model,
        y=y,
        r=r,
        tree_lib=tree_lib,
        s_range=s_range,
        c_range=c_range,  # 确保传入完整的c_range
        n_range=n_range,
        K=K,
        delay=delay_alpha_beta_all,
        pos=None,
        show_vars=True,  # 改为True显示变量值
        do_plot=True,
        pause_time=0.1
    )

    # 提取和分析所有chunk的调度信息
    print("\n=== 所有Chunk的传输调度 ===")
    schedule = extract_transmission_schedule(r, s_range, c_range, n_range, K)
    for s in s_range:
        for c in c_range:
            print(f"\n源节点 {s} Chunk {c} 的传输调度:")
            if (s, c) in schedule:
                schedule_data = schedule[(s, c)]
                # 检查返回的数据格式
                if isinstance(schedule_data, dict):
                    # 如果是字典格式 {epoch: transmissions}
                    for epoch, transmissions in schedule_data.items():
                        if transmissions:
                            print(f"  Epoch {epoch}: {transmissions}")
                elif isinstance(schedule_data, list):
                    # 如果是列表格式 [(epoch, transmissions), ...]
                    for item in schedule_data:
                        if isinstance(item, tuple) and len(item) == 2:
                            epoch, transmissions = item
                            if transmissions:
                                print(f"  Epoch {epoch}: {transmissions}")
                        else:
                            print(f"  {item}")
                else:
                    # 其他格式，直接打印
                    print(f"  {schedule_data}")
            else:
                print("  无传输调度")

    # 分析每个chunk的完成时间
    print("\n=== 所有Chunk的完成时间 ===")
    completion = extract_completion_times(R, s_range, d_range, c_range, K)
    for s in s_range:
        for c in c_range:
            print(f"\n源节点 {s} Chunk {c} 的完成情况:")
            if (s, c) in completion:
                completion_data = completion[(s, c)]
                # 检查返回的数据格式
                if isinstance(completion_data, dict):
                    # 如果是字典格式 {dest: epoch}
                    for dest, epoch in completion_data.items():
                        if epoch < K:
                            print(f"  到达节点 {dest}: Epoch {epoch}")
                elif isinstance(completion_data, list):
                    # 如果是列表格式 [(dest, epoch), ...]
                    for item in completion_data:
                        if isinstance(item, tuple) and len(item) == 2:
                            dest, epoch = item
                            if epoch < K:
                                print(f"  到达节点 {dest}: Epoch {epoch}")
                        else:
                            print(f"  {item}")
                else:
                    # 其他格式，直接打印
                    print(f"  {completion_data}")
            else:
                print("  无完成信息")
    
    # 分析链路利用率（考虑所有chunk）
    link_schedule = analyze_link_utilization(r, s_range, c_range, n_range, K)
    
    # 找到所有chunk都满足需求的最早epoch
    first_all_satisfied_epoch = find_first_all_demand_satisfied_epoch(R, s_range, d_range, c_range, K)
    
    # 详细打印调度信息（包含所有chunk）
    propagation_delay_seconds = chunk_size / capacity
    transfer_delay_seconds = transfer_delay
    print_schedule_from_r(r, capacity, transfer_delay_seconds, propagation_delay_seconds, fastest_link_epoch_duration,
                          s_range, c_range, n_range, K,
                          capacity_is_chunks_per_epoch=True)
    
    print("最早所有demand都满足的epoch为:", first_all_satisfied_epoch)
    print("时间为:", (first_all_satisfied_epoch-1)*fastest_link_epoch_duration+0.7e-6)
    print(f"模型优化完成！耗时: {optimization_time:.2f} 秒")
    
    # 额外添加：打印每个chunk在每个epoch的buffer状态
    print("\n=== Buffer状态检查（前10个epoch）===")
    for s in list(s_range)[:min(2, len(list(s_range)))]:  # 只检查前两个源节点
        for c in list(c_range)[:min(2, len(list(c_range)))]:  # 只检查前两个chunk
            print(f"\n源 {s} Chunk {c} 的Buffer状态:")
            for k in range(min(10, K)):
                for i in i_range:
                    b_val = B[s][i][c][k].X
                    if b_val > 0.01:
                        print(f"  Epoch {k} 节点 {i}: B={b_val:.4f}")
elif model.status == GRB.INFEASIBLE:
    print("Model infeasible -> computing IIS ...")
    model.computeIIS()
    # 写出可用的文件类型（.ilp 是通用的）；部分 Gurobi 版本不识别 .iis 扩展
    try:
        model.write("model.ilp")
        print("Wrote model.ilp (includes IIS information).")
    except Exception as e:
        print("Failed to write model.ilp:", e)
    # 打印 IIS 中的约束名，便于定位冲突
    print("IIS constraints:")
    for c in model.getConstrs():
        if c.IISConstr:
            print("  ", c.constrName)
    # 可选：将模型写为 .mps 以便用其他工具检查（如需要）
    try:
        model.write("model_iis.mps")
        print("Also wrote model_iis.mps")
    except Exception:
        pass
else:
    print("No optimal solution found.")