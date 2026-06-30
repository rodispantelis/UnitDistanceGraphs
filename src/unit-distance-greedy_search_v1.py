# -*- coding: utf-8 -*-

import os
# Adjust PyTorch allocator settings before importing torch to prevent VRAM fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import csv
from queue import Queue
from threading import Thread
import gc

# ---------------------------------------------------------
# USER DEFINED PARAMETERS
# ---------------------------------------------------------

NUM_ITERATIONS = 1000
TARGET_K = 1.33
H = 0.01

# Sequence of denominators composed only of 2 and 4k+1 primes

DENOMINATORS = [5, 25, 50, 65, 85, 100, 125, 130, 145, 170, 185, 200]


def simplify_fraction(num, den):
    """Simplifies a fraction and ensures the denominator is positive."""
    if den == 0: return None
    g = math.gcd(num, den)
    num //= g
    den //= g
    if den < 0:
        num, den = -num, -den
    return num, den

def run_pass_gpu(discovered_points, discovered_set, max_denominator, device, max_broadcast_elements=15_000_000):
    """
    Optimized Two-Pass approach using double-batching (both candidates and passes)
    and generator-like block processing to guarantee flat RAM/VRAM utilization profile.
    """
    num_existing = len(discovered_points)

    # 1. Calculate current circular boundary on CPU
    x_floats = [p[0][0] / p[0][1] for p in discovered_points]
    y_floats = [p[1][0] / p[1][1] for p in discovered_points]
    distances = [math.sqrt(xf**2 + yf**2) for xf, yf in zip(x_floats, y_floats)]

    R = max(distances) + H
    R_sq = R ** 2

    # Define the bounding box of the circle
    x_min, x_max = -R, R
    y_min, y_max = -R, R

    # 2. Pre-generate unique simplified fractions
    unique_x = set()
    x2 = max_denominator
    x1_start = math.floor(x_min * x2 - 1e-9)
    x1_end = math.ceil(x_max * x2 + 1e-9)
    unique_x.update(simplify_fraction(x1, x2) for x1 in range(x1_start, x1_end + 1))

    unique_y = set()
    y2 = max_denominator
    y1_start = math.floor(y_min * y2 - 1e-9)
    y1_end = math.ceil(y_max * y2 + 1e-9)
    unique_y.update(simplify_fraction(y1, y2) for y1 in range(y1_start, y1_end + 1))

    # Pre-allocate existing coordinates onto GPU (flat 1D arrays are lightweight)
    s_x = torch.tensor(x_floats, dtype=torch.float64, device=device).unsqueeze(0)
    s_y = torch.tensor(y_floats, dtype=torch.float64, device=device).unsqueeze(0)

    s1 = torch.tensor([p[0][0] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)
    s2 = torch.tensor([p[0][1] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)
    n1 = torch.tensor([p[1][0] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)
    n2 = torch.tensor([p[1][1] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)

    list_x = list(unique_x)
    list_y = list(unique_y)

    # Dynamic GPU safety batch limit (removes the hard-coded 1000 element floor)
    gpu_batch_size = max(1, max_broadcast_elements // max(1, num_existing))

    # Process Cartesian product in CPU chunks to prevent massive 'candidates' list overhead
    x_block_size = max(1, 5_000_000 // max(1, len(list_y)))

    primes = [2147483647, 2147483629, 2147483587, 2147483579, 2147483549]
    valid_batch_points = []
    global_max_connections = 0

    for xi in range(0, len(list_x), x_block_size):
        x_sub = list_x[xi:xi + x_block_size]

        candidates = []
        for pair in itertools.product(x_sub, list_y):
            if pair not in discovered_set:
                xf = pair[0][0] / pair[0][1]
                yf = pair[1][0] / pair[1][1]
                if (xf**2 + yf**2) <= R_sq:
                    candidates.append(pair)

        M_cands = len(candidates)
        if M_cands == 0:
            continue

        # 3. Process candidates in strict memory-bounded GPU chunks
        for i in range(0, M_cands, gpu_batch_size):
            end_i = min(i + gpu_batch_size, M_cands)
            batch_cands = candidates[i:end_i]

            # Allocate float tensors ONLY for the current micro-batch
            bx = torch.tensor([c[0][0] / c[0][1] for c in batch_cands], dtype=torch.float64, device=device).unsqueeze(1)
            by = torch.tensor([c[1][0] / c[1][1] for c in batch_cands], dtype=torch.float64, device=device).unsqueeze(1)

            dist_sq = (bx - s_x)**2 + (by - s_y)**2
            float_unit_mask = torch.abs(dist_sq - 1.0) < 1e-7
            potential_connections = float_unit_mask.sum(dim=1)

            local_potential_indices = (potential_connections > 0).nonzero().flatten()

            if len(local_potential_indices) > 0:
                passed_cands = [batch_cands[idx] for idx in local_potential_indices.tolist()]

                # 4. PASS 2 SUB-BATCHING: Prevents int64 matrix explosion in Modulo arithmetic
                pass2_batch_size = max(1, 5_000_000 // max(1, num_existing))

                for j in range(0, len(passed_cands), pass2_batch_size):
                    end_j = min(j + pass2_batch_size, len(passed_cands))
                    sub_passed = passed_cands[j:end_j]

                    d1_b = torch.tensor([c[0][0] for c in sub_passed], dtype=torch.int64, device=device).unsqueeze(1)
                    d2_b = torch.tensor([c[0][1] for c in sub_passed], dtype=torch.int64, device=device).unsqueeze(1)
                    u1_b = torch.tensor([c[1][0] for c in sub_passed], dtype=torch.int64, device=device).unsqueeze(1)
                    u2_b = torch.tensor([c[1][1] for c in sub_passed], dtype=torch.int64, device=device).unsqueeze(1)

                    exact_unit_mask = torch.ones((len(sub_passed), num_existing), dtype=torch.bool, device=device)

                    for mod_prime in primes:
                        d1_m, d2_m = d1_b % mod_prime, d2_b % mod_prime
                        u1_m, u2_m = u1_b % mod_prime, u2_b % mod_prime
                        s1_m, s2_m = s1 % mod_prime, s2 % mod_prime
                        n1_m, n2_m = n1 % mod_prime, n2 % mod_prime

                        x_num_m = (d1_m * s2_m - s1_m * d2_m) % mod_prime
                        x_den_m = (d2_m * s2_m) % mod_prime
                        y_num_m = (u1_m * n2_m - n1_m * u2_m) % mod_prime
                        y_den_m = (u2_m * n2_m) % mod_prime

                        x_num_sq = (x_num_m * x_num_m) % mod_prime
                        x_den_sq = (x_den_m * x_den_m) % mod_prime
                        y_num_sq = (y_num_m * y_num_m) % mod_prime
                        y_den_sq = (y_den_m * y_den_m) % mod_prime

                        lhs_m = ((x_num_sq * y_den_sq) % mod_prime + (y_num_sq * x_den_sq) % mod_prime) % mod_prime
                        rhs_m = (x_den_sq * y_den_sq) % mod_prime

                        exact_unit_mask &= (lhs_m == rhs_m)

                    connections_per_cand = exact_unit_mask.sum(dim=1)
                    valid_sub_indices = (connections_per_cand > 0).nonzero().flatten().tolist()

                    for idx in valid_sub_indices:
                        valid_batch_points.append(sub_passed[idx])

                    current_max = torch.max(connections_per_cand).item()
                    if current_max > global_max_connections:
                        global_max_connections = current_max

    return (valid_batch_points, global_max_connections) if valid_batch_points else (None, 0)


def analyze_final_graph(final_points, device):
    """Recomputes connections processing rows in batches to completely safe-guard VRAM."""
    num_points = len(final_points)
    if num_points < 2:
        print("=== Final Graph Analysis ===")
        print(f"Total Points (Nodes): {num_points}\nTotal Connections (Edges): 0")
        return None

    s1 = torch.tensor([p[0][0] for p in final_points], dtype=torch.int64, device=device)
    s2 = torch.tensor([p[0][1] for p in final_points], dtype=torch.int64, device=device)
    n1 = torch.tensor([p[1][0] for p in final_points], dtype=torch.int64, device=device)
    n2 = torch.tensor([p[1][1] for p in final_points], dtype=torch.int64, device=device)

    # Allocate matrix on CPU instead of GPU VRAM
    adjacency_matrix_cpu = torch.zeros((num_points, num_points), dtype=torch.bool, device='cpu')

    s1_col, s2_col = s1.unsqueeze(0), s2.unsqueeze(0)
    n1_col, n2_col = n1.unsqueeze(0), n2.unsqueeze(0)

    primes = [2147483647, 2147483629, 2147483587, 2147483579, 2147483549]
    row_batch_size = max(1, 15_000_000 // num_points)

    for i in range(0, num_points, row_batch_size):
        end_i = min(i + row_batch_size, num_points)

        s1_row = s1[i:end_i].unsqueeze(1)
        s2_row = s2[i:end_i].unsqueeze(1)
        n1_row = n1[i:end_i].unsqueeze(1)
        n2_row = n2[i:end_i].unsqueeze(1)

        chunk_mask = torch.ones((end_i - i, num_points), dtype=torch.bool, device=device)

        for mod_prime in primes:
            s1_r_m, s1_c_m = s1_row % mod_prime, s1_col % mod_prime
            s2_r_m, s2_c_m = s2_row % mod_prime, s2_col % mod_prime
            n1_r_m, n1_c_m = n1_row % mod_prime, n1_col % mod_prime
            n2_r_m, n2_c_m = n2_row % mod_prime, n2_col % mod_prime

            x_num_m = (s1_r_m * s2_c_m - s1_c_m * s2_r_m) % mod_prime
            x_den_m = (s2_r_m * s2_c_m) % mod_prime
            y_num_m = (n1_r_m * n2_c_m - n1_c_m * n2_r_m) % mod_prime
            y_den_m = (n2_r_m * n2_c_m) % mod_prime

            x_num_sq = (x_num_m * x_num_m) % mod_prime
            x_den_sq = (x_den_m * x_den_m) % mod_prime
            y_num_sq = (y_num_m * y_num_m) % mod_prime
            y_den_sq = (y_den_m * y_den_m) % mod_prime

            lhs_m = ((x_num_sq * y_den_sq) % mod_prime + (y_num_sq * x_den_sq) % mod_prime) % mod_prime
            rhs_m = (x_den_sq * y_den_sq) % mod_prime

            chunk_mask &= (lhs_m == rhs_m)

        adjacency_matrix_cpu[i:end_i] = chunk_mask.cpu()

    unique_edges_matrix = torch.triu(adjacency_matrix_cpu, diagonal=1)
    total_edges = unique_edges_matrix.sum().item()
    avg_degree = (total_edges * 2) / num_points

    print("==================================================")
    print("                FINAL GRAPH ANALYSIS              ")
    print("==================================================")
    print(f"Total Points (Nodes)      : {num_points}")
    print(f"Total Connections (Edges) : {total_edges}")
    print(f"Average Node Connectivity : {avg_degree:.2f} connections/point")
    print("==================================================")

    if num_points > 1 and total_edges > 0:
        k = np.log(total_edges) / np.log(num_points)
        print(f"k if E = N^k: k = {k:.4f}")
        c = total_edges / (num_points**(4/3))
        print(f"c if E = c * N^(4/3): c = {c:.4f}")

    return unique_edges_matrix


def plot_final_graph(final_points, unique_edges_matrix):
    """Renders the final unit-distance graph visually using matplotlib."""
    if unique_edges_matrix is None:
        print("Not enough points to plot a geometric network.")
        return

    x_coords = [p[0][0] / p[0][1] for p in final_points]
    y_coords = [p[1][0] / p[1][1] for p in final_points]

    plt.figure(figsize=(10, 10))
    edge_indices = unique_edges_matrix.nonzero()

    for edge in edge_indices:
        idx1, idx2 = edge[0].item(), edge[1].item()
        plt.plot([x_coords[idx1], x_coords[idx2]],
                 [y_coords[idx1], y_coords[idx2]],
                 color='#34495e', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)

    plt.scatter(x_coords, y_coords, color='#e74c3c', edgecolors='#2c3e50', s=70, zorder=2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Discovered Rational Unit-Distance Graph", fontsize=14, weight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel("X Axis (Rational Space)", fontsize=10)
    plt.ylabel("Y Axis (Rational Space)", fontsize=10)

    print("\nDisplaying interactive graph window...")
    plt.show()


def async_file_writer(queue, filename="live_points.csv"):
    """Background worker that writes points asynchronously."""
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    mode = 'a' if file_exists else 'w'

    with open(filename, mode=mode, newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["x_num", "x_den", "y_num", "y_den", "x_float", "y_float"])
            file.flush()

        while True:
            point = queue.get()
            if point is None:
                queue.task_done()
                break

            x_n, x_d = point[0]
            y_n, y_d = point[1]
            writer.writerow([x_n, x_d, y_n, y_d, x_n / x_d, y_n / y_d])
            file.flush()
            queue.task_done()


# --- Execution Loop ---

def compute_network_density(points, device):
    """
    Calculates the network density (k) efficiently without storing
    the full N x N boolean adjacency matrix in system memory.
    """
    num_points = len(points)
    if num_points <= 1:
        return 0.0, 0, num_points

    s1 = torch.tensor([p[0][0] for p in points], dtype=torch.int64, device=device)
    s2 = torch.tensor([p[0][1] for p in points], dtype=torch.int64, device=device)
    n1 = torch.tensor([p[1][0] for p in points], dtype=torch.int64, device=device)
    n2 = torch.tensor([p[1][1] for p in points], dtype=torch.int64, device=device)

    s1_col, s2_col = s1.unsqueeze(0), s2.unsqueeze(0)
    n1_col, n2_col = n1.unsqueeze(0), n2.unsqueeze(0)

    primes = [2147483647, 2147483629, 2147483587, 2147483579, 2147483549]
    row_batch_size = max(1, 15_000_000 // num_points)

    total_directed_edges = 0

    for i in range(0, num_points, row_batch_size):
        end_i = min(i + row_batch_size, num_points)

        s1_row = s1[i:end_i].unsqueeze(1)
        s2_row = s2[i:end_i].unsqueeze(1)
        n1_row = n1[i:end_i].unsqueeze(1)
        n2_row = n2[i:end_i].unsqueeze(1)

        chunk_mask = torch.ones((end_i - i, num_points), dtype=torch.bool, device=device)

        for mod_prime in primes:
            s1_r_m, s1_c_m = s1_row % mod_prime, s1_col % mod_prime
            s2_r_m, s2_c_m = s2_row % mod_prime, s2_col % mod_prime
            n1_r_m, n1_c_m = n1_row % mod_prime, n1_col % mod_prime
            n2_r_m, n2_c_m = n2_row % mod_prime, n2_col % mod_prime

            x_num_m = (s1_r_m * s2_c_m - s1_c_m * s2_r_m) % mod_prime
            x_den_m = (s2_r_m * s2_c_m) % mod_prime
            y_num_m = (n1_r_m * n2_c_m - n1_c_m * n2_r_m) % mod_prime
            y_den_m = (n2_r_m * n2_c_m) % mod_prime

            x_num_sq = (x_num_m * x_num_m) % mod_prime
            x_den_sq = (x_den_m * x_den_m) % mod_prime
            y_num_sq = (y_num_m * y_num_m) % mod_prime
            y_den_sq = (y_den_m * y_den_m) % mod_prime

            lhs_m = ((x_num_sq * y_den_sq) % mod_prime + (y_num_sq * x_den_sq) % mod_prime) % mod_prime
            rhs_m = (x_den_sq * y_den_sq) % mod_prime

            chunk_mask &= (lhs_m == rhs_m)

        # Sum truth values directly to save memory
        total_directed_edges += chunk_mask.sum().item()

    # Divide by 2 because the mask counts both (A->B) and (B->A).
    # Self-loops are 0 due to unit-distance math.
    total_edges = total_directed_edges // 2

    if total_edges == 0:
        return 0.0, 0, num_points

    k = math.log(total_edges) / math.log(num_points)
    return k, total_edges, num_points


# --- Execution Loop ---

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Processing Device: {device.type.upper()}\n")

    filename = "live_points.csv"
    discovered_points = []
    file_exists = os.path.exists(filename)

    if file_exists:
        print(f"Found existing data file '{filename}'. Loading points...")
        try:
            with open(filename, mode='r', newline='') as f:
                reader = csv.reader(f)
                for line_num, row in enumerate(reader, start=1):
                    if not row or any(char.isalpha() for char in row[0]):
                        continue
                    try:
                        pt = ((int(row[0]), int(row[1])), (int(row[2]), int(row[3])))
                        discovered_points.append(pt)
                    except (ValueError, IndexError):
                        continue
            print(f"Successfully loaded {len(discovered_points)} points from cache.")
        except Exception as e:
            print(f"Critical error reading {filename}: {e}. Starting fresh.")
            discovered_points = []

    if not discovered_points:
        print("No existing checkpoint found. Initializing with default seed points.")
        discovered_points = [((0, 1), (0, 1)), ((-1, 1), (0, 1))]

    discovered_set = set(discovered_points)

    save_queue = Queue()
    writer_thread = Thread(target=async_file_writer, args=(save_queue, filename), daemon=True)
    writer_thread.start()

    if not file_exists:
        for seed in discovered_points:
            save_queue.put(seed)


    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"\n{'='*50}")
        print(f" ITERATION {iteration:02d}")
        print(f"{'='*50}")

        denom_idx = 0
        goal_achieved = False

        while not goal_achieved:
            current_max_denom = DENOMINATORS[denom_idx]
            new_points, connections = run_pass_gpu(discovered_points, discovered_set, current_max_denom, device)

            points_added_this_pass = 0
            if new_points and connections > 0:
                for pt in new_points:
                    if pt not in discovered_set:
                        discovered_points.append(pt)
                        discovered_set.add(pt)
                        save_queue.put(pt)
                        points_added_this_pass += 1

            # Evaluate k dynamically
            k, total_edges, num_points = compute_network_density(discovered_points, device)

            print(f"\n[EVALUATION] Checking graph state with max_denom = {current_max_denom}:")
            print(f" -> Points Added This Pass : {points_added_this_pass}")
            print(f" -> Total Nodes (N)        : {num_points}")
            print(f" -> Total Edges (E)        : {total_edges}")
            print(f" -> Graph Density (k)      : {k:.4f}")

            if k >= TARGET_K:
                print(f"\n k ({k:.4f}) >= Target ({TARGET_K}).")
                goal_achieved = True
            else:
                denom_idx += 1
                print(f"\n[INCOMPLETE] Density is below target ({TARGET_K}).")

                if denom_idx >= len(DENOMINATORS):
                    print(f"\n[WARNING] Reached end of curated FRUITFUL_DENOMINATORS list.")
                    print("Forcing transition to next iteration to prevent memory overflow.")
                    goal_achieved = True
                else:
                    print(f"Expanding rational fraction space to max_denominator = {DENOMINATORS[denom_idx]}")

            torch.cuda.empty_cache()
            gc.collect()

    save_queue.put(None)
    writer_thread.join()