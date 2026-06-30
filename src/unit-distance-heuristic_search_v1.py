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

NUM_ITERATIONS = 999
current_max_denom = 5
H = math.sqrt(2)

def simplify_fraction(num, den):
    """Simplifies a fraction and ensures the denominator is positive."""
    if den == 0: return None
    g = math.gcd(num, den)
    num //= g
    den //= g
    if den < 0:
        num, den = -num, -den
    return num, den

def run_pass_gpu(discovered_points, max_denominator, device, batch_size=5000):
    """
    Uses a Streaming Two-Pass approach to process candidates in bounded blocks.
    Protects against int64 arithmetic overflows using 5 large CRT prime moduli
    while matching your precise tie-breaker and fractional generation space.
    """
    existing_set = set(discovered_points)

    # 1. Calculate current circular boundary on CPU (Centered at origin)
    x_floats = [p[0][0] / p[0][1] for p in discovered_points]
    y_floats = [p[1][0] / p[1][1] for p in discovered_points]

    distances = [math.sqrt(xf**2 + yf**2) for xf, yf in zip(x_floats, y_floats)]
    R = max(distances) + H
    R_sq = R ** 2

    x_min, x_max = -R, R
    y_min, y_max = -R, R

    # 2. Pre-generate unique simplified fractions covering ALL denominators up to max_denominator
    unique_x = set()
    for x2 in range(1, max_denominator + 1):
        x1_start = math.floor(x_min * x2 - 1e-9)
        x1_end = math.ceil(x_max * x2 + 1e-9)
        unique_x.update(simplify_fraction(x1, x2) for x1 in range(x1_start, x1_end + 1))

    unique_y = set()
    for y2 in range(1, max_denominator + 1):
        y1_start = math.floor(y_min * y2 - 1e-9)
        y1_end = math.ceil(y_max * y2 + 1e-9)
        unique_y.update(simplify_fraction(y1, y2) for y1 in range(y1_start, y1_end + 1))

    # 3. Pre-load static target tensors to GPU memory once per iteration
    s_x = torch.tensor([p[0][0] / p[0][1] for p in discovered_points], dtype=torch.float64, device=device).unsqueeze(0)
    s_y = torch.tensor([p[1][0] / p[1][1] for p in discovered_points], dtype=torch.float64, device=device).unsqueeze(0)

    s1 = torch.tensor([p[0][0] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)
    s2 = torch.tensor([p[0][1] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)
    n1 = torch.tensor([p[1][0] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)
    n2 = torch.tensor([p[1][1] for p in discovered_points], dtype=torch.int64, device=device).unsqueeze(0)

    # 5 large 31-bit primes for Chinese Remainder Theorem modulus mapping (prevents overflow)
    primes = [2147483647, 2147483629, 2147483587, 2147483579, 2147483549]

    # 4. Stream and process candidate space dynamically
    best_candidate = None
    max_connections = 0
    max_denom_sum = -1

    product_iter = itertools.product(unique_x, unique_y)
    batch_candidates = []

    with torch.no_grad():
        while True:
            # Build the pipeline buffer up to the execution batch size
            while len(batch_candidates) < batch_size:
                try:
                    pair = next(product_iter)
                    if pair not in existing_set:
                        xf = pair[0][0] / pair[0][1]
                        yf = pair[1][0] / pair[1][1]
                        if (xf**2 + yf**2) <= R_sq:
                            batch_candidates.append(pair)
                except StopIteration:
                    break

            if not batch_candidates:
                break  # Complete run over candidate combinations completed

            # --- PASS 1: FLOAT64 FILTER ---
            c_x = torch.tensor([c[0][0] / c[0][1] for c in batch_candidates], dtype=torch.float64, device=device).unsqueeze(1)
            c_y = torch.tensor([c[1][0] / c[1][1] for c in batch_candidates], dtype=torch.float64, device=device).unsqueeze(1)

            dist_sq = (c_x - s_x)**2 + (c_y - s_y)**2
            float_unit_mask = torch.abs(dist_sq - 1.0) < 1e-7
            potential_connections = float_unit_mask.sum(dim=1)
            local_potential_indices = (potential_connections > 0).nonzero().flatten()

            if len(local_potential_indices) > 0:
                passed_indices = local_potential_indices.tolist()
                passed_cands = [batch_candidates[idx] for idx in passed_indices]

                # --- PASS 2: EXACT INTEGER MATH (Modulo Primes to guarantee zero overflow) ---
                d1_b = torch.tensor([c[0][0] for c in passed_cands], dtype=torch.int64, device=device).unsqueeze(1)
                d2_b = torch.tensor([c[0][1] for c in passed_cands], dtype=torch.int64, device=device).unsqueeze(1)
                u1_b = torch.tensor([c[1][0] for c in passed_cands], dtype=torch.int64, device=device).unsqueeze(1)
                u2_b = torch.tensor([c[1][1] for c in passed_cands], dtype=torch.int64, device=device).unsqueeze(1)

                exact_unit_mask = torch.ones((len(passed_cands), s1.shape[1]), dtype=torch.bool, device=device)

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

                exact_connections = exact_unit_mask.sum(dim=1).tolist()

                # --- ONLINE GLOBAL TIE-BREAKER LOGIC ---
                for local_idx, exact_conn in enumerate(exact_connections):
                    if exact_conn > 0:
                        if exact_conn > max_connections:
                            max_connections = exact_conn
                            best_candidate = passed_cands[local_idx]
                            max_denom_sum = best_candidate[0][1] + best_candidate[1][1]
                        elif exact_conn == max_connections:
                            cand = passed_cands[local_idx]
                            denom_sum = cand[0][1] + cand[1][1]
                            if denom_sum > max_denom_sum:
                                max_denom_sum = denom_sum
                                best_candidate = cand

            # Flush batch data variables to prevent VRAM leakages
            batch_candidates.clear()
            del c_x, c_y, dist_sq, float_unit_mask, potential_connections

    return best_candidate, max_connections


def analyze_final_graph(final_points, device):
    """Recomputes all internal connections among the discovered points accurately without integer overflow."""
    num_points = len(final_points)
    if num_points < 2:
        print("=== Final Graph Analysis ===")
        print(f"Total Points (Nodes): {num_points}\nTotal Connections (Edges): 0")
        return None

    s1 = torch.tensor([p[0][0] for p in final_points], dtype=torch.int64, device=device)
    s2 = torch.tensor([p[0][1] for p in final_points], dtype=torch.int64, device=device)
    n1 = torch.tensor([p[1][0] for p in final_points], dtype=torch.int64, device=device)
    n2 = torch.tensor([p[1][1] for p in final_points], dtype=torch.int64, device=device)

    s1_row, s1_col = s1.unsqueeze(1), s1.unsqueeze(0)
    s2_row, s2_col = s2.unsqueeze(1), s2.unsqueeze(0)
    n1_row, n1_col = n1.unsqueeze(1), n1.unsqueeze(0)
    n2_row, n2_col = n2.unsqueeze(1), n2.unsqueeze(0)

    primes = [2147483647, 2147483629, 2147483587, 2147483579, 2147483549]
    adjacency_matrix = torch.ones((num_points, num_points), dtype=torch.bool, device=device)

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

        adjacency_matrix &= (lhs_m == rhs_m)

    unique_edges_matrix = torch.triu(adjacency_matrix, diagonal=1)
    total_edges = unique_edges_matrix.sum().item()
    avg_degree = (total_edges * 2) / num_points

    print("==================================================")
    print("                 FINAL GRAPH ANALYSIS             ")
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
    edge_indices = unique_edges_matrix.cpu().nonzero()

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
        discovered_points = [((0, 1), (0, 1))]

    discovered_set = set(discovered_points)

    save_queue = Queue()
    writer_thread = Thread(target=async_file_writer, args=(save_queue, filename), daemon=True)
    writer_thread.start()

    if not file_exists:
        for seed in discovered_points:
            save_queue.put(seed)


    for iteration in range(1, NUM_ITERATIONS + 1):
        current_max_denom = 5

        new_point, connections = run_pass_gpu(discovered_points, current_max_denom, device)

        if new_point and connections > 0:
            discovered_points.append(new_point)
            save_queue.put(new_point)

            px, py = new_point
            print(f"{px[0]},{px[1]},{py[0]},{py[1]},{px[0] / px[1]},{py[0] / py[1]} ")
        else:
            print(f"Search concluded early at iteration {iteration}. No more valid steps found.")
            break

        torch.cuda.empty_cache()
        gc.collect()

    save_queue.put(None)
    writer_thread.join()

    print("\nAll points saved sequentially. Generating final graphs...")
    edge_matrix = analyze_final_graph(discovered_points, device)
    plot_final_graph(discovered_points, edge_matrix)