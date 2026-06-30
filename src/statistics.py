import csv
import os
import math
import matplotlib.pyplot as plt

# ==========================================
# USER CONFIGURATIONS
# ==========================================
FILENAME = r"greedy-search_509K.csv"  # Target file to read
MAX_LINES = 1000      # Number of rows to read (excluding header). Set to None to read all.
# ==========================================

def load_points_from_csv(filename, max_lines):
    """
    Reads points from the CSV and reconstructs the tuple format.
    """
    points = []
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' could not be found.")
        return None

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        
        try:
            next(reader) # Skip the header
        except StopIteration:
            print("File is empty.")
            return points

        for i, row in enumerate(reader):
            if max_lines is not None and i >= max_lines:
                break
            
            try:
                # Extract numerator and denominator integers
                x_num, x_den = int(row[0]), int(row[1])
                y_num, y_den = int(row[2]), int(row[3])
                
                # Reconstruct: ((x_num, x_den), (y_num, y_den))
                points.append(((x_num, x_den), (y_num, y_den)))
            except (ValueError, IndexError):
                print(f"Warning: Skipping malformed row {i+2}: {row}")
                continue
                
    return points

def analyze_final_graph_pure_python(final_points):
    """
    Computes internal connections using pure Python exact arbitrary-precision 
    integer arithmetic to verify mathematical purity.
    """
    num_points = len(final_points)
    if num_points < 2:
        print("=== Final Graph Analysis ===")
        print(f"Total Points (Nodes): {num_points}\nTotal Connections (Edges): 0")
        print("Not enough points to analyze network connectivity.")
        return None

    print("Computing exact fractional distances (Pure Python)...")
    
    total_edges = 0
    unique_edges = []

    # Iterate through all unique pairs of points (combinations)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            p1 = final_points[i]
            p2 = final_points[j]

            s1, s2 = p1[0]
            n1, n2 = p1[1]

            u1, u2 = p2[0]
            v1, v2 = p2[1]

            # Cross-multiplication for dx = (s1/s2) - (u1/u2)
            x_num = s1 * u2 - u1 * s2
            x_den = s2 * u2

            # Cross-multiplication for dy = (n1/n2) - (v1/v2)
            y_num = n1 * v2 - v1 * n2
            y_den = n2 * v2

            # Exact integer check for unit distance: (dx)^2 + (dy)^2 == 1
            # (x_num^2 * y_den^2) + (y_num^2 * x_den^2) == (x_den^2 * y_den^2)
            lhs = (x_num ** 2) * (y_den ** 2) + (y_num ** 2) * (x_den ** 2)
            rhs = (x_den ** 2) * (y_den ** 2)

            if lhs == rhs:
                total_edges += 1
                unique_edges.append((i, j))
                
    avg_degree = (total_edges * 2) / num_points

    # Print Statistics
    print("\n==================================================")
    print("             FINAL GRAPH ANALYSIS                 ")
    print("==================================================")
    print(f"Total Points (Nodes)      : {num_points}")
    print(f"Total Connections (Edges) : {total_edges}")
    print(f"Average Node Connectivity : {avg_degree:.2f} connections/point")
    print("==================================================")

    N = num_points
    E = total_edges

    # Erdős unit distance scaling laws using the math module
    if N > 1 and E > 0:
        k = math.log(E) / math.log(N)
        print(f"k if E = N^k: k = {k:.6f}")

        # If E = c * N^(4/3)
        c = E / (N**(4/3))
        print(f"c if E = c * N^(4/3): c = {c:.6f}")

    return unique_edges

def plot_final_graph(final_points, unique_edges_matrix):
    """
    Renders the final unit-distance graph visually using matplotlib.
    """
    if unique_edges_matrix is None:
        print("Not enough points to plot a geometric network.")
        return

    x_coords = [p[0][0] / p[0][1] for p in final_points]
    y_coords = [p[1][0] / p[1][1] for p in final_points]

    plt.figure(figsize=(7, 7))

    for idx1, idx2 in unique_edges_matrix:
        plt.plot([x_coords[idx1], x_coords[idx2]],
                 [y_coords[idx1], y_coords[idx2]],
                 color='#34495e', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)

    plt.scatter(x_coords, y_coords, color='#e74c3c', edgecolors='#2c3e50', s=70, zorder=2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"{MAX_LINES}-node Unit-Distance Graph", fontsize=14, weight='bold', pad=15)
    #plt.grid(True, linestyle=':', alpha=0.6)
    #plt.xlabel("X Axis (Rational Space)", fontsize=10)
    #plt.ylabel("Y Axis (Rational Space)", fontsize=10)

    print("\nDisplaying interactive graph window...")
    plt.show()

if __name__ == "__main__":
    print("Using Processing Environment: Pure Python (Arbitrary Precision Integers)")
    print(f"Reading up to {MAX_LINES if MAX_LINES is not None else 'ALL'} lines from '{FILENAME}'...\n")

    # Load and process data
    discovered_points = load_points_from_csv(FILENAME, MAX_LINES)

    if discovered_points is not None:
        if len(discovered_points) == 0:
            print("No data rows found to process.")
        else:
            edge_matrix = analyze_final_graph_pure_python(discovered_points)
            plot_final_graph(discovered_points, edge_matrix)
