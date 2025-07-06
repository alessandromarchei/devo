import torch
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def draw_patch_graph(ii_tensor, jj_tensor, kk_tensor, output_path="grouped_graph_debug.pdf"):
    # Convert to list of ints
    ii = [int(x) for x in ii_tensor]
    jj = [int(x) for x in jj_tensor]
    kk = [int(x) for x in kk_tensor]

    n_edges = len(ii)

    # Group patches per frame
    frame_to_patches = defaultdict(list)
    for j, k in zip(jj, kk):
        frame_to_patches[j].append(k)

    # Create grouped target nodes with debug labels
    grouped_fp_nodes = {}
    for j, patches in frame_to_patches.items():
        unique_sorted = sorted(set(patches))
        min_patch, max_patch = unique_sorted[0], unique_sorted[-1]
        patch_label = f"{min_patch}" if min_patch == max_patch else f"{min_patch}-{max_patch}"
        display_label = f"TARGET {j}\npatches {patch_label}"
        grouped_fp_nodes[j] = display_label

    # Create graph
    G = nx.Graph()

    # Create labeled keyframe nodes
    kf_nodes = sorted(set(ii))
    kf_labels = {i: f"KEYFRAME {i}\n(id {i})" for i in kf_nodes}

    G.add_nodes_from(kf_labels.values(), bipartite=0)
    G.add_nodes_from(grouped_fp_nodes.values(), bipartite=1)

    # Add edges
    for i, j in zip(ii, jj):
        u = kf_labels[i]
        v = grouped_fp_nodes[j]
        G.add_edge(u, v)

    # Layout positions
    pos = {}
    for idx, node in enumerate(sorted(n for n in G.nodes if n.startswith("KEYFRAME"))):
        pos[node] = (0, -idx * 1.5)
    for idx, node in enumerate(sorted(n for n in G.nodes if n.startswith("TARGET"))):
        pos[node] = (3, -idx * 1.5)

    # Draw graph
    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray',
            node_size=2400, font_size=9, font_weight='bold', width=1.2)

    # Add number of edges as footer
    plt.figtext(0.5, 0.01, f"Total edges: {n_edges}", ha='center', fontsize=12)

    plt.title("Grouped Bipartite Graph: Keyframes to Patch Sets (Debug View)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[INFO] Grouped graph with debug labels saved to: {output_path}")
