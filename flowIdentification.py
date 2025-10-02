import json
import csv
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import time
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx  # For OpenStreetMap background
from mpl_toolkits.basemap import Basemap
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict,Counter
from loaddata import find_flights_within_next_n_hours
from loaddata import rad_to_deg
from loaddata import douglas_peucker
from loaddata import compute_cluster_centers
from loaddata import find_cluster_connections
from loaddata import apply_dbscan
from loaddata import assign_cluster_colors
from loaddata import track_cluster_changes
from loaddata import fl2df
from loaddata import find_all_waypoint_connections
import data_harmonize
from data_harmonize import read_data_v2
from geopy.distance import geodesic
import networkx as nx
from networkx.algorithms.tree.branchings import minimum_spanning_arborescence
from scipy.cluster.hierarchy import dendrogram
import cluster_threshold
import matplotlib.dates as mdates



def generate_row_colors(n):
    cmap = plt.get_cmap('tab20')  # or 'tab10', 'Set1', etc.
    return [cmap(i % cmap.N) for i in range(n)]

def build_count_matrix_from_dict(data_dict):
    """
    Given a dictionary where keys are (i, j) tuples and values are lists,
    return a DataFrame where each (i, j) entry contains the length of the list.
    
    Parameters:
        data_dict (dict): Dictionary with keys as (i, j) and values as lists.
        
    Returns:
        pd.DataFrame: Matrix with list lengths, indexed by i and j values.
    """
    # Extract unique row and column indices
    row_keys = sorted(set(k[0] for k in data_dict))
    col_keys = sorted(set(k[1] for k in data_dict))

    # Create index mappings
    row_map = {val: idx for idx, val in enumerate(row_keys)}
    col_map = {val: idx for idx, val in enumerate(col_keys)}

    # Initialize matrix
    matrix = np.zeros((len(row_keys), len(col_keys)), dtype=int)

    # Fill matrix with list lengths
    for (i, j), v in data_dict.items():
        r_idx = row_map[i]
        c_idx = col_map[j]
        matrix[r_idx, c_idx] = len(v)

    # Return as DataFrame for readability
    return pd.DataFrame(matrix, index=row_keys, columns=col_keys)

def is_within_radius(center, point, radius_nm):
    """Check if a point is within radius_nm (nautical miles) of the center"""
    return geodesic(center, point).nautical <= radius_nm

def interpolate_to_boundary(point1, point2, center, radius_nm, steps=100):
    """
    Interpolate from point1 to point2 (both in (lon, lat)) until the distance from center exceeds radius_nm.
    
    Args:
        point1, point2: (lon, lat) tuples
        center: (lon, lat) tuple
        radius_nm: radius in nautical miles
        steps: number of interpolation steps
    
    Returns:
        (lon, lat) tuple representing approximate boundary intersection
    """
    lon1, lat1 = point1
    lon2, lat2 = point2
    center_lat, center_lon = center[1], center[0]  # convert to (lat, lon) for geopy

    for alpha in range(1, steps):
        frac = alpha / steps
        lon = lon1 + frac * (lon2 - lon1)
        lat = lat1 + frac * (lat2 - lat1)
        if geodesic((lat, lon), (center_lat, center_lon)).nautical >= radius_nm:
            return (lon, lat)

    return point2

def draw_tree(G, root):
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # top-down layout

    # Get node attributes
    node_weights = nx.get_node_attributes(G, 'weight')
    node_lambdas = nx.get_node_attributes(G, 'lambda')
    node_Ps = nx.get_node_attributes(G, 'P')
    node_flows = nx.get_node_attributes(G, 'flow')
    node_ends = nx.get_node_attributes(G, 'end')

    custom_labels = {}
    for node in G.nodes():
        lon, lat = node
        lon_str = f"{lon:.2f}"
        lat_str = f"{lat:.2f}"
        weight = node_weights.get(node, '?')
        lam = node_lambdas.get(node, '?')
        lam_str = f"{lam:.2f}nm" if isinstance(lam, float) else str(lam)
        P = node_Ps.get(node)
        P_str = f"\nP:{P:.1f}" if isinstance(P, (float, int)) else ""

        # Flags for flow and end
        F_str = " [F]" if node_flows.get(node) else ""
        E_str = " [E]" if node_ends.get(node) else ""

        label = f"({lon_str}, {lat_str})\nW:{weight}, λ:{lam_str}{P_str}{F_str}{E_str}"
        custom_labels[node] = label

    plt.figure(figsize=(10, 7))
    nx.draw(
        G, pos,
        labels=custom_labels,
        with_labels=True,
        arrows=True,
        node_size=1000,
        node_color="lightblue",
        font_size=10
    )
    plt.title(f"Flow Tree with Weight, Lambda, P-values, [F]=flow, [E]=end")
    plt.tight_layout()
    plt.show()

    
def simplify_by_weight(G, root):
    """
    Simplify the graph from `root` by collapsing all paths to the same end node,
    keeping only the shortest one by edge weight. Preserves node attributes like 'weight'.
    fligths have mutual origin-destination will be on the same branches because the shortest path is the same
    """
    G = G.copy()
    simplified = nx.DiGraph()
    
    # Ensure the root node is added with its attributes
    if root in G.nodes:
        simplified.add_node(root, **G.nodes[root])

    # Get all reachable nodes from root
    reachable = nx.descendants(G, root)

    for target in reachable:
        try:
            # Find the shortest path using edge weights
            shortest_path = nx.shortest_path(G, source=root, target=target, weight='weight')
            
            # Add all edges in that path
            simplified.add_edges_from(zip(shortest_path, shortest_path[1:]))

            # Copy node attributes for each node in the path
            for node in shortest_path:
                if node not in simplified.nodes:
                    simplified.add_node(node)
                for attr_key, attr_val in G.nodes[node].items():
                    simplified.nodes[node][attr_key] = attr_val

        except nx.NetworkXNoPath:
            continue

    return simplified


def compute_lambda_values(G, root, h):
    """
    Compute lambda values for nodes:
    - If weight ≥ h: lambda = distance to root
    - Else: inherit from nearest ancestor with weight ≥ h

    Assumes:
    - Nodes are (lon, lat)
    - 'weight' is a node attribute
    """
    lambda_values = {}
    root_pos = root  # node = (lon, lat)

    for node in nx.topological_sort(G):
        weight = G.nodes[node].get('weight', 0)

        if weight >= 0:
            # Compute geodesic distance to root
            lambda_val = geodesic((root_pos[1], root_pos[0]), (node[1], node[0])).nautical
        else:
            lambda_val = None
            # Traverse ancestors in BFS order for nearest logic
            for ancestor in nx.bfs_tree(G.reverse(), node):
                if ancestor == node:
                    continue
                anc_weight = G.nodes[ancestor].get('weight', 0)
                if anc_weight >= h:
                    lambda_val = lambda_values.get(ancestor)
                    if lambda_val is not None:
                        break  # use nearest one only

        lambda_values[node] = lambda_val
        G.nodes[node]['lambda'] = lambda_val

    return lambda_values
def mark_birth_end_nodes(G, h):
    """
    Marks 'birth' and 'end' nodes based on weight threshold h.

    - A node is marked as 'end' if:
        1. weight > h, and
        2. (it is a leaf) OR (none of its descendants have weight > h)
    - A node is marked as 'birth' if:
        1. it has more than one independent high-weight branch (> h)
           (i.e., high-weight descendants not directly connected in the same chain)
        2. such node is also marked as 'end'
    """
    for node in reversed(list(nx.topological_sort(G))):
        weight = G.nodes[node].get('weight', 0)
        descendants = nx.descendants(G, node)

        # Find all high-weight descendants
        high_desc = [d for d in descendants if G.nodes[d].get('weight', 0) >= h]

        # Filter high_desc to keep only the roots of high-weight branches
        independent_high_desc = []
        for d in high_desc:
            # Is d directly reachable from another high-weight descendant?
            parents = list(G.predecessors(d))
            if not any(p in high_desc for p in parents):
                independent_high_desc.append(d)

        is_leaf = G.out_degree(node) == 0

        # end condition
        if weight >= h and (is_leaf or len(high_desc) == 0):
            G.nodes[node]['end'] = True
        else:
            G.nodes[node]['end'] = False

        # birth condition: more than one independent high-weight branch
        G.nodes[node]['birth'] = len(independent_high_desc) > 1

        # If birth, it's also end
        if G.nodes[node]['birth']:
            G.nodes[node]['end'] = True


def compute_lambda_birth(G):
    """
    For each node, compute 'lambda_birth' = lambda of the nearest ancestor with mark 'end'.
    """
    lambda_birth = {}
    for node in G.nodes():
        lam_b = None
        for anc in nx.bfs_tree(G.reverse(), node):
            if anc == node:
                continue
            if G.nodes[anc].get('end'):
                lam_b = G.nodes[anc].get('lambda')
                break
        G.nodes[node]['lambda_birth'] = lam_b
        lambda_birth[node] = lam_b
    return lambda_birth

def compute_P_values(G, h):
    """
    For each 'end' node, find its nearest 'birth' ancestor.
    Then compute:
    P = sum over all leaf nodes under this subtree of:
        (lambda - lambda_birth) * weight
    """
    for node in reversed(list(nx.topological_sort(G))):
        if not G.nodes[node].get('end'):
            continue               

        if G.nodes[node].get('lambda_birth') is None:
            lambda_birth = 0
        else:
            lambda_birth = G.nodes[node].get('lambda_birth')
        total_P = 0
        lamnode = G.nodes[node].get('lambda')
        weight = G.nodes[node].get('weight', 0)
        if lamnode is not None and lambda_birth is not None:
            total_P = (lamnode - lambda_birth) * weight
            
        if not G.nodes[node].get('birth'):
            fullbranchP = 0
            for child in G.successors(node):
                grands = nx.descendants(G, child)
                grands.add(child)
                branchP = 0
                for leaf in grands:
                    # lam = geodesic((leaf[1], leaf[0]), (node[1], node[0])).nautical
                    weight = G.nodes[leaf].get('weight', 0)
                    lam = G.nodes[leaf].get('lambda')
                    if lam is not None:                     
                            branchP = max((lam-lamnode) * weight, branchP)
                fullbranchP = fullbranchP + branchP
            if total_P < fullbranchP:
                G.nodes[node]['flow'] = "cancel"
        G.nodes[node]['P'] = total_P
        # lambda


def adjust_P_hierarchically(G):
    """
    Adjust P values bottom-up in the tree:
    - If a node's P < sum of all its descendants' P, replace it
    - Else, remove all P values from its descendants
    """
    for node in reversed(list(nx.topological_sort(G))):  # bottom-up
        P_self = G.nodes[node].get('P')
        if P_self is None:
            continue  # skip nodes with no P

        # Collect descendant P values
        descendant_Ps = [
            G.nodes[d].get('P') for d in nx.descendants(G, node)
            if G.nodes[d].get('P') is not None
        ]
        total_desc_P = sum(descendant_Ps)

        if total_desc_P >= P_self:
            G.nodes[node]['P'] = total_desc_P
        else:
            # Remove all descendant P values
            for d in nx.descendants(G, node):
                if 'P' in G.nodes[d]:
                    del G.nodes[d]['P']

def mark_flow_clusters(G,md2c,mfc):
    """
    Mark 'flow' nodes:
    A node is marked as flow if it has a P value
    and none of its descendants have a P value.
    """
    for node in G.nodes():
        # if G.nodes[node].get('flow') == "cancel":
        #     G.nodes[node]['flow'] = False
        #     continue
        if 'P' not in G.nodes[node]:
            G.nodes[node]['flow'] = False
            continue
        if G.nodes[node]['lambda']<md2c:
            G.nodes[node]['flow'] = False
            continue
        if G.nodes[node]['weight']<mfc:
            G.nodes[node]['flow'] = False
            continue

        has_descendant_with_P = any(
            'P' in G.nodes[desc] for desc in nx.descendants(G, node)
        )

        G.nodes[node]['flow'] = not has_descendant_with_P
        
def map_nodes_to_clusters(G, cluster_centers):
    """
    For nodes with 'flow' = True or root, map their (lon, lat) to the cluster key.
    Stores 'cluster_name' attribute on the node.
    """
    pos_to_cluster = {v: k for k, v in cluster_centers.items()}

    for node in G.nodes():
        if node in pos_to_cluster:
            G.nodes[node]['cluster_name'] = pos_to_cluster[node]
        else:
            G.nodes[node]['cluster_name'] = None
def assign_flow_colors(G):
    """
    Assign a unique color to each flow node based on its cluster_name.
    Stores 'flow_color' as node attribute.
    """
    # Extract flow cluster names
    flow_clusters = {
        G.nodes[n]['cluster_name']
        for n in G.nodes()
        if G.nodes[n].get('flow') and G.nodes[n].get('cluster_name') is not None
    }

    sorted_keys = sorted(flow_clusters)
    cmap = cm.get_cmap('tab10', len(sorted_keys))  # can change colormap if needed
    cluster_to_color = {
        key: mcolors.to_hex(cmap(i)) for i, key in enumerate(sorted_keys)
    }

    for node in G.nodes():
        cluster_name = G.nodes[node].get('cluster_name')
        if G.nodes[node].get('flow') and cluster_name in cluster_to_color:
            G.nodes[node]['flow_color'] = cluster_to_color[cluster_name]
        else:
            G.nodes[node]['flow_color'] = "#cccccc"  # fallback gray
def build_weighted_digraph(edges):
    """
    Builds a directed graph from red_edges and assigns node weights based on
    the maximum of in-degree and out-degree (counted with repetition).

    Parameters:
        red_edges (list of tuples): Each tuple is (u, v, weight)

    Returns:
        networkx.DiGraph: A graph with edge weights and node attribute 'weight'
    """
    G = nx.DiGraph()
    in_counts = defaultdict(int)
    out_counts = defaultdict(int)

    # Add edges and count in/out degrees
    for u, v, weight in edges:
        if u!=v:
            G.add_edge(u, v, weight=weight)
            out_counts[u] += 1
            in_counts[v] += 1

    # Compute node weights
    node_weights = {
        node: max(in_counts[node], out_counts[node])
        for node in G.nodes()
    }

    # Assign as node attributes
    nx.set_node_attributes(G, node_weights, 'weight')

    return G 
def build_flow_tree_with_annotations(G, root_node, edges, mfc, mind2c, cluster_centers):
    """
    Build and annotate a flow tree rooted at `root_node` from graph `G`,
    applying hierarchical simplification, lambda-based scoring, and clustering.

    Parameters:
        G (nx.DiGraph): The input graph
        root_node (hashable): Root node to build the tree from
        mfc (float): Minimum flow capacity threshold (used in lambda and P computations)
        mind2c (float): Minimum distance to center (used in flow clustering)
        cluster_centers (dict): Mapping or coordinates for cluster centers

    Returns:
        nx.DiGraph: The annotated, simplified flow tree (mst)
    """
    mst = simplify_by_weight(G, root_node)
    mst = update_node_weights_from_edges(mst, edges)
    compute_lambda_values(mst, root_node, mfc)
    mark_birth_end_nodes(mst, mfc)
    compute_lambda_birth(mst)
    compute_P_values(mst, mfc)
    adjust_P_hierarchically(mst)
    mark_flow_clusters(mst, mind2c,mfc)
    map_nodes_to_clusters(mst, cluster_centers)
    assign_flow_colors(mst)  
     
    return mst  

def update_node_weights_from_edges(simplified, edges):
    """
    Set all node weights in `simplified` to 0. Then for each edge in `edges`, 
    if node u or v is on the shortest path between u and v in the simplified tree, 
    increment its weight by 1.

    Parameters:
        simplified (nx.DiGraph): Tree from simplify_by_weight
        G (nx.DiGraph): Original graph
        edges (list of tuples): List of (u, v), where u and v are OD node coordinates
    """

    # # Reset all weights in the simplified tree to 0
    # for node in simplified.nodes:
    #     simplified.nodes[node]['weight'] = 0

    # # Update weights based on shortest paths in simplified tree
    # for u, v in edges:
    #     if u in simplified.nodes and v in simplified.nodes:
    #         try:
    #             path = nx.shortest_path(simplified, source=u, target=v)
    #             for node in path:
    #                 simplified.nodes[node]['weight'] += 1
    #         except nx.NetworkXNoPath:
    #             continue
    # # Collect edges to remove: edges where the target node has zero weight
    # edges_to_remove = [(u, v) for u, v in simplified.edges if simplified.nodes[v]['weight'] == 0]
    # simplified.remove_edges_from(edges_to_remove)
    # # Step 4: Remove nodes with weight 0
    # nodes_to_remove = [node for node, data in simplified.nodes(data=True) if data['weight'] == 0]
    # simplified.remove_nodes_from(nodes_to_remove)
    return  simplified   
     
def applyonlineclustering(all_waypoints,previous_clusters,previous_labels,cluster_colors,dbscan_eps=0.01, dbscan_min_samples=3):
    new_clusters = {}
    if all_waypoints:
        # labels = apply_dbscan(all_waypoints, eps=dbscan_eps, min_samples=dbscan_min_samples)
        points = pd.DataFrame(all_waypoints, columns = ['lon', 'lat'])
        clustered_points = cluster_threshold.apply_dbscan(points)
        refined_clusters0 = cluster_threshold.split_by_reclustering(clustered_points)
        refined_clusters = cluster_threshold.reclustering(refined_clusters0)
        # Organize clusters as {label: set(points)}
        labels = refined_clusters['cluster']
        current_clusters = defaultdict(set)
        for i, label in enumerate(labels):
            if label != -1:  # Ignore noise
                current_clusters[label].add(all_waypoints[i])
        # Compute cluster centers
        # Track cluster changes with max overlapping
        matched_clusters, new_clusters, disappeared_clusters = track_cluster_changes(previous_clusters, current_clusters, previous_labels)
        print(f"New clusters formed: {new_clusters}")
        print(f"Clusters disappeared: {disappeared_clusters}")
        # Assign consistent colors
        cluster_colors = assign_cluster_colors(matched_clusters, new_clusters, cluster_colors)
        # Rename keys
        current_clusterscopy = defaultdict(set)
        for old_key, new_key in matched_clusters.items():
            if old_key in current_clusters:
                current_clusterscopy[new_key] = current_clusters.pop(old_key)  # Move values and remove old key
        
    return labels,current_clusterscopy, matched_clusters, cluster_colors

def merge_defaultdicts(d1, d2):
    merged = defaultdict(list)

    # Add from the first dictionary
    for k, v in d1.items():
        merged[k].extend(v)

    # Add from the second dictionary
    for k, v in d2.items():
        merged[k].extend(v)

    return merged

def merge_overlapping_dict_values(data,score,minfc):
    # Convert to list of (key, set of values)
    items = [(k, set(v)) for k, v in data.items()]
    scores = [(k, v) for k, v in score.items()]
    merged = []
    merged_score = []
    while items:
        base_k, base_set = items.pop(0)
        _, base_score = scores.pop(0)
        combined_keys = [base_k]
        combined_values = set(base_set)
        combined_score = base_score
        i = 0
        while i < len(items):
            k, v = items[i]
            _,scorei = scores[i]
            overlapflights = combined_values & v
            if len(overlapflights) > 0:  
                lon1, lat1, lon2, lat2 = zip(*combined_keys)                 
                start_point0 = lon1 + lat1
                end_point0 = lon2 + lat2
                d1 = geodesic(start_point0, end_point0).nautical
                lon1, lat1, lon2, lat2 = zip(*[k])                
                start_point = lon1 + lat1
                end_point = lon2 + lat2
                d2 = geodesic(start_point, end_point).nautical
                if end_point0 == end_point:
                    if base_score < scorei:
                        combined_keys = [k]
                        combined_values = set(v)   
                        combined_score =  scorei                  
                    items.pop(i)
                    scores.pop(i)
                    i  = i-1
                else:
                    if base_score >= scorei:
                        v = v - overlapflights
                        items[i] = (k,v)
                        scores[i] = (k, scorei*len(v)/(len(v)+len(overlapflights)))
                    else:
                        combined_values = combined_values - overlapflights
                        combined_score = combined_score*len(combined_values)/(len(combined_values)+len(overlapflights))
            i += 1  
        if len(combined_values)>=minfc:
            merged.append((combined_keys, list(combined_values),combined_score))
            # merged_score

    return merged

def extract_points_by_sequence(segment, lons, lats, cluster_center,flightcluster):
    """
    Extract all (lat, lon) points in lats/lons that lie between (lat1, lon1) and (lat2, lon2) by sequence.
    """
    lon1, lat1, lon2, lat2 = segment
    path = list(zip(lons, lats))
    
    start_point = (lon1, lat1)
    end_point = (lon2, lat2)
    keys = next((k for k, v in cluster_center.items() if v == start_point), None)
    keye = next((k for k, v in cluster_center.items() if v == end_point), None)
    if keys>-1 and keye>-1:
        try:
            start_idx = flightcluster.index(keys)   
        except ValueError:
            start_idx = -1
            return []
        try:
            end_idx = len(flightcluster) - 1 - flightcluster[::-1].index(keye)
        except ValueError:
            end_idx = -1
            return []        
    return path[start_idx:end_idx+1],start_idx,end_idx


def plot_traffic_t(flights_in_window, previous_clusters,previous_labels,cluster_colors, epsilon=0.1, mfc = 15, mind2c = 500,maxd2c = 3000):
    ########################################################################################################################process data     
    all_waypoints = []
    waypoint_indices = []  # Keep track of point indices for tracking clusters
    flight_clusters = defaultdict(list)  # Track cluster sequences for each flight
    flight_waypoints = defaultdict(list)  # Store waypoints per flight
    flow_of_flights_network = {}
    flowscore_network = {}
    ################################################################################################simplify traj and get all way points.
    for _, flight in flights_in_window.iterrows():
        waypoints = flight["waypoints"]
        waypoints = [wp for wp in waypoints if wp['z'] >= 6000]

        lon = rad_to_deg([wp["x"] for wp in waypoints])
        lat = rad_to_deg([wp["y"] for wp in waypoints])
        alt = [wp["z"] for wp in waypoints]  # Altitude in meters

        # Filter waypoints within the desired region
        filtered_coords = [(lo, la, wp["name"], wp["t"]) for lo, la, al, wp in zip(lon, lat, alt, waypoints) if -20 <= lo <= 30 and 26 <= la <= 66]
        # Apply Douglas-Peucker simplification
        if len(filtered_coords) > 2:
            simplified_coords = douglas_peucker([(lo, la) for lo, la, _, _ in filtered_coords], epsilon)
            simplified_waypoints = [wp for wp in filtered_coords if (wp[0], wp[1]) in simplified_coords]
        else:
            simplified_waypoints = filtered_coords.copy()

        # Store waypoints for clustering
        for lo, la, name, t in simplified_waypoints:
            all_waypoints.append((lo, la))
            waypoint_indices.append((flight.callsign, name))  # ✅ Track by (flight_id, waypoint_name)
            flight_waypoints[flight.callsign].append((lo, la, t))  

        if simplified_coords:
            simplified_lon, simplified_lat = zip(*simplified_coords)
            # m.plot(simplified_lon, simplified_lat, marker='.', linestyle='-', markersize=1, label=f"{flight['callsign']}", alpha=0.2) 
    ############################################################################################################################################
    
    if all_waypoints:
        # Apply DBSCAN clustering, input all_waypoints, output current_clusterscopy, clustercolors 
        ### labels: original cluster labels
        ### current_clusterscopy: updated cluster of original clustered results revised by previous clusters
        ### matched_clusters: x: y, x is the new label from the present clustering and y is the revised label after matching with previous ones
        labels,current_clusterscopy, matched_clusters, cluster_colors = applyonlineclustering(all_waypoints,previous_clusters,previous_labels,cluster_colors)
        # Compute cluster centers
        cluster_centers = compute_cluster_centers(current_clusterscopy)    
        # assign flight to clusters
        updated_labels = []
        for i, (point, label) in enumerate(zip(all_waypoints, labels)):
            updated_label = matched_clusters.get(label,label)  # ✅ Use updated label
            flight_id, waypoint_name = waypoint_indices[i]
            flight_clusters[flight_id].append(updated_label)
            updated_labels.append(updated_label)# ✅ Track cluster sequence per flight
        # Find connections based on flights transitioning between clusters
        connection_counts, flight_routes = find_all_waypoint_connections(flight_clusters)   
        ########### get cluster ranking here (write a function)
        adjmat = build_count_matrix_from_dict(flight_routes)
        for idx in adjmat.index:
            if idx in adjmat.columns:
                adjmat.loc[idx, idx] = 0
        row_sums = adjmat.sum(axis=1)
        # Sort rows by sum in descending order
        sorted_row_sums = row_sums.sort_values(ascending=False)
        sorted_row_sums = sorted_row_sums.drop(index=-1)

        ##########################################################################################################################################
        ########### for loop all clusters, untill a point with less than minimum flow number passing through
        for idx_f in sorted_row_sums.index:
            # if idx_f!=29:
            #     continue
            if sorted_row_sums[idx_f] < mfc:
                break
            red_edges = []        ##########form the root
            OD_list = []
            selected_flights = []                  
            for (cluster1, cluster2), flights in flight_routes.items():
                if cluster1 == idx_f:
                    flights = list(dict.fromkeys(flights))
                    new_flights = [f for f in flights if f not in selected_flights]
                    selected_flights.extend(new_flights)
                    subset_coords = []                    
                    for flight in new_flights:
                        if flight in flight_waypoints:                           
                            clusterid = flight_clusters[flight]  
                            # # Slice the list from that index onwards
                            start_index = clusterid.index(idx_f)
                            filtered_list = clusterid[start_index:]                                
                            subset_coords = [cluster_centers[k] for k in filtered_list if k != -1]
                            # Coordinates of reference cluster (cluster 21)
                            ref_coord = cluster_centers[idx_f]
                            # Loop through trajectory segments and color based on proximity
                            if len(subset_coords)<2:
                                continue
                            p0 = subset_coords[0]
                            plast = subset_coords[-1]
                            for i in range(len(subset_coords) - 1):
                                p1 = subset_coords[i]
                                p2 = subset_coords[i + 1]   
                                if p1 == p2:
                                    continue                                
                                # Convert distance to nautical miles
                                d1 = geodesic(p1, ref_coord).nautical
                                d2 = geodesic(p2, ref_coord).nautical
                                distance = geodesic(p1, p2).nautical  # edge weight
                                if d1 < maxd2c and d2 < maxd2c:
                                    # Add to red_edges for MST
                                    red_edges.append((p1, p2, distance))
                                    plast = p2
                                elif d1 < maxd2c:
                                    boundary_point = interpolate_to_boundary(p1, p2, ref_coord, maxd2c)
                                    distance = geodesic(p1, boundary_point).nautical
                                    red_edges.append((p1, boundary_point, distance))
                                    plast = boundary_point
                                color = 'red' if d1 < maxd2c or d2 < maxd2c else 'blue'                                  
                            pe = plast
                            if p0 !=pe:
                                OD_list.append ((p0,pe))

            #####build a graph from the red edges, input only need red edges
            #######################################################################build a graph from the red edges, only need red edges
            flow_of_flights = defaultdict(list)
            if red_edges:
                G = build_weighted_digraph(red_edges)
                ###########################################################################################################################
                # Step 2: Choose a root node 
                root_node = cluster_centers[idx_f]  # must exist in your graph                    
                ###########converting graph to tree, require inut of graph        
                mst = build_flow_tree_with_annotations(G, root_node, OD_list, mfc, mind2c, cluster_centers)
                # draw_tree(mst, root_node)
                # Step 4: for each Flow node, get all flights in it. Input require mst, selected flights, flight_waypoints, and flight_clusters
                ############################################################################################################################
                
                for flow_node in mst.nodes():
                    if mst.nodes[flow_node].get("flow"):
                        for flight in selected_flights:
                            if flight in flight_waypoints and flight in flight_clusters:
                                clusterid = flight_clusters[flight]                                
                                filtered_list = clusterid[:]
                                subset_coords = [cluster_centers[k] for k in filtered_list if k != -1]
                                # Check if any of the subset_coords match a flow node location
                                matched_flow_index = None
                                matched_root_index = None
                                for idx, coord in enumerate(subset_coords):
                                    if matched_flow_index is None and np.allclose(coord, flow_node, atol=1e-4):
                                        matched_flow_index = idx
                                    if matched_root_index is None and np.allclose(coord, root_node, atol=1e-4):
                                        matched_root_index = idx
                                    if matched_flow_index is not None and matched_root_index is not None:                                                
                                        if matched_flow_index>matched_root_index:
                                            lon, lat, _ = zip(*flight_waypoints[flight])
                                            flow_of_flights[flow_node].append(flight)
                                        break
                        if len(flow_of_flights[flow_node])<3:
                            1   
                
                flow_of_flights2 = {}
                flowscorego = {}
                for k, v in flow_of_flights.items():
                    aa = root_node + k
                    flow_of_flights2[aa] = flow_of_flights[k]
                    flowscorego[aa] = mst.nodes[k]['weight']*mst.nodes[k]['lambda']
                    if flowscorego[aa]<5000:
                        1
                flow_of_flights = flow_of_flights2.copy()
            flow_of_flightsgo = flow_of_flights.copy()
            
            flow_of_flights = defaultdict(list)
            red_edges = []        ##########form the root
            OD_list = []
            selected_flights = []                  
            for (cluster1, cluster2), flights in flight_routes.items():
                if cluster2 == idx_f:
                    flights = list(dict.fromkeys(flights))
                    new_flights = [f for f in flights if f not in selected_flights]
                    selected_flights.extend(new_flights)
                    subset_coords = []
                    for flight in new_flights:
                        if flight in flight_waypoints:                           
                            clusterid = flight_clusters[flight]  
                            clusterid = clusterid[::-1] 
                            # # Slice the list from that index onwards
                            start_index = clusterid.index(idx_f)
                            filtered_list = clusterid[start_index:]                                
                            subset_coords = [cluster_centers[k] for k in filtered_list if k != -1]
                            # Coordinates of reference cluster (cluster 21)
                            ref_coord = cluster_centers[idx_f]
                            # Loop through trajectory segments and color based on proximity
                            if len(subset_coords)<2:
                                continue
                            p0 = subset_coords[0]
                            plast = subset_coords[-1]                            
                            for i in range(len(subset_coords) - 1):
                                p1 = subset_coords[i]
                                p2 = subset_coords[i + 1]   
                                if p1 == p2:
                                    continue                                
                                # Convert distance to nautical miles
                                d1 = geodesic(p1, ref_coord).nautical
                                d2 = geodesic(p2, ref_coord).nautical
                                distance = geodesic(p1, p2).nautical  # edge weight
                                if d1 < maxd2c and d2 < maxd2c:
                                    # Add to red_edges for MST
                                    red_edges.append((p1, p2, distance))
                                    plast = p2
                                elif d1 < maxd2c:
                                    boundary_point = interpolate_to_boundary(p1, p2, ref_coord, maxd2c)
                                    distance = geodesic(p1, boundary_point).nautical
                                    red_edges.append((p1, boundary_point, distance))
                                    plast = boundary_point
                                color = 'red' if d1 < maxd2c or d2 < maxd2c else 'blue' 
                            pe = plast
                            if p0 !=pe:
                                OD_list.append ((p0,pe))

            #####build a graph from the red edges, input only need red edges
            #######################################################################build a graph from the red edges, only need red edges
            if red_edges:
                G = build_weighted_digraph(red_edges)
                ###########################################################################################################################
                # Step 2: Choose a root node 
                root_node = cluster_centers[idx_f]  # must exist in your graph                    
                ###########converting graph to tree, require inut of graph        
                mst = build_flow_tree_with_annotations(G, root_node, OD_list, mfc, mind2c, cluster_centers)
                # draw_tree(mst, root_node)
                # Step 4: for each Flow node, get all flights in it. Input require mst, selected flights, flight_waypoints, and flight_clusters
                ############################################################################################################################
                color_to_flights = defaultdict(list)
                flow_of_flights = defaultdict(list)
                for flow_node in mst.nodes():
                    if mst.nodes[flow_node].get("flow"):
                        for flight in selected_flights:
                            if flight in flight_waypoints and flight in flight_clusters:
                                clusterid = flight_clusters[flight]                                
                                filtered_list = clusterid[:]
                                subset_coords = [cluster_centers[k] for k in filtered_list if k != -1]
                                # Check if any of the subset_coords match a flow node location
                                matched_flow_index = None
                                matched_root_index = None
                                for idx, coord in enumerate(subset_coords):
                                    if matched_flow_index is None and np.allclose(coord, flow_node, atol=1e-4):
                                        matched_flow_index = idx
                                    if matched_root_index is None and np.allclose(coord, root_node, atol=1e-4):
                                        matched_root_index = idx
                                    if matched_flow_index is not None and matched_root_index is not None:                                                
                                        if matched_flow_index<matched_root_index:
                                            lon, lat, _ = zip(*flight_waypoints[flight])
                                            color_to_flights[color].append(flight)
                                            flow_of_flights[flow_node].append(flight)
                                        break
                        if len(flow_of_flights[flow_node])<3:
                            1    

                flow_of_flights2 = {}
                flowscorecome = {}
                for k, v in flow_of_flights.items():
                    aa = k+root_node
                    flow_of_flights2[aa] = flow_of_flights[k]
                    flowscorecome[aa] = mst.nodes[k]['weight']*mst.nodes[k]['lambda']
                    if flowscorecome[aa]<5000:
                        1
                flow_of_flights = flow_of_flights2.copy()
            
            flow_of_flightscome = flow_of_flights.copy()  
                      
            flow_of_flights_node = flow_of_flightsgo | flow_of_flightscome
            flowscore_node = flowscorego | flowscorecome
            flow_of_flights_network =  flow_of_flights_network | flow_of_flights_node
            flowscore_network = flowscore_network | flowscore_node    
            
                    
        flow_of_flights_network_merged = merge_overlapping_dict_values(flow_of_flights_network,flowscore_network,mfc)
        flow_timings = []
        row_colors = generate_row_colors(len(flow_of_flights_network_merged))
        figi = 0
        for (coord, flights,_),color in zip(flow_of_flights_network_merged,row_colors):
            for flight in flights:
                lon, lat, t = zip(*flight_waypoints[flight])
                path, ids, ide = extract_points_by_sequence(zip(*coord), lon, lat, cluster_centers,flight_clusters[flight])
        previous_clusters = current_clusterscopy.copy()
        previous_labels = matched_clusters.copy()

    return previous_clusters, previous_labels, cluster_colors, flow_of_flights_network_merged


if __name__ == "__main__":
    # df_flights = read_data_v2("DeepFlow_Data_v00.02/InitialFlow/20230714_NW_SW_Axis_InitialFlw.so6")
    # df_flights.to_pickle("my_dataframe.pkl")
    df_flights = pd.read_pickle("my_dataframe14.pkl")
    # # Extract only the date (year-month-day)
    date_only = df_flights["timebase"].dt.date
    # Find the most frequent date
    most_common_date = date_only.value_counts().idxmax()
    # Convert to datetime (with time = 00:00:00)
    most_common_datetime = pd.to_datetime(most_common_date)
    t0 = most_common_datetime #+timedelta(hours=11)

    previous_clusters = {}
    cluster_colors = {}
    previous_labels = {}  # Stores label assignments across frames
    window_hours = 24
    current_time = t0

    flights_in_window = find_flights_within_next_n_hours(df_flights,current_time,window_hours)
    previous_clusters, previous_labels, cluster_colors, flow = plot_traffic_t(flights_in_window,previous_clusters,previous_labels,cluster_colors)
    flowodt = []
    flow_length = []
    flow_counts = []
    for coord, flights, _ in flow:
        fodts = []    
        for flight in flights:
            matched_flight = flights_in_window[flights_in_window["callsign"] == flight]
            waypoints = matched_flight.iloc[0]["waypoints"]
            lon = rad_to_deg([wp["x"] for wp in waypoints])
            lat = rad_to_deg([wp["y"] for wp in waypoints])
            ts = [wp["t"] for wp in waypoints]
            all_coords = list(zip(lat, lon))  # geodesic expects (lat, lon)
            lon1, lat1, lon2, lat2 = zip(*coord)
            # Target coordinate pairs
            target1 = (lat1, lon1)
            target2 = (lat2, lon2)
            target1 = tuple(map(float, np.ravel(target1)))
            target2 = tuple(map(float, np.ravel(target2)))
            distflow = geodesic(target1, target2)

            # Find index with minimum geodesic distance to target1
            dist1_list = [geodesic(target1, pt).nautical for pt in all_coords]
            idx1 = int(np.argmin(dist1_list))

            # Find index with minimum geodesic distance to target2
            dist2_list = [geodesic(target2, pt).nautical for pt in all_coords]
            idx2 = int(np.argmin(dist2_list))
            info = {
                "FlightID": flight,
                "st": ts[idx1],
                "et": ts[idx2],
                "sid": idx1,
                "eid": idx2,
                "waypoints": waypoints
                }
            fodts.append(info)
        flow_length.append(distflow)
        flow_counts.append(len(flights))
        flowodt.append({
            "OD": coord,
            "info": fodts
            })
    df_flowodt = pd.DataFrame(flowodt)
    df_flowodt.to_pickle("df_flow.pkl")
    # allcounts = []
    # figi = 0
    # for _, flowi in df_flowodt.iterrows():
    #     1
    #     coord = flowi["OD"]
    #     infoi = flowi["info"]
    #     ets = [wp["et"] for wp in infoi]
    #     sts = [wp["st"] for wp in infoi]
    #     time_points = list(range(round(min(sts)), round(max(ets)) + 1, 60))
    #     counts = []
    #     for t in time_points:
    #         count = sum(st <= t <= et for st, et in zip(sts, ets))
    #         counts.append((t, count))
    #     allcounts.append(counts)
    #     ts, cs = zip(*counts)
    #     ts_datetime = pd.to_datetime(ts, unit='s')
    #     plt.figure(figsize=(4, 5))
    #     plt.plot(ts_datetime, cs, marker='o', linestyle='-')
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    #     plt.xlabel("Time",fontsize=12)
    #     plt.ylabel("#Flights in flow",fontsize=12)
    #     plt.grid(True)
    #     filenamefig = f"mainflowplots/count_{figi+1}.png"
    #     figi = figi+1
    #     plt.savefig(filenamefig, dpi=600, bbox_inches='tight')
    #     # plt.tight_layout()
    #     # plt.show()
    # allcounts[1]
    # counts = flow_counts
    # lengths = [d.km for d in flow_length]
    # total_length = sum(c * l for c, l in zip(counts, lengths))
    # total_count = sum(counts)

    # average_length = total_length / total_count

        
    
