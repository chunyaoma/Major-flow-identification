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
from sklearn.cluster import DBSCAN,HDBSCAN,OPTICS
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict,Counter
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple,Union
import hdbscan
from sklearn.preprocessing import MinMaxScaler



# Load JSON data from file
def fl2df(pathname):
    with open(pathname, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Prepare list for DataFrame
    flights_data = []

    # Extract flight details
    for flight in data["flights"]:
        flight_id = flight["id"]
        callsign = flight["callsign"].strip()
        aircraft_type = flight["actType"].strip()
        cruise_speed = flight["cruiseSpeed"]
        cruise_altitude = flight["cruiseAltitude"]
        takeoff_mass = flight["takeOffMass"]
        takeoff_fuel = flight["takeOffFuel"]
        
        # Format waypoints into a list of dictionaries
        waypoints_list = [
            {
                "name": wp["name"].strip(),
                "x": wp["x"],
                "y": wp["y"],
                "z": wp["z"],
                "t": wp["t"]
            }
            for wp in flight["waypoints"]
        ]
        # Get timebase (first waypoint time) and endtime (last waypoint time)
        timebase = waypoints_list[0]["t"] if waypoints_list else None
        endtime = waypoints_list[-1]["t"] if waypoints_list else None
        
        # Append flight information with waypoints
        flights_data.append({
            "id": flight_id,
            "callsign": callsign,
            "actType": aircraft_type,
            "cruiseSpeed": cruise_speed,
            "cruiseAltitude": cruise_altitude,
            "takeOffMass": takeoff_mass,
            "takeOffFuel": takeoff_fuel,
            "waypoints": waypoints_list,  # Store waypoints as a structured list
            "timebase": timebase,
            "endtime": endtime
        })

    # Create DataFrame
    df_flights = pd.DataFrame(flights_data)
    # Convert timebase and endtime columns to datetime format
    df_flights["timebase"] = pd.to_datetime(df_flights["timebase"], unit='s')
    df_flights["endtime"] = pd.to_datetime(df_flights["endtime"], unit='s')
    
    return df_flights
# df_flights.head(2)
# print(df_flights.iloc[-1])



# Function to convert radians to degrees
def rad_to_deg(rad):
    return np.degrees(rad)



def has_restricted_waypoints(waypoints):
    """
    Returns True if a flight has at least two waypoints:
    - One with a prefix from {"EG", "EI"}.
    - Another with a prefix from {"LE", "LP", "GC"}.
    - Both waypoints must have coordinates in the region (longitude: [-20,30], latitude: [26,66]).
    """
    prefix_set_1 = {"EG", "EI"}
    prefix_set_2 = {"LE", "LP", "GC"}

    if len(waypoints) < 2:
        return False  # Not enough waypoints to check

    first_wp, last_wp = waypoints[0], waypoints[-1]  # First and last waypoint

    # Extract names and convert coordinates
    first_prefix, last_prefix = first_wp["name"][:2], last_wp["name"][:2]
    first_lon, first_lat = rad_to_deg(first_wp["x"]), rad_to_deg(first_wp["y"])
    last_lon, last_lat = rad_to_deg(last_wp["x"]), rad_to_deg(last_wp["y"])

    # Check if waypoints match name and coordinate criteria
    first_in_region = -20 <= first_lon <= 30 and 26 <= first_lat <= 66
    last_in_region = -20 <= last_lon <= 30 and 26 <= last_lat <= 66

    if first_prefix in prefix_set_1 and last_prefix in prefix_set_2 and first_in_region and last_in_region:
        return True  # Filter out this flight
    if first_prefix in prefix_set_2 and last_prefix in prefix_set_1 and first_in_region and last_in_region:
        return True  # Filter out this flight

    return True         # Should be False


# Function to find flights within the next 4 hours of a given time t
def find_flights_within_next_n_hours(df_flights,t,n):
    t = pd.to_datetime(t, unit='s')  # Convert t to datetime
    t_plus_4h = t + timedelta(hours=n)  # t + 4 hours

    # Filter flights that satisfy the condition:
    filtered_flights = df_flights[
        ((df_flights["timebase"] > t) | (df_flights["endtime"] > t)) & 
        (df_flights["timebase"] < t_plus_4h)
    ]
    
    return filtered_flights[filtered_flights["waypoints"].apply(has_restricted_waypoints)]


# t_sample = df_flights["timebase"].min()  # Using the minimum timebase as example
# filtered_flights_df = find_flights_within_next_4_hours(t_sample)


os.makedirs("plots", exist_ok=True)

# Douglas-Peucker Algorithm for trajectory simplification
def douglas_peucker(points, epsilon):
    """
    Simplifies a trajectory using the Douglas-Peucker algorithm.

    :param points: List of (lon, lat) tuples representing the trajectory.
    :param epsilon: Threshold distance for simplification (higher = more aggressive reduction).
    :return: Simplified list of (lon, lat) tuples.
    """
    if len(points) < 3:
        return points  # No need to simplify if only two points

    # Find the point with the maximum distance from the line segment connecting endpoints
    start, end = points[0], points[-1]
    max_dist = 0
    index = 0

    for i in range(1, len(points) - 1):
        dist = np.abs(np.cross(np.array(end) - np.array(start), np.array(points[i]) - np.array(start))) / euclidean(start, end)
        if dist > max_dist:
            max_dist = dist
            index = i

    # If the max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        left = douglas_peucker(points[:index+1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return left[:-1] + right  # Merge results, removing duplicate middle point

    return [start, end]  # Only keep endpoints if simplification is sufficient

# Function to calculate cluster centers
def compute_cluster_centers(cluster_dict):
    cluster_centers = {}
    for label, points in cluster_dict.items():
        if points:
            lons, lats = zip(*points)
            cluster_centers[label] = (np.mean(lons), np.mean(lats))  # Compute mean
    return cluster_centers

# Function to find cluster connections based on flight transitions
# def find_cluster_connections(flight_clusters):
#     connections = set()

#     for flight, cluster_sequence in flight_clusters.items():
#         clustered_path = [cluster for cluster in cluster_sequence if cluster != -1]  # Ignore unclustered waypoints

#         for i in range(len(clustered_path) - 1):
#             cluster1, cluster2 = clustered_path[i], clustered_path[i + 1]
#             if cluster1 != cluster2:  # Only link different clusters
#                 connections.add((cluster1, cluster2))

#     return connections

# Function to find unique cluster connections and their frequencies
def find_cluster_connections(flight_clusters):
    connection_counts = Counter()  # Store unique connections and frequencies
    flight_routes = defaultdict(list)

    for flight, cluster_sequence in flight_clusters.items():
        clustered_path = [cluster for cluster in cluster_sequence if cluster != -1]  # Ignore unclustered waypoints

        for i in range(len(clustered_path) - 1):
            cluster1, cluster2 = clustered_path[i], clustered_path[i + 1]
            if cluster1 != cluster2:  # Only link different clusters
                connection = tuple(sorted([cluster1, cluster2]))  # Ensure order consistency
                connection_counts[connection] += 1  # Count frequency
                flight_routes[connection].append(flight)  # Track flights per connection
    # Sort connections by frequency (descending order)
    sorted_connections = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_flight_routes = {conn: flight_routes[conn] for conn, _ in sorted_connections}

    return sorted_connections, sorted_flight_routes
def find_all_waypoint_connections(flight_waypoints):
    """
    Identifies connections between all pairs of waypoints in the same trajectory.
    Any two waypoints that belong to the same flight are considered connected.

    Parameters:
    - flight_waypoints (dict): {flight_id: [(lon1, lat1), (lon2, lat2), ...]}

    Returns:
    - sorted_connections (list): [(waypoint1, waypoint2), frequency] sorted by frequency.
    - waypoint_routes (dict): {connection: [flight_ids]}
    """
    connection_counts = Counter()
    waypoint_routes = defaultdict(list)

    for flight, waypoints in flight_waypoints.items():
        for i in range(len(waypoints) - 1):
            wp1, wp2 = waypoints[i], waypoints[i + 1]
            connection = tuple(sorted([wp1, wp2]))  # Ensure consistency
            connection_counts[connection] += 1
            waypoint_routes[connection].append(flight)

    # Sort connections by frequency (descending order)
    sorted_connections = sorted(connection_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_waypoint_routes = {conn: waypoint_routes[conn] for conn, _ in sorted_connections}

    return sorted_connections, sorted_waypoint_routes
# def variable_dbscan(points, min_samples=3, percentiles=[20, 40, 60, 80]):
#     # 1. Compute distance to k-th nearest neighbor as local density measure
#     nbrs = NearestNeighbors(n_neighbors=8).fit(points)
#     dists, _ = nbrs.kneighbors(points)
#     knn_dist = dists[:, -1]
    
#     # 2. Define multiple eps values at specified percentiles
#     eps_vals = np.percentile(knn_dist, percentiles)
    
#     labels = -np.ones(len(points), dtype=int)
#     label_offset = 0
#     eps_vals = np.percentile(knn_dist, percentiles)
#     eps_vals = [float(eps) for eps in eps_vals if eps > 1e-6]
#     # 3. Apply DBSCAN for each eps, merging results
#     for eps in eps_vals:
#         db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
#         new = db.labels_.copy()
        
#         # Shift cluster indices to avoid overlap
#         unique = set(new) - {-1}
#         mapping = {old: idx + label_offset for idx, old in enumerate(unique)}
#         for i, old in enumerate(new):
#             if old in mapping:
#                 new[i] = mapping[old]
        
#         # Merge non-noise labels into final labels
#         mask = new != -1
#         labels[mask] = new[mask]
        
#         label_offset = labels.max() + 1
    
#     return labels

# def dbscan_elbow_detection(
#     data: Union[List[List[float]], np.ndarray],
#     eps_range: List[float],
#     min_samples_values: List[int],
#     plot: bool = True
# ) -> Tuple[float, int, List[Tuple[float, int, int]]]:
#     """
#     Grid search on DBSCAN to find the elbow point (sharpest drop in #clusters).

#     Returns:
#         best_eps (float): Eps at elbow point
#         best_min_samples (int): min_samples at elbow point
#         results (list): (eps, min_samples, n_clusters) for each trial
#     """
#     data = np.array(data)
#     results = []
#     elbow_info = None
#     max_delta = 0

#     for min_samples in min_samples_values:
#         cluster_counts = []
#         for eps in eps_range:
#             labels = DBSCAN(eps=eps, min_samples=min_samples,metric='haversine').fit_predict(data)
#             n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#             cluster_counts.append(n_clusters)
#             results.append((eps, min_samples, n_clusters))

#         # Detect sharpest change (elbow)
#         deltas = np.diff(cluster_counts)
#         if len(deltas) > 0:
#             idx = np.argmax(np.abs(deltas))
#             if np.abs(deltas[idx]) > max_delta:
#                 max_delta = np.abs(deltas[idx])
#                 elbow_info = (eps_range[idx], min_samples)

#     if plot:
#         plt.figure(figsize=(10, 6))
#         for ms in min_samples_values:
#             counts = [r[2] for r in results if r[1] == ms]
#             plt.plot(eps_range, counts, label=f'min_samples={ms}', marker='o')

#         if elbow_info:
#             plt.axvline(elbow_info[0], linestyle='--', color='r', label=f'Elbow at eps={elbow_info[0]:.3f}')
#         plt.xlabel("eps")
#         plt.ylabel("Number of Clusters")
#         plt.title("DBSCAN: Clusters vs eps")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     if elbow_info:
#         print(f"Elbow Point: eps = {elbow_info[0]:.3f}, min_samples = {elbow_info[1]}")
#     return elbow_info[0]-0.001, elbow_info[1]+2, results
def hdbscan_param_effect_plot(
    data: Union[List[List[float]], np.ndarray],
    min_cluster_sizes: List[int],
    cluster_selection_epsilons: List[float],
    plot: bool = True
) -> Tuple[float, int, List[Tuple[float, int, int]]]:
    """
    Run HDBSCAN for combinations of min_cluster_size and cluster_selection_epsilon.
    Plot normalized number of clusters and number of noise points.

    Returns:
        results: List of tuples (min_cluster_size, epsilon, n_clusters, n_noise)
    """
    # data = np.array(data)
    results = []

    for mcs in min_cluster_sizes:
        for eps in cluster_selection_epsilons:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=mcs,  # set min_samples same as min_cluster_size
                cluster_selection_epsilon=float(eps),
                metric='haversine'
            )
            labels = clusterer.fit_predict(np.radians(data))
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            results.append((mcs, eps, n_clusters, n_noise))

    if plot:
        # Extract values
        n_clusters_list = [r[2] for r in results]
        n_noise_list = [r[3] for r in results]
        labels_list = [f"{r[0]}-{r[1]:.3f}" for r in results]

        # Normalize both metrics
        scaler = MinMaxScaler()
        clusters_norm = scaler.fit_transform(np.array(n_clusters_list).reshape(-1, 1)).flatten()
        noise_norm = scaler.fit_transform(np.array(n_noise_list).reshape(-1, 1)).flatten()

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(results))
        ax.plot(x, clusters_norm, 'o-', label='Normalized #Clusters')
        ax.plot(x, noise_norm, 'x-', label='Normalized #Noise Points')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=90)
        ax.set_xlabel("min_cluster_size - cluster_selection_epsilon")
        ax.set_ylabel("Normalized Values")
        ax.set_title("Effect of HDBSCAN Parameters on Clusters and Noise")
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        
        
        # Convert results into DataFrame


    return results
# Function to apply DBSCAN clustering
def apply_dbscan(points, eps=0.01, min_samples=1):
    """
    Applies DBSCAN clustering to a set of points.
    :param points: List of (lon, lat) tuples.
    :param eps: Maximum distance between two samples for one to be considered as in the neighborhood.
    :param min_samples: Minimum number of samples required to form a dense region.
    :return: Cluster labels.
    """
    if len(points) < min_samples:
        return [-1] * len(points)  # Assign all points to noise if not enough data
    # eps_values = np.arange(0.005, 0.021, 0.001)
    # min_samples_values = np.arange(2, 22, 2)
    # best_eps, best_min_samples, all_results = dbscan_elbow_detection(points, eps_values, min_samples_values)
    # clustering = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='haversine').fit(np.radians(points))
    # results = hdbscan_param_effect_plot(points, min_samples_values, eps_values)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size= 12, #12, #9,       # Controls how big a dense region must be
        min_samples=4,             # Increase to be stricter (lower = more permissive)
        cluster_selection_epsilon= 0.007, #0.007,  # Helps control merging small clusters
        metric='haversine'
    )

    labels = clusterer.fit_predict(np.radians(points))
    # clustering = DBSCAN(eps=0.01, min_samples=3, metric='haversine').fit(np.radians(points))
    # clustering = variable_dbscan(points, min_samples=5, percentiles=[20, 40, 60, 80])
    # clustering = HDBSCAN(min_cluster_size=12, min_samples = 2, metric='haversine').fit(np.radians(points))

    return labels #clustering.labels_

# Dictionary to store persistent cluster colors
cluster_colors = {}
# Function to track cluster changes across time windows

# Function to track cluster changes and preserve labels
def track_cluster_changes(previous_clusters, new_clusters, previous_labels):
    matched_clusters = {}  # Map new clusters to old labels
    disappeared_clusters = set(previous_clusters.keys())  # Start with all old clusters
    new_formed_clusters = set(new_clusters.keys())

    used_labels = set(previous_labels.values())  # Track all used labels
    next_new_label = max(used_labels, default=-1) + 1  # Get next available label

    for new_label, new_points in new_clusters.items():
        best_match = None
        max_overlap = 0

        for old_label, old_points in previous_clusters.items():
            overlap = len(new_points & old_points)  # Find common waypoints
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = old_label

        if best_match is not None:
            # Assign the old label to the new cluster
            matched_clusters[new_label] = best_match# previous_labels[best_match]
            disappeared_clusters.discard(best_match)  # Remove from disappeared list
        else:
            # Assign a new unique label to clusters with no match
            matched_clusters[new_label] = next_new_label
            new_formed_clusters.add(next_new_label)
            used_labels.add(next_new_label)
            next_new_label += 1  

    return matched_clusters, new_formed_clusters, disappeared_clusters

# Function to assign consistent colors to clusters
def assign_cluster_colors(matched_clusters, new_clusters, existing_colors):
    # import matplotlib.pyplot as plt
    num_colors = 20  # Adjust for a higher number of clusters
    cmap = plt.get_cmap("tab20")

    for new_label, old_label in matched_clusters.items():
        if old_label not in existing_colors:
            existing_colors[old_label] = cmap(len(existing_colors) % num_colors)
    # for new_label in new_clusters:
    #     if new_label not in existing_colors:
    #         existing_colors[new_label] = cmap(len(existing_colors) % num_colors)

    return existing_colors

# Function to dynamically plot every 4-hour traffic with a sliding window of 10 seconds
def plot_sliding_window_traffic_basemap(df, step_seconds=1800, window_hours=4,epsilon=0.1,dbscan_eps=0.01, dbscan_min_samples=3):
    min_time = df["timebase"].min()+timedelta(hours=10)
    max_time = df["endtime"].max() - timedelta(hours=window_hours)

    current_time = min_time
    previous_clusters = {}
    cluster_colors = {}
    previous_labels = {}  # Stores label assignments across frames
        
    while current_time <= max_time:
        flights_in_window = find_flights_within_next_n_hours(df,current_time)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define Basemap with Cylindrical Projection
        m = Basemap(projection='cyl', llcrnrlon=-20, urcrnrlon=30, llcrnrlat=26, urcrnrlat=66, resolution='i', ax=ax)
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(26, 66, 5), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-20, 30, 5), labels=[0, 0, 0, 1])

        all_waypoints = []
        waypoint_indices = []  # Keep track of point indices for tracking clusters
        flight_clusters = defaultdict(list)  # Track cluster sequences for each flight

        for _, flight in flights_in_window.iterrows():
            waypoints = flight["waypoints"]
            lon = rad_to_deg([wp["x"] for wp in waypoints])
            lat = rad_to_deg([wp["y"] for wp in waypoints])
            alt = [wp["z"] for wp in waypoints]  # Altitude in meters

            # Filter waypoints within the desired region
            # filtered_coords = [(lo, la) for lo, la in zip(lon, lat) if -20 <= lo <= 30 and 26 <= la <= 66]
            filtered_coords = [(lo, la, wp["name"]) for lo, la, al, wp in zip(lon, lat, alt, waypoints) if -20 <= lo <= 30 and 26 <= la <= 66 and al >= 6000]

            # if filtered_coords:
            #     filtered_lon, filtered_lat = zip(*filtered_coords)
            #     m.plot(filtered_lon, filtered_lat, marker='o', linestyle='-', markersize=4, label=f"{flight['callsign']}", alpha=0.8)

            # Apply Douglas-Peucker simplification
            if len(filtered_coords) > 2:
                simplified_coords = douglas_peucker([(lo, la) for lo, la, _ in filtered_coords], epsilon)
                simplified_waypoints = [wp for wp in filtered_coords if (wp[0], wp[1]) in simplified_coords]
            else:
                simplified_waypoints = filtered_coords

            # Store waypoints for clustering
            for lo, la, name in simplified_waypoints:
                all_waypoints.append((lo, la))
                waypoint_indices.append((flight.callsign, name))  # ✅ Track by (flight_id, waypoint_name)

            if simplified_coords:
                simplified_lon, simplified_lat = zip(*simplified_coords)
                # m.plot(simplified_lon, simplified_lat, marker='.', linestyle='-', markersize=5, 
                #        label=f"{flight['callsign']}", alpha=0.2, linewidth = 1, color = "m",zorder=2)

        # Apply DBSCAN clustering
        new_clusters = {}
        if all_waypoints:
            labels = apply_dbscan(all_waypoints, eps=dbscan_eps, min_samples=dbscan_min_samples)
            matched_clusters = {}
            # Organize clusters as {label: set(points)}
            current_clusters = defaultdict(set)
            for i, label in enumerate(labels):
                if label != -1:  # Ignore noise
                    current_clusters[label].add(all_waypoints[i])
                    flight_id, waypoint_name = waypoint_indices[i]
            # Compute cluster centers
            cluster_centers = compute_cluster_centers(current_clusters)
            # Track cluster changes with majority voting
            # if not previous_clusters:
            #     print("First iteration: No previous clusters to compare.")
            #     previous_labels = {label: label for label in current_clusters.keys()}  
            # else:
            matched_clusters, new_clusters, disappeared_clusters = track_cluster_changes(previous_clusters, current_clusters, previous_labels)

            print(f"New clusters formed: {new_clusters}")
            print(f"Clusters disappeared: {disappeared_clusters}")

            # Assign consistent colors
            cluster_colors = assign_cluster_colors(matched_clusters, new_clusters, cluster_colors)

        # Plot clusters with consistent colors
            for i, (point, label) in enumerate(zip(all_waypoints, labels)):
                updated_label = matched_clusters.get(label)  # ✅ Use updated label
                color = cluster_colors.get(updated_label, "black") if updated_label != -1 else "gray"
                flight_id, waypoint_name = waypoint_indices[i]
                flight_clusters[flight_id].append(updated_label)  # ✅ Track cluster sequence per flight
                # m.scatter(point[0], point[1], color=color, s=10, alpha=1,zorder=1)
                # ✅ Update previous_clusters at the end of each loop
            # Rename keys
            current_clusterscopy = defaultdict(set)
            for old_key, new_key in matched_clusters.items():
                if old_key in current_clusters:
                    current_clusterscopy[new_key] = current_clusters.pop(old_key)  # Move values and remove old key

            # Compute cluster centers
            cluster_centers = compute_cluster_centers(current_clusterscopy)
            # Find connections based on flights transitioning between clusters
            connection_counts, flight_routes = find_cluster_connections(flight_clusters)
            # Plot cluster centers
            for label, (lon, lat) in cluster_centers.items():
                color = cluster_colors.get(label, "black")
                m.scatter(lon, lat, color=color, edgecolors="black", s=30, marker="X", linewidth=0.5, label=f"Cluster {label}",zorder=4)

            # Draw connections between cluster centers with thickness based on frequency
            for (label1, label2), freq in connection_counts:
                if label1 in cluster_centers and label2 in cluster_centers:
                    lon1, lat1 = cluster_centers[label1]
                    lon2, lat2 = cluster_centers[label2]
                    m.plot([lon1, lon2], [lat1, lat2], color="black", linestyle="-", alpha=0.6, linewidth=0.8)

            
            previous_clusters = current_clusterscopy.copy()
            previous_labels = matched_clusters


        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(f"Filtered & Simplified Air Traffic with Clustering from {current_time} to {current_time + timedelta(hours=window_hours)}")

        # ax.legend()
        ax.grid()
        

        # Save the plot
        filename = f"plots/filtered_traffic_{current_time.strftime('%Y%m%d_%H%M%S')}path_cluster.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot: {filename}")

        current_time += timedelta(seconds=step_seconds)

# # Run the function using Basemap
# if __name__ == "__main__":
#     df_flights = fl2df("/Users/machunyao/Documents/DeepFlow/fl_sim/fl_mar/20230301_m1.so6.7z_json")
#     plot_sliding_window_traffic_basemap(df_flights)
