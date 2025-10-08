## Importing necessary libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj

from IPython.display import display

from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize
from shapely.ops import transform

from scipy.spatial import Delaunay
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


## Defining functions for clustering and shape processing
def apply_dbscan(points, min_cluster_size=12, min_samples=4, cluster_selection_epsilon=0.007, metric = 'haversine'):
    """
    Apply HDBSCAN clustering to the given points.
    
    Parameters:
    :param points: DataFrame with 'lon' and 'lat' columns representing latitude and longitude in deg.
    :param min_cluster_size: Minimum size of clusters.
    :param min_samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
    :param cluster_selection_epsilon: Epsilon value for cluster selection.
    :param metric: Distance metric to use for clustering.
    
    Returns:
    :return DataFrame with cluster labels.
    """
    if len(points) < min_samples:
        return [-1] * len(points)               # Return all points as noise if not enough points

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples, 
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric = metric
        )
    
    points['cluster'] = clusterer.fit_predict(np.radians(points[['lon', 'lat']]))
    
    return points


def alpha_shape(points, alpha):
    """
    Computes the alpha shape of a set of points.
    :param points: List of (lon, lat) tuples or a similar 2D array-like structure.
    :param alpha: Alpha parameter for the alpha shape; models how "tight" the shape is.

    :return: A Shapely polygon representing the alpha shape.
    """
    if len(points) < 4:
        return MultiPoint(list(points)).convex_hull
    coords = np.array(points)
    try:
        tri = Delaunay(coords)
    except Exception:
        return MultiPoint(list(points)).convex_hull
    triangles = coords[tri.simplices]
    a = np.linalg.norm(triangles[:,0] - triangles[:,1], axis=1)
    b = np.linalg.norm(triangles[:,1] - triangles[:,2], axis=1)
    c = np.linalg.norm(triangles[:,2] - triangles[:,0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    with np.errstate(divide='ignore', invalid='ignore'):
        circum_r = np.where(area > 1e-10, a * b * c / (4.0 * area), np.inf)
    filter_ = circum_r < 1.0 / alpha
    edges = set()
    for ia, simplex in enumerate(tri.simplices[filter_]):
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i+1)%3])))
            edges.add(edge)
    edge_points = [coords[list(edge)] for edge in edges]
    m = MultiPoint(coords)
    if not edge_points:
        return m.convex_hull
    mls = unary_union([LineString(edge) for edge in edge_points])
    result = list(polygonize(mls))
    if not result:
        return m.convex_hull
    return unary_union(result)


def count_concave_vertices(polygon, cross_threshold=1e-8):
    """
    Counts the number of concave vertices in a polygon.
    :param polygon: A Shapely polygon object.
    :param cross_threshold: Threshold for the cross product to consider a vertex concave.

    :return: Number of concave vertices in the polygon.
    """
    # Returns the number of concave vertices in a polygon (cyclic, CCW)
    coords = list(polygon.exterior.coords)
    n = len(coords)
    count = 0

    # Calculate signed area to determine orientation (positive = CCW)
    area = 0
    for i in range(n - 1):
        area += coords[i][0] * coords[i + 1][1] - coords[i + 1][0] * coords[i][1]
    ccw = area > 0

    for i in range(n):
        p0 = np.array(coords[i - 1])
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % n])
        v1 = p1 - p0
        v2 = p2 - p1
        cross = np.cross(v1, v2)
        # Only count if cross product is significant
        if abs(cross) > cross_threshold:
            if (cross < 0 and ccw) or (cross > 0 and not ccw):
                count += 1
    return count


def simplify_shape_km(shape, cluster_points, tolerance_km):
    """
    Simplifies a Shapely polygon using the Douglas-Peucker algorithm. 
    :param shape: A Shapely polygon object to be simplified.
    :param cluster_points: DataFrame with 'lon' and 'lat' columns representing latitude and longitude in deg.
    :param tolerance_km: Tolerance in kilometers for simplification.

    :return: A simplified Shapely polygon object.
    """
    # 1. Find centroid for local projection
    lon0 = cluster_points['lon'].mean()
    lat0 = cluster_points['lat'].mean()
    # 2. Define Azimuthal Equidistant projection centered on cluster
    proj_wgs84 = pyproj.CRS('EPSG:4326')
    proj_aeqd = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    project = pyproj.Transformer.from_crs(proj_wgs84, proj_aeqd, always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(proj_aeqd, proj_wgs84, always_xy=True).transform

    # 3. Project shape to meters
    shape_proj = transform(project, shape)
    # 4. Simplify in meters (tolerance_km*1000)
    simple_proj = shape_proj.simplify(tolerance_km*1000, preserve_topology=True)
    # 5. Project back to lat/lon
    simple_shape = transform(project_back, simple_proj)
    return simple_shape


def split_by_simplified_alpha_shape(points_df, alpha=2, tolerance_km=120, min_size=1000, min_concave=1):
    """
    For each cluster, smooth the alpha shape using Douglas-Peucker, count concave vertices,
    and split using KMeans with k = n_concave + 1.
    """
    df = points_df.copy()
    next_cluster = df['cluster'].max() + 1
    unique_clusters = df['cluster'].unique()
    unique_clusters = unique_clusters[unique_clusters != -1]
    for cluster_id in unique_clusters:
        cluster_points = df[df['cluster'] == cluster_id]
        if len(cluster_points) < 4:
            continue
        coords = cluster_points[['lon', 'lat']].values
        shape = alpha_shape(coords, alpha)
        if shape.geom_type == 'Polygon':
            # Simplify the shape
            simple_shape = simplify_shape_km(shape, cluster_points, tolerance_km = tolerance_km)
            n_concave = count_concave_vertices(simple_shape)
            if n_concave >= min_concave and len(cluster_points) > min_size:
                k = n_concave + 1
                if k < 2 or k >= len(cluster_points):
                    continue  # Skip if k is not valid
                kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
                sub_labels = kmeans.fit_predict(coords)
                df.loc[cluster_points.index, 'cluster'] = sub_labels + next_cluster
                next_cluster += k
    return df


def plot_clusters(refined_clusters, filepath = None, show = 1):
    """
    Plots clustered DataFrame on a Cartopy map ignoring noise points.

    :param refined_clusters: DataFrame with 'lon', 'lat', and 'cluster' columns.
    :param filepath: Complete absolute file path (including file name) and .png extension to save plotted figure.

    :return: None
    """
    # Set extent for the map
    LON_START = -20.00
    LON_END = 30.00
    LAT_START = 26.00
    LAT_END = 66.00

    # Plotting the results
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent to your region of interest (adjust as needed)
    ax.set_extent([LON_START, LON_END, LAT_START, LAT_END], crs = ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle = ":")
    ax.add_feature(cfeature.LAND, facecolor = '#e8eaeb')
    ax.add_feature(cfeature.OCEAN, facecolor = '#6bb8c9')
    ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)

    # Add longitude and latitude ticks
    xticks = np.arange(LON_START, LON_END + 1, 2)
    yticks = np.arange(LAT_START, LAT_END + 1, 2)
    ax.set_xticks(xticks[::2], crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Get unique clusters (excluding noise points)
    unique_clusters = refined_clusters['cluster'].unique()
    unique_clusters = unique_clusters[unique_clusters != -1]

    # Plot noise points in gray
    noise_points = refined_clusters[refined_clusters['cluster'] == -1]
    ax.scatter(noise_points['lon'], noise_points['lat'], 
            c='gray', alpha=0, s=10, label='Noise',
            transform=ccrs.PlateCarree())

    # Plot clustered points with different colors
    for cluster_id in unique_clusters:
        cluster_points = refined_clusters[refined_clusters['cluster'] == cluster_id]
        ax.scatter(cluster_points['lon'], cluster_points['lat'],
                alpha=0.65, s=20, label=f'Cluster {cluster_id}',
                transform=ccrs.PlateCarree())

    # # Add legend with smaller font size and outside the plot
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
    #          fontsize=8, markerscale=0.8)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches = 'tight', dpi = 300)
    
    if show:
        plt.show()

    return fig



# ## Demonstration of the developed methods
# with open('/Users/atmri-mac-217/Desktop/Python Documents/DeepFlow/data/crida_cloud/points.pkl', 'rb') as f:
#     points = pickle.load(f)
# points = pd.DataFrame(points, columns = ['lon', 'lat'])

# clustered_points = apply_dbscan(points)
# refined_clusters = split_by_simplified_alpha_shape(clustered_points, alpha = 2, tolerance_km = 160, min_concave=1)

# plot_clusters(refined_clusters)


