# Importing necessary libraries
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from IPython.display import display
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Function to correct SO6-style time: Date + seconds since midnight
# def parse_datetime(date_val, time_sec):
#     date_str = str(int(date_val)).zfill(6)         # e.g. "230713"
#     full_date_str = "20" + date_str                # → "20230713"
#     date_base = datetime.strptime(full_date_str, "%Y%m%d")
#     return date_base + timedelta(seconds=int(time_sec))

def assign_flight_instance(group):
    z = group['z'].values
    n = len(z)
    # Mark in-air points
    in_air = z > 0

    # Find first and last in-air indices
    if not in_air.any():
        # All points are on ground, treat as one segment
        seg = np.zeros(n, dtype=int)
    else:
        first_air = np.argmax(in_air)
        last_air = n - 1 - np.argmax(in_air[::-1])
        seg = np.zeros(n, dtype=int)
        # Only consider splitting between first_air and last_air
        # Find transitions from ground to air in this region
        transitions = np.flatnonzero((z[first_air:last_air] == 0) & (z[first_air+1:last_air+1] > 0)) + first_air + 1
        # Assign segment numbers
        seg_num = 0
        prev = 0
        for t in transitions:
            seg[prev:t] = seg_num
            seg_num += 1
            prev = t
        seg[prev:] = seg_num  # assign remaining
    # Assign flight_instance
    return group.assign(
        flight_instance=group['FlightID'].astype(str) + '_' + (seg + 1).astype(str)
    )

def row_to_dict(row):
    return {
        'name': row['wp_name'],
        'x': row['x'],
        'y': row['y'],
        'z': row['z'],
        't': row['t']
    }

def make_flight_track(group):
    waypoints = group.sort_values('t').apply(row_to_dict, axis=1).tolist()
    start_time = pd.to_datetime(group['t'].min(), unit='s')
    end_time = pd.to_datetime(group['t'].max(), unit='s')
    return pd.Series({'waypoints': waypoints, 'start_time': start_time, 'end_time': end_time})

def trim_zeros_and_update_times(row):
    waypoints = row['waypoints']
    # Remove extra leading zeros
    i = 0
    while i + 1 < len(waypoints) and waypoints[i]['z'] == 0 and waypoints[i+1]['z'] == 0:
        i += 1
    # Remove extra trailing zeros
    j = len(waypoints) - 1
    while j - 1 > i and waypoints[j]['z'] == 0 and waypoints[j-1]['z'] == 0:
        j -= 1
    trimmed = waypoints[i:j+1]
    # Update times
    start_time = pd.to_datetime(trimmed[0]['t'], unit='s')
    end_time = pd.to_datetime(trimmed[-1]['t'], unit='s')
    return pd.Series({'waypoints': trimmed, 'start_time': start_time, 'end_time': end_time})

def hhmmss_to_timedelta(val):
    s = str(val).zfill(6)
    h, m, s = int(s[:2]), int(s[2:4]), int(s[4:6])
    return timedelta(hours=h, minutes=m, seconds=s)
def read_data_v2(filename):
    columns = ["SegmentID", "Origin", "Destination", "Time1", "Time2",
            "FL_begin", "FL_end", "Status", "Callsign", "Date_begin", "Date_end",
            "Lat_begin", "Lon_begin", "Lat_end", "Lon_end",
            "FlightID", "Sequence", "SegmentLength", "SegmentParity"]

    df = pd.read_csv(filename, sep=r'\s+', names=columns, low_memory=False)
    # Convert lat/lon to degrees
    df["Lat_begin_deg"] = df["Lat_begin"] / 60
    df["Lon_begin_deg"] = df["Lon_begin"] / 60
    df["Lat_end_deg"] = df["Lat_end"] / 60
    df["Lon_end_deg"] = df["Lon_end"] / 60

    df["Lat_begin_rad"] = np.deg2rad(df["Lat_begin"] / 60)
    df["Lon_begin_rad"] = np.deg2rad(df["Lon_begin"] / 60)
    df["Lat_end_rad"] = np.deg2rad(df["Lat_end"] / 60)
    df["Lon_end_rad"] = np.deg2rad(df["Lon_end"] / 60)

    # Convert date column to datetime first
    df["Date_begin"] = pd.to_datetime("20" + df["Date_begin"].astype(int).astype(str).str.zfill(6), format="%Y%m%d")
    df["Date_end"] = pd.to_datetime("20" + df["Date_end"].astype(int).astype(str).str.zfill(6), format="%Y%m%d")

    # Add seconds as timedelta
    # df["Timestamp1"] = df["Date_begin"] + pd.to_timedelta(df["Time1"], unit='s')
    df['Timestamp1'] = df.apply(lambda row: row['Date_begin'] + hhmmss_to_timedelta(row['Time1']), axis=1)
    df['Timestamp2'] = df.apply(lambda row: row['Date_end'] + hhmmss_to_timedelta(row['Time2']), axis=1)

    # filtered_df = df[df['FlightID'] == 263264937]

        
    df[['waypoint1', 'waypoint2']] = df["SegmentID"].str.split("_", expand = True)

    df1 = df.copy()
    df1['wp_name'] = df1['waypoint1']
    df1['t'] = df1['Timestamp1'].apply(lambda x: int(x.timestamp()))
    df1['x'] = df1['Lon_begin_rad']
    df1['y'] = df1['Lat_begin_rad']
    # df1['x'] = df1['Lon_begin']
    # df1['y'] = df1['Lat_begin']
    df1['z'] = df1['FL_begin'] * 100 * 0.3048               # Convert FL to meters
    df1 = df1[['FlightID', 'Origin', 'Destination', 'wp_name', 'x', 'y', 'z', 't']]


    df2 = df.copy()
    df2['wp_name'] = df2['waypoint2']
    df2['t'] = df2['Timestamp2'].apply(lambda x: int(x.timestamp()))
    df2['x'] = df2['Lon_end_rad']
    df2['y'] = df2['Lat_end_rad']
    # df2['x'] = df2['Lon_end']
    # df2['y'] = df2['Lat_end']
    df2['z'] = df2['FL_end'] * 100 * 0.3048                 # Convert FL to meters
    df2 = df2[['FlightID', 'Origin', 'Destination', 'wp_name', 'x', 'y', 'z', 't']]

    df_combined = pd.concat([df1, df2], ignore_index = True).drop_duplicates().reset_index(drop=True)
    df_combined['wp_name'] = df_combined['wp_name'].astype(str)
    points = df_combined.sort_values(['FlightID', 't'], ascending=[True, True]).reset_index(drop=True)

    # points = points.groupby('FlightID', group_keys=False).apply(assign_flight_instance)
    df_flights = (
        points.groupby('FlightID')
        .apply(make_flight_track)
        .reset_index()
        #   .rename(columns={'flight_instance': 'FlightID'})
    )
    df_flights = df_flights.rename(columns={
        "FlightID": "id",
        "start_time": "timebase",
        "end_time": "endtime"
    })
    df_flights["callsign"] = df_flights["id"]
    return df_flights
    
# # Run the function using Basemap
# if __name__ == "__main__":    
#     filename = "DeepFlow_Data_v00.03/20230714_NW_SW_Axis_InitialFlw.so6"
#     df_flights = read_data_v2(filename)
