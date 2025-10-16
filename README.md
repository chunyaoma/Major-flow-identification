"flowIdentification" is the main function to get the major flow identification output. Read the data through the code "df_flights = read_data_v2("/Path/general_initial_flow_NW_SW_Axis.so6")." Replace the "Path" with the corresponding folder where the input data is stored. The date for flow identification can be changed via the code -- date = "2023-07-20" by replacing it with the new date. The output is the pandas dataframe file “df_flow.pkl”. When you read it, first you will see two columns: “OD” (describing the start and end location of each major flow in longitude and latitude), and “info”, describing the flight information in each major flow. In the many rows of each “info”, each row describes a flight in the flow: flight ID, start (entry) time in the flow (st), end (exit) time in the flow (et), start waypoint index in “waypoints” (sid), end waypoint index in “waypoints” (eid), and the waypoints (“waypoints”).

"data_harmonize" is to reorganize the format of the input data based on each flight.

"loaddata" provides basic functions for data processing.

"cluster_threshold" is used to refine the clustering result, identifying significant areas that may shape the origin, termination, or transit of major flows.
