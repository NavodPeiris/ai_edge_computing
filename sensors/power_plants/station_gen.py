import numpy as np


# lats are horizontal, lons are vertical
def station_provider(src, dest):
    # Coefficients of the equations
    A = np.array([[src["lat"], 1], [dest["lat"], 1]])  # Coefficient matrix
    B = np.array([src["lon"], dest["lon"]])           # Constant terms

    # Solve for x
    solution = np.linalg.solve(A, B)
    m = solution[0]
    c = solution[1]
    print("Solution:")
    print(f"m = {m}, c = {c}")

    stations = []
    src_id = src["id"]
    dest_id = dest["id"]
    
    if(dest["lat"] > src["lat"]):
        lat_gap = dest["lat"] - src["lat"]
        station1_lat = src["lat"] + (lat_gap / 4)
        station2_lat = src["lat"] + (lat_gap / 4) * 2
        station3_lat = src["lat"] + (lat_gap / 4) * 3
        stations.append({
            "id": int(f"{src_id}{dest_id}1"),
            "title": f"station_{src_id}{dest_id}_1",
            "lat": round(station1_lat, 6),
            "lon": round(m * station1_lat + c, 6),
            "next": int(f"{src_id}{dest_id}2")
        })

        stations.append({
            "id": int(f"{src_id}{dest_id}2"),
            "title": f"station_{src_id}{dest_id}_2",
            "lat": round(station2_lat, 6),
            "lon":  round(m * station2_lat + c, 6),
            "next": int(f"{src_id}{dest_id}3")
        })

        stations.append({
            "id": int(f"{src_id}{dest_id}3"),
            "title": f"station_{src_id}{dest_id}_3",
            "lat": round(station3_lat, 6),
            "lon":  round(m * station3_lat + c, 6),
            "next": int(dest_id)
        })
    else:
        lat_gap = src["lat"] - dest["lat"]
        station1_lat = dest["lat"] + (lat_gap / 4)
        station2_lat = dest["lat"] + (lat_gap / 4) * 2
        station3_lat = dest["lat"] + (lat_gap / 4) * 3
        stations.append({
            "id": int(f"{src_id}{dest_id}1"),
            "title": f"station_{src_id}{dest_id}_1",
            "lat": round(station1_lat, 6),
            "lon": round(m * station1_lat + c, 6),
            "next": int(f"{src_id}{dest_id}2")
        })

        stations.append({
            "id": int(f"{src_id}{dest_id}2"),
            "title": f"station_{src_id}{dest_id}_2",
            "lat": round(station2_lat, 6),
            "lon":  round(m * station2_lat + c, 6),
            "next": int(f"{src_id}{dest_id}3")
        })

        stations.append({
            "id": int(f"{src_id}{dest_id}3"),
            "title": f"station_{src_id}{dest_id}_3",
            "lat": round(station3_lat, 6),
            "lon":  round(m * station3_lat + c, 6),
            "next": int(src_id)
        })
    
    return stations
