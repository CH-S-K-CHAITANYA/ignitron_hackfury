import csv
import requests
import logging
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pathlib import Path
import random
import heapq
import math

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CSV_PATH = Path("hubs_bangalore.csv")


# ------------------------
# Utility: Haversine distance (in km)
# ------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ------------------------
# Load hubs from your CSV only
# ------------------------
def load_hubs_from_csv():
    hubs = []
    if not CSV_PATH.exists():
        logging.error(f"Hub CSV not found: {CSV_PATH.resolve()}")
        return []

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for i, row in enumerate(reader, start=2):
            try:
                node_id = str(row[0]).strip()
                lat = float(row[1].strip())
                lon = float(row[2].strip())
                name = row[3].strip()
                hubs.append({"id": node_id, "name": name, "lat": lat, "lng": lon})
            except Exception as e:
                logging.warning(f"Skipping row {i} due to parse error: {e}")
                continue
    return hubs


# ------------------------
# Attach load simulation (temporary)
# Replace this with your IoT data later
# ------------------------
def attach_loads(hubs):
    for h in hubs:
        h["load_percent"] = random.randint(10, 95)
        h["capacity"] = 100
    return hubs


# ------------------------
# A* Pathfinding Algorithm
# ------------------------
def a_star_search(source, destination, hubs):
    """
    A* pathfinding to find the optimal route through hubs
    considering distance and the destination heuristic.
    """

    # Calculate distance between two lat/lng points using Haversine formula
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the Earth in km
        dlat = (lat2 - lat1) * math.pi / 180
        dlng = (lon2 - lon1) * math.pi / 180
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dlng / 2) * math.sin(dlng / 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # Heuristic: Straight-line distance to destination (destination is now just lat/lng)
    def heuristic(hub, destination):
        return haversine_km(hub["lat"], hub["lng"], destination["lat"], destination["lng"])

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(source, destination), 0, source, []))  # (f, g, node, path)
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        # Check if we've reached the destination by proximity
        dist_to_dest = haversine_km(current["lat"], current["lng"], destination["lat"], destination["lng"])
        if dist_to_dest <= 7:  # If we are within 7 km of the destination
            return path + [current]

        if current["id"] in visited:
            continue

        visited.add(current["id"])

        # Explore neighbors (other hubs)
        for neighbor in hubs:
            if neighbor["id"] == current["id"] or neighbor["id"] in visited:
                continue

            g_cost = g + haversine_km(current["lat"], current["lng"], neighbor["lat"], neighbor["lng"])
            f_cost = g_cost + heuristic(neighbor, destination)

            heapq.heappush(open_set, (f_cost, g_cost, neighbor, path + [current]))

    return []  # No path found


# ------------------------
# Flask Routes
# ------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/hubs", methods=["GET"])
def api_hubs():
    hubs = load_hubs_from_csv()
    if not hubs:
        logging.error("No valid hubs loaded from CSV. Returning 500.")
        return jsonify({"error": "No valid hubs found in CSV. Please check the file format."}), 500

    hubs = attach_loads(hubs)
    logging.info(f"Loaded {len(hubs)} hubs from CSV.")
    return jsonify({"hubs": hubs})


@app.route("/api/geocode", methods=["POST"])
def api_geocode():
    data = request.get_json() or {}
    address = (data.get("address") or "").strip()
    if not address:
        return jsonify({"error": "Provide address"}), 400

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "mappls-opt-demo/1.0 (you@example.com)"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return jsonify({"error": "Address not found"}), 404
        top = results[0]
        lat = float(top["lat"])
        lng = float(top["lon"])
        display = top.get("display_name", address)
        logging.info(f"Geocode: '{address}' -> {lat:.6f},{lng:.6f}")
        return jsonify({"lat": lat, "lng": lng, "display_name": display})
    except requests.RequestException as e:
        logging.error(f"Geocode request failed: {e}")
        return jsonify({"error": "Geocoding failed"}), 502



@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """
    Stepwise routing: select hubs that steadily move toward destination.
    Picks hubs with acceptable load (<=66%) within ~10 km,
    continuing until we’re within 3 km of destination or hit 15 hops.
    """

    data = request.get_json() or {}
    source_id = str(data.get("sourceHubId", "")).strip()
    dest = data.get("destination")

    if not source_id or not dest or "lat" not in dest or "lng" not in dest:
        return jsonify({"error": "Invalid request: missing sourceHubId or destination"}), 400

    hubs = attach_loads(load_hubs_from_csv())
    source = next((h for h in hubs if str(h["id"]) == source_id), None)
    if not source:
        return jsonify({"error": f"Source hub ID {source_id} not found"}), 404

    ACCEPTABLE_LOAD = 66
    RANGE_MIN = 2.0   # skip tiny hops
    RANGE_MAX = 10.0  # max distance per hop
    STOP_DISTANCE = 3.0  # when to end near destination

    selected = [source]
    visited = {source["id"]}
    current = source

    for hop in range(15):
        dist_to_dest = haversine_km(current["lat"], current["lng"], dest["lat"], dest["lng"])
        logging.info(f"Hop {hop}: {current['name']} ({dist_to_dest:.2f} km to dest)")

        # stop when really close to destination
        if dist_to_dest <= STOP_DISTANCE:
            logging.info("Destination proximity reached (<3 km).")
            break

        # candidate hubs within 2–10 km that move closer & have good load
        candidates = []
        for h in hubs:
            if h["id"] in visited:
                continue
            d = haversine_km(current["lat"], current["lng"], h["lat"], h["lng"])
            if not (RANGE_MIN <= d <= RANGE_MAX):
                continue
            load = h["load_percent"]
            if load > ACCEPTABLE_LOAD:
                continue
            progress = dist_to_dest - haversine_km(h["lat"], h["lng"], dest["lat"], dest["lng"])
            if progress <= 0:
                continue
            # higher progress & lighter load are better
            score = progress * 10 + (100 - load) - d
            candidates.append((score, h))

        if not candidates:
            logging.info("No good hub found this hop.")
            break

        # choose the highest scoring hub
        best = max(candidates, key=lambda x: x[0])[1]
        selected.append(best)
        visited.add(best["id"])
        current = best

    # add final "destination" node explicitly
    selected.append({"id": "DEST", "name": "Destination", "lat": dest["lat"], "lng": dest["lng"], "load_percent": 0})
    logging.info(f"✅ Optimization complete with {len(selected)} hubs (incl. destination).")

    waypoints = [
        {"id": h["id"], "name": h["name"], "lat": h["lat"], "lng": h["lng"], "load": h["load_percent"]}
        for h in selected
    ]
    return jsonify({"selected_hubs": waypoints})




if __name__ == "__main__":
    logging.info("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)