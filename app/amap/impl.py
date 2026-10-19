from typing import Any, Dict, Optional
from . import client


async def maps_regeocode(location: str) -> Dict[str, Any]:
    data, err = await client.get("/v3/geocode/regeo", {"location": location})
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"RGeocoding failed: {data.get('info') or data.get('infocode')}"}
    return {
        "province": data["regeocode"]["addressComponent"]["province"],
        "city": data["regeocode"]["addressComponent"]["city"],
        "district": data["regeocode"]["addressComponent"]["district"],
    }


async def maps_geo(address: str, city: Optional[str] = None) -> Dict[str, Any]:
    params = {"address": address}
    if city:
        params["city"] = city
    data, err = await client.get("/v3/geocode/geo", params)
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Geocoding failed: {data.get('info') or data.get('infocode')}"}
    geocodes = data.get("geocodes", [])
    results = [
        {
            "country": geo.get("country"),
            "province": geo.get("province"),
            "city": geo.get("city"),
            "citycode": geo.get("citycode"),
            "district": geo.get("district"),
            "street": geo.get("street"),
            "number": geo.get("number"),
            "adcode": geo.get("adcode"),
            "location": geo.get("location"),
            "level": geo.get("level"),
        }
        for geo in geocodes
    ]
    return {"return": results}


async def maps_ip_location(ip: str) -> Dict[str, Any]:
    data, err = await client.get("/v3/ip", {"ip": ip})
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"IP Location failed: {data.get('info') or data.get('infocode')}"}
    return {
        "province": data.get("province"),
        "city": data.get("city"),
        "adcode": data.get("adcode"),
        "rectangle": data.get("rectangle"),
    }


async def maps_weather(city: str) -> Dict[str, Any]:
    data, err = await client.get("/v3/weather/weatherInfo", {"city": city, "extensions": "all"})
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Get weather failed: {data.get('info') or data.get('infocode')}"}
    forecasts = data.get("forecasts", [])
    if not forecasts:
        return {"error": "No forecast data available"}
    return {
        "city": forecasts[0]["city"],
        "forecasts": forecasts[0]["casts"],
    }


async def maps_bicycling_by_coordinates(
    origin_coordinates: str, destination_coordinates: str
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v4/direction/bicycling",
        {
            "origin": origin_coordinates,
            "destination": destination_coordinates,
        },
    )
    if err:
        return {"error": err}
    if data.get("errcode") != 0:
        return {"error": f"Direction bicycling failed: {data.get('info') or data.get('infocode')}"}
    paths = []
    for path in data["data"]["paths"]:
        steps = [
            {
                "instruction": step.get("instruction"),
                "road": step.get("road"),
                "distance": step.get("distance"),
                "orientation": step.get("orientation"),
                "duration": step.get("duration"),
            }
            for step in path["steps"]
        ]
        paths.append(
            {
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "steps": steps,
            }
        )
    return {
        "data": {
            "origin": data["data"]["origin"],
            "destination": data["data"]["destination"],
            "paths": paths,
        }
    }


async def maps_bicycling_by_address(
    origin_address: str,
    destination_address: str,
    origin_city: Optional[str] = None,
    destination_city: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        origin_result = await maps_geo(origin_address, origin_city)
        if "error" in origin_result:
            return {"error": f"Failed to geocode origin address: {origin_result['error']}"}
        if not origin_result.get("return") or not origin_result["return"]:
            return {"error": "No geocoding results found for origin address"}
        origin_location = origin_result["return"][0].get("location")
        if not origin_location:
            return {"error": "Could not extract coordinates from origin geocoding result"}

        destination_result = await maps_geo(destination_address, destination_city)
        if "error" in destination_result:
            return {"error": f"Failed to geocode destination address: {destination_result['error']}"}
        if not destination_result.get("return") or not destination_result["return"]:
            return {"error": "No geocoding results found for destination address"}
        destination_location = destination_result["return"][0].get("location")
        if not destination_location:
            return {"error": "Could not extract coordinates from destination geocoding result"}

        route_result = await maps_bicycling_by_coordinates(origin_location, destination_location)
        if "error" not in route_result:
            route_result["addresses"] = {
                "origin": {
                    "address": origin_address,
                    "coordinates": origin_location,
                },
                "destination": {
                    "address": destination_address,
                    "coordinates": destination_location,
                },
            }
        return route_result
    except Exception as e:
        return {"error": f"Route planning failed: {str(e)}"}


async def maps_direction_walking_by_coordinates(
    origin: str, destination: str
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v3/direction/walking",
        {"origin": origin, "destination": destination},
    )
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Direction Walking failed: {data.get('info') or data.get('infocode')}"}
    paths = []
    for path in data["route"]["paths"]:
        steps = [
            {
                "instruction": step.get("instruction"),
                "road": step.get("road"),
                "distance": step.get("distance"),
                "orientation": step.get("orientation"),
                "duration": step.get("duration"),
            }
            for step in path["steps"]
        ]
        paths.append(
            {
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "steps": steps,
            }
        )
    return {
        "route": {
            "origin": data["route"]["origin"],
            "destination": data["route"]["destination"],
            "paths": paths,
        }
    }


async def maps_direction_walking_by_address(
    origin_address: str,
    destination_address: str,
    origin_city: Optional[str] = None,
    destination_city: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        origin_result = await maps_geo(origin_address, origin_city)
        if "error" in origin_result:
            return {"error": f"Failed to geocode origin address: {origin_result['error']}"}
        if not origin_result.get("return") or not origin_result["return"]:
            return {"error": "No geocoding results found for origin address"}
        origin_location = origin_result["return"][0].get("location")
        if not origin_location:
            return {"error": "Could not extract coordinates from origin geocoding result"}

        destination_result = await maps_geo(destination_address, destination_city)
        if "error" in destination_result:
            return {"error": f"Failed to geocode destination address: {destination_result['error']}"}
        if not destination_result.get("return") or not destination_result["return"]:
            return {"error": "No geocoding results found for destination address"}
        destination_location = destination_result["return"][0].get("location")
        if not destination_location:
            return {"error": "Could not extract coordinates from destination geocoding result"}

        route_result = await maps_direction_walking_by_coordinates(origin_location, destination_location)
        if "error" not in route_result:
            route_result["addresses"] = {
                "origin": {
                    "address": origin_address,
                    "coordinates": origin_location,
                },
                "destination": {
                    "address": destination_address,
                    "coordinates": destination_location,
                },
            }
        return route_result
    except Exception as e:
        return {"error": f"Route planning failed: {str(e)}"}


async def maps_direction_driving_by_coordinates(
    origin: str, destination: str
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v3/direction/driving",
        {"origin": origin, "destination": destination},
    )
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Direction Driving failed: {data.get('info') or data.get('infocode')}"}
    paths = []
    for path in data["route"]["paths"]:
        steps = [
            {
                "instruction": step.get("instruction"),
                "road": step.get("road"),
                "distance": step.get("distance"),
                "orientation": step.get("orientation"),
                "duration": step.get("duration"),
            }
            for step in path["steps"]
        ]
        paths.append(
            {
                "path": path.get("path"),
                "distance": path.get("distance"),
                "duration": path.get("duration"),
                "steps": steps,
            }
        )
    return {
        "route": {
            "origin": data["route"]["origin"],
            "destination": data["route"]["destination"],
            "paths": paths,
        }
    }


async def maps_direction_driving_by_address(
    origin_address: str,
    destination_address: str,
    origin_city: Optional[str] = None,
    destination_city: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        origin_result = await maps_geo(origin_address, origin_city)
        if "error" in origin_result:
            return {"error": f"Failed to geocode origin address: {origin_result['error']}"}
        if not origin_result.get("return") or not origin_result["return"]:
            return {"error": "No geocoding results found for origin address"}
        origin_location = origin_result["return"][0].get("location")
        if not origin_location:
            return {"error": "Could not extract coordinates from origin geocoding result"}

        destination_result = await maps_geo(destination_address, destination_city)
        if "error" in destination_result:
            return {"error": f"Failed to geocode destination address: {destination_result['error']}"}
        if not destination_result.get("return") or not destination_result["return"]:
            return {"error": "No geocoding results found for destination address"}
        destination_location = destination_result["return"][0].get("location")
        if not destination_location:
            return {"error": "Could not extract coordinates from destination geocoding result"}

        route_result = await maps_direction_driving_by_coordinates(origin_location, destination_location)
        if "error" not in route_result:
            route_result["addresses"] = {
                "origin": {
                    "address": origin_address,
                    "coordinates": origin_location,
                },
                "destination": {
                    "address": destination_address,
                    "coordinates": destination_location,
                },
            }
        return route_result
    except Exception as e:
        return {"error": f"Route planning failed: {str(e)}"}


async def maps_direction_transit_integrated_by_coordinates(
    origin: str, destination: str, city: str, cityd: str
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v3/direction/transit/integrated",
        {
            "origin": origin,
            "destination": destination,
            "city": city,
            "cityd": cityd,
        },
    )
    if err:
        return {"error": err}
    if data.get("status") != "1":
        return {"error": f"Direction Transit Integrated failed: {data.get('info') or data.get('infocode')}"}
    route_data = data.get("route")
    if not isinstance(route_data, dict):
        return {"error": "No route data available"}

    transits = []
    transits_data = route_data.get("transits")
    if isinstance(transits_data, list):
        for transit in transits_data:
            if not isinstance(transit, dict):
                continue
            segments = []
            segments_data = transit.get("segments")
            if isinstance(segments_data, list):
                for segment in segments_data:
                    if not isinstance(segment, dict):
                        continue
                    walking_data = segment.get("walking")
                    if not isinstance(walking_data, dict):
                        walking_data = {}
                    walking_steps = []
                    steps_data = walking_data.get("steps")
                    if isinstance(steps_data, list):
                        for step in steps_data:
                            if isinstance(step, dict):
                                walking_steps.append(
                                    {
                                        "instruction": step.get("instruction"),
                                        "road": step.get("road"),
                                        "distance": step.get("distance"),
                                        "action": step.get("action"),
                                        "assistant_action": step.get("assistant_action"),
                                    }
                                )

                    bus_data = segment.get("bus")
                    if not isinstance(bus_data, dict):
                        bus_data = {}
                    buslines = []
                    buslines_data = bus_data.get("buslines")
                    if isinstance(buslines_data, list):
                        for busline in buslines_data:
                            if not isinstance(busline, dict):
                                continue
                            via_stops = []
                            via_stops_data = busline.get("via_stops")
                            if isinstance(via_stops_data, list):
                                for stop in via_stops_data:
                                    if isinstance(stop, dict):
                                        via_stops.append({"name": stop.get("name")})
                            dep_stop = busline.get("departure_stop")
                            if not isinstance(dep_stop, dict):
                                dep_stop = {}
                            arr_stop = busline.get("arrival_stop")
                            if not isinstance(arr_stop, dict):
                                arr_stop = {}
                            buslines.append(
                                {
                                    "name": busline.get("name"),
                                    "departure_stop": {"name": dep_stop.get("name")},
                                    "arrival_stop": {"name": arr_stop.get("name")},
                                    "distance": busline.get("distance"),
                                    "duration": busline.get("duration"),
                                    "via_stops": via_stops,
                                }
                            )

                    entrance_data = segment.get("entrance")
                    if not isinstance(entrance_data, dict):
                        entrance_data = {}
                    exit_data = segment.get("exit")
                    if not isinstance(exit_data, dict):
                        exit_data = {}
                    railway_data = segment.get("railway")
                    if not isinstance(railway_data, dict):
                        railway_data = {}

                    segments.append(
                        {
                            "walking": {
                                "origin": walking_data.get("origin"),
                                "destination": walking_data.get("destination"),
                                "distance": walking_data.get("distance"),
                                "duration": walking_data.get("duration"),
                                "steps": walking_steps,
                            },
                            "bus": {"buslines": buslines},
                            "entrance": {"name": entrance_data.get("name")},
                            "exit": {"name": exit_data.get("name")},
                            "railway": {
                                "name": railway_data.get("name"),
                                "trip": railway_data.get("trip"),
                            },
                        }
                    )

            transits.append(
                {
                    "duration": transit.get("duration"),
                    "walking_distance": transit.get("walking_distance"),
                    "segments": segments,
                }
            )

    return {
        "route": {
            "origin": route_data.get("origin"),
            "destination": route_data.get("destination"),
            "distance": route_data.get("distance"),
            "transits": transits,
        }
    }


async def maps_direction_transit_integrated_by_address(
    origin_address: str,
    destination_address: str,
    origin_city: str,
    destination_city: str,
) -> Dict[str, Any]:
    try:
        origin_result = await maps_geo(origin_address, origin_city)
        if "error" in origin_result:
            return {"error": f"Failed to geocode origin address: {origin_result['error']}"}
        if not origin_result.get("return") or not origin_result["return"]:
            return {"error": "No geocoding results found for origin address"}
        origin_location = origin_result["return"][0].get("location")
        if not origin_location:
            return {"error": "Could not extract coordinates from origin geocoding result"}

        destination_result = await maps_geo(destination_address, destination_city)
        if "error" in destination_result:
            return {"error": f"Failed to geocode destination address: {destination_result['error']}"}
        if not destination_result.get("return") or not destination_result["return"]:
            return {"error": "No geocoding results found for destination address"}
        destination_location = destination_result["return"][0].get("location")
        if not destination_location:
            return {"error": "Could not extract coordinates from destination geocoding result"}

        route_result = await maps_direction_transit_integrated_by_coordinates(
            origin_location, destination_location, origin_city, destination_city
        )
        if "error" not in route_result:
            route_result["addresses"] = {
                "origin": {
                    "address": origin_address,
                    "coordinates": origin_location,
                },
                "destination": {
                    "address": destination_address,
                    "coordinates": destination_location,
                },
            }
        return route_result
    except Exception as e:
        return {"error": f"Route planning failed: {str(e)}"}


async def maps_distance(
    origins: str, destination: str, type: str = "1"
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v3/distance",
        {"origins": origins, "destination": destination, "type": type},
    )
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Direction Distance failed: {data.get('info') or data.get('infocode')}"}
    results = [
        {
            "origin_id": result.get("origin_id"),
            "dest_id": result.get("dest_id"),
            "distance": result.get("distance"),
            "duration": result.get("duration"),
        }
        for result in data["results"]
    ]
    return {"results": results}


async def maps_text_search(
    keywords: str, city: str = "", citylimit: str = "false"
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v3/place/text",
        {"keywords": keywords, "city": city, "citylimit": citylimit},
    )
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Text Search failed: {data.get('info') or data.get('infocode')}"}
    suggestion_cities = []
    if data.get("suggestion", {}).get("cities"):
        for c in data["suggestion"]["cities"]:
            suggestion_cities.append({"name": c.get("name")})
    pois = [
        {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "address": poi.get("address"),
            "typecode": poi.get("typecode"),
        }
        for poi in data.get("pois", [])
    ]
    return {
        "suggestion": {
            "keywords": data.get("suggestion", {}).get("keywords"),
            "cities": suggestion_cities,
        },
        "pois": pois,
    }


async def maps_around_search(
    location: str, radius: str = "1000", keywords: str = ""
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v3/place/around",
        {"location": location, "radius": radius, "keywords": keywords},
    )
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Around Search failed: {data.get('info') or data.get('infocode')}"}
    pois = [
        {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "address": poi.get("address"),
            "typecode": poi.get("typecode"),
        }
        for poi in data.get("pois", [])
    ]
    return {"pois": pois}


async def maps_search_detail(id: str) -> Dict[str, Any]:
    data, err = await client.get("/v3/place/detail", {"id": id})
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Get poi detail failed: {data.get('info') or data.get('infocode')}"}
    if not data.get("pois"):
        return {"error": "No POI found"}
    poi = data["pois"][0]
    result = {
        "id": poi.get("id"),
        "name": poi.get("name"),
        "location": poi.get("location"),
        "address": poi.get("address"),
        "business_area": poi.get("business_area"),
        "city": poi.get("cityname"),
        "type": poi.get("type"),
        "alias": poi.get("alias"),
    }
    if poi.get("biz_ext"):
        result.update(poi["biz_ext"])
    return result
