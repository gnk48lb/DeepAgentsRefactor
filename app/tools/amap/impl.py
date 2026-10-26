import re
from typing import Any, Awaitable, Callable, Dict, Optional

from app.tools.amap import client

_COORD_RE = re.compile(r"^-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?$")
_MAX_GEOCODED_ORIGINS = 10  # maps_distance 一次最多自动地理编码的文字地址个数（防止打爆高德 QPS）


def normalize_coordinates(val: Any) -> Optional[str]:
    """'经度,纬度' → 规整后的字符串：去空白、全角逗号转半角、小数位超过 6 位时四舍五入（高德要求 ≤6 位）。
    不是坐标返回 None。"""
    if not isinstance(val, str):
        return None
    s = re.sub(r"\s+", "", val).replace("，", ",")
    if not _COORD_RE.match(s):
        return None
    out = []
    for p in s.split(","):
        if "." in p and len(p.split(".")[1]) > 6:
            p = f"{float(p):.6f}"
        out.append(p)
    return ",".join(out)


def is_coordinates(val: Any) -> bool:
    return normalize_coordinates(val) is not None


def _city_key(cand: dict) -> tuple:
    p, c = cand.get("province"), cand.get("city")  # 高德用 [] 表示空值
    return (p if isinstance(p, str) else "", c if isinstance(c, str) else "")


def _city_label(key: tuple) -> str:
    p, c = key
    return p + c if c and c != p else (p or c)


async def _geocode(text: str, city: Optional[str], label: str) -> tuple[Optional[dict], Optional[str]]:
    """文字地址 → 第一个候选。没给城市且候选横跨多个城市时拒绝（不要悄悄取第一个）。返回 (候选, 错误信息)。"""
    city = (city or "").strip() or None
    geo = await maps_geo(text, city)
    if "error" in geo:
        return None, f"Failed to geocode {label} '{text}': {geo['error']}"
    cands = geo.get("return") or []
    if not cands:
        return None, f"No geocoding results found for {label} '{text}'"
    if city is None:
        keys = list(dict.fromkeys(_city_key(c) for c in cands))
        if len(keys) > 1:
            names = "、".join(_city_label(k) for k in keys[:4])
            return None, (
                f"{label} '{text}' 有歧义：匹配到多个城市（{names}）。"
                f"请在地址里写明城市，或填写对应的城市参数后重试。"
            )
    if not cands[0].get("location"):
        return None, f"Could not extract coordinates for {label} '{text}'"
    return cands[0], None


async def _resolve_location(
    val: str, city: Optional[str], label: str
) -> tuple[Optional[str], Optional[str], bool]:
    """坐标直接规整，文字地址自动地理编码。返回 (坐标, 错误信息, 是否做了地理编码)。"""
    coord = normalize_coordinates(val)
    if coord:
        return coord, None, False
    text = (val or "").strip() if isinstance(val, str) else ""
    if not text:
        return None, f"{label} 不能为空", False
    cand, err = await _geocode(text, city, label)
    if err:
        return None, err, True
    return cand["location"], None, True


async def maps_regeocode(location: str) -> Dict[str, Any]:
    loc = normalize_coordinates(location) or location
    data, err = await client.get("/v3/geocode/regeo", {"location": loc})
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
    code = (city or "").strip()
    if not re.fullmatch(r"\d{6}", code):  # 高德天气接口官方只认 adcode：名称先自动解析
        cand, err = await _geocode(code, None, "city")
        if err:
            return {"error": err}
        code = cand.get("adcode")
        if not isinstance(code, str) or not code:
            return {"error": f"Could not resolve adcode for city '{city}'"}
    data, err = await client.get("/v3/weather/weatherInfo", {"city": code, "extensions": "all"})
    if err:
        return {"error": err}
    if data["status"] != "1":
        return {"error": f"Get weather failed: {data.get('info') or data.get('infocode')}"}
    forecasts = data.get("forecasts", [])
    if not forecasts:
        return {"error": "No forecast data available"}
    return {"city": forecasts[0]["city"], "forecasts": forecasts[0]["casts"]}


async def _bicycling_by_coordinates(
    origin: str, destination: str
) -> Dict[str, Any]:
    data, err = await client.get(
        "/v4/direction/bicycling",
        {
            "origin": origin,
            "destination": destination,
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


async def _walking_by_coordinates(
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


async def _driving_by_coordinates(
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


async def _transit_by_coordinates(
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


async def _route(
    coord_fn: Callable[[str, str], Awaitable[Dict[str, Any]]],
    origin: str,
    destination: str,
    origin_city: Optional[str],
    destination_city: Optional[str],
) -> Dict[str, Any]:
    try:
        o_loc, err, o_geo = await _resolve_location(origin, origin_city, "origin")
        if err:
            return {"error": err}
        d_loc, err, d_geo = await _resolve_location(destination, destination_city, "destination")
        if err:
            return {"error": err}
        result = await coord_fn(o_loc, d_loc)
        if "error" not in result and (o_geo or d_geo):
            addresses: Dict[str, Any] = {}
            if o_geo:
                addresses["origin"] = {"address": origin, "coordinates": o_loc}
            if d_geo:
                addresses["destination"] = {"address": destination, "coordinates": d_loc}
            result["addresses"] = addresses
        return result
    except Exception as e:
        return {"error": f"Route planning failed: {e}"}


async def maps_route_walking(
    origin: str, destination: str, origin_city: Optional[str] = None, destination_city: Optional[str] = None
) -> Dict[str, Any]:
    return await _route(_walking_by_coordinates, origin, destination, origin_city, destination_city)


async def maps_route_bicycling(
    origin: str, destination: str, origin_city: Optional[str] = None, destination_city: Optional[str] = None
) -> Dict[str, Any]:
    return await _route(_bicycling_by_coordinates, origin, destination, origin_city, destination_city)


async def maps_route_driving(
    origin: str, destination: str, origin_city: Optional[str] = None, destination_city: Optional[str] = None
) -> Dict[str, Any]:
    return await _route(_driving_by_coordinates, origin, destination, origin_city, destination_city)


async def maps_route_transit(
    origin: str, destination: str, origin_city: str = "", destination_city: Optional[str] = None
) -> Dict[str, Any]:
    o_city = (origin_city or "").strip()
    if not o_city:
        return {"error": "origin_city 不能为空：公共交通规划必须提供起点所在城市（跨城还要提供 destination_city）"}
    d_city = (destination_city or "").strip() or o_city

    async def _fn(o: str, d: str) -> Dict[str, Any]:
        return await _transit_by_coordinates(o, d, o_city, d_city)

    return await _route(_fn, origin, destination, o_city, d_city)


async def maps_distance(
    origins: str, destination: str, type: str = "1", city: Optional[str] = None
) -> Dict[str, Any]:
    parts = [p.strip() for p in str(origins).split("|") if p.strip()]
    if not parts:
        return {"error": "origins 不能为空"}
    if sum(1 for p in parts if not is_coordinates(p)) > _MAX_GEOCODED_ORIGINS:
        return {"error": f"origins 里的文字地址最多 {_MAX_GEOCODED_ORIGINS} 个，请改传坐标或分批查询"}
    resolved = []
    for i, part in enumerate(parts, 1):
        loc, err, _ = await _resolve_location(part, city, f"origin #{i}")
        if err:
            return {"error": err}
        resolved.append(loc)
    dest_loc, err, _ = await _resolve_location(destination, city, "destination")
    if err:
        return {"error": err}

    data, http_err = await client.get(
        "/v3/distance",
        {"origins": "|".join(resolved), "destination": dest_loc, "type": str(type)},
    )
    if http_err:
        return {"error": http_err}
    if data["status"] != "1":
        return {"error": f"Direction Distance failed: {data.get('info') or data.get('infocode')}"}
    results = []
    for r in data["results"]:
        try:
            idx = int(r.get("origin_id")) - 1
        except (TypeError, ValueError):
            idx = -1
        item = {
            "origin_id": r.get("origin_id"),
            "origin": parts[idx] if 0 <= idx < len(parts) else None,
            "distance": r.get("distance"),
            "duration": r.get("duration"),
        }
        if r.get("info"):  # 单个起点算不出来时高德会在该项里给 info/code，别把它显示成 0 米
            item["info"] = r["info"]
            item["code"] = r.get("code")
        results.append(item)
    return {"results": results}


async def maps_around_search(
    location: str, radius: int | str = 1000, keywords: str = "", city: Optional[str] = None
) -> Dict[str, Any]:
    loc, err, _ = await _resolve_location(location, city, "location")
    if err:
        return {"error": err}
    data, http_err = await client.get(
        "/v3/place/around",
        {"location": loc, "radius": str(int(float(radius))), "keywords": keywords},
    )
    if http_err:
        return {"error": http_err}
    if data["status"] != "1":
        return {"error": f"Around Search failed: {data.get('info') or data.get('infocode')}"}
    pois = [
        {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "address": poi.get("address"),
            "location": poi.get("location"),
            "distance": poi.get("distance"),
            "type": poi.get("type"),
        }
        for poi in data.get("pois", [])
    ]
    return {"pois": pois}


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
            "location": poi.get("location"),
            "type": poi.get("type"),
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
        "tel": poi.get("tel"),
        "business_area": poi.get("business_area"),
        "city": poi.get("cityname"),
        "type": poi.get("type"),
        "alias": poi.get("alias"),
    }
    if poi.get("biz_ext"):
        result.update(poi["biz_ext"])
    return result
