import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple
from uuid import uuid4

import numpy as np
import pandapower as pp
import pandas as pd
from pypsdm.models.input.container.raw_grid import RawGridContainer
from pypsdm.models.input.create.grid_elements import (
    create_lines,
    create_nodes,
)


@dataclass
class UuidIdxMaps:
    node: dict[str, int] = field(default_factory=dict)
    line: dict[str, int] = field(default_factory=dict)
    trafo: dict[str, int] = field(default_factory=dict)


def convert_grid(
    grid: pp.pandapowerNet, name: str = "", s_rated_mva: float = 1
) -> Tuple[RawGridContainer, UuidIdxMaps]:

    uuid_idx = UuidIdxMaps()

    net = RawGridContainer.empty()

    nodes = convert_nodes(grid.bus)

    convert_lines(grid)

    for uuid, line in grid.line.data.iterrows():
        idx = convert_line(net, line, uuid_idx.node)
        uuid_idx.line[uuid] = idx  # type: ignore

    for uuid, trafo in grid.trafo.data.iterrows():
        idx = convert_transformer(net, trafo, uuid_idx.node)
        uuid_idx.trafo[uuid] = idx  # type: ignore

    # TODO convert switches

    return net, uuid_idx


def get_default(value, default):
    return value if not pd.isna(value) else default


def convert_nodes(grid):
    df = grid.bus
    geodata = grid.bus_geodata

    def format_geo_position(row):
        if not pd.isna(row["x"]) and not pd.isna(row["y"]):
            return f'{{"type":"Point","coordinates":[{row["x"]},{row["y"]}],"crs":{{"type":"name","properties":{{"name":"EPSG:0"}}}}}}'
        return None

    node_index_uuid_map = {idx: str(uuid4()) for idx in df.index}

    data_dict = {
        "id": df["name"].tolist(),
        "uuid": [node_index_uuid_map[idx] for idx in df.index],
        "geo_position": [
            format_geo_position(geodata.iloc[idx]) for idx in range(len(df))
        ],
        "subnet": [get_default(row.get("zone"), 101) for _, row in df.iterrows()],
        "v_rated": df["vn_kv"].tolist(),
        "v_target": df["vn_kv"].tolist(),
        "volt_lvl": [
            get_default(row.get("vlt_lvl"), row["vn_kv"]) for _, row in df.iterrows()
        ],
        "slack": ["false"] * len(df),
        "operates_from": [get_operation_times(row)[0] for _, row in df.iterrows()],
        "operates_until": [get_operation_times(row)[1] for _, row in df.iterrows()],
    }

    return create_nodes(data_dict), node_index_uuid_map


def get_operation_times(row):
    if row["in_service"]:
        operates_from = None
        operates_until = None
    else:
        operates_from = datetime(1980, 1, 1)
        operates_until = datetime(1980, 1, 2)

    return operates_from, operates_until


def get_node_uuid(nodes, node_id):
    for _, node_data in nodes.data.iterrows():
        if node_data["id"] == node_id:
            return node_data.name
    raise ValueError(f"No matching node found for id {node_id}")


def get_v_target_for_node(nodes, node_uuid):
    return nodes[node_uuid]["v_target"]


def line_param_conversion(c_nf_per_km: float, g_us_per_km: float):
    g_us = g_us_per_km
    f = 50
    b_us = c_nf_per_km * (2 * np.pi * f * 1e-3)

    return g_us, b_us


def convert_line_types(grid, nodes, node_index_uuid_map):

    # Geo Position of Line not implemented

    df = grid.line
    line_types = {}

    for idx, row in df.iterrows():
        line_index_line_type_uuid_map = {idx: str(uuid4()) for idx in df.index}

        # Convert line parameters
        g_us, b_us = line_param_conversion(row["c_nf_per_km"], row["g_us_per_km"])

        # Use index_uuid_map to retrieve UUIDs for from_bus and to_bus
        node_a_uuid = node_index_uuid_map.get(row["from_bus"])
        node_b_uuid = node_index_uuid_map.get(row["to_bus"])

        # Check if the UUIDs were found
        if node_a_uuid is None or node_b_uuid is None:
            raise KeyError(
                f"UUID not found for from_bus {row['from_bus']} or to_bus {row['to_bus']}"
            )

        # Retrieve v_target for node_a and node_b (assuming these functions exist)
        v_target_a = get_v_target_for_node(nodes, node_a_uuid)
        v_target_b = get_v_target_for_node(nodes, node_b_uuid)

        # Ensure v_target_a and v_target_b are the same
        if v_target_a != v_target_b:
            raise ValueError(
                f"v_target mismatch between node_a ({v_target_a}) and node_b ({v_target_b}) for line {row['from_bus']} to {row['to_bus']}"
            )

        uuid = line_index_line_type_uuid_map[idx]

        line_types[uuid] = {
            "uuid": uuid,
            "r": row["r_ohm_per_km"],
            "x": row["x_ohm_per_km"],
            "b": b_us,
            "g": g_us,
            "i_max": row["max_i_ka"] * 1000,  # Convert to amperes
            "id": f"line_type_{idx + 1}",
            "v_rated": v_target_a,
        }

    line_types = pd.DataFrame.from_dict(line_types, orient="index").reset_index()
    line_types = line_types.set_index("uuid").drop(columns="index")
    return line_types, line_index_line_type_uuid_map


def convert_lines(grid, line_index_line_type_uuid_map, node_index_uuid_map):
    df = grid.line
    lines_data = []

    # Create a mapping of line index to UUID
    line_index_uuid_map = {idx: str(uuid4()) for idx in df.index}

    for idx, row in df.iterrows():
        # Retrieve line type UUID
        line_type_uuid = line_index_line_type_uuid_map.get(idx)
        if not line_type_uuid:
            raise ValueError(f"No matching line type found for line {row['name']}")

        # Retrieve node_a and node_b UUIDs based on from_bus and to_bus
        node_a_uuid = node_index_uuid_map.get(row["from_bus"])
        node_b_uuid = node_index_uuid_map.get(row["to_bus"])

        # Collect data for each line
        line_data = {
            "id": row["name"],
            "uuid": line_index_uuid_map[idx],
            "geo_position": None,
            "length": row["length_km"],
            "node_a": node_a_uuid,
            "node_b": node_b_uuid,
            "olm_characteristic": "olm:{(0.0,1.0)}",
            "operates_from": get_operation_times(row)[0],
            "operates_until": get_operation_times(row)[1],
            "operator": None,
            "parallel_devices": row["parallel"],
            "type": line_type_uuid,
        }
        lines_data.append(line_data)

    data_dict = {key: [d[key] for d in lines_data] for key in lines_data[0]}

    return create_lines(data_dict)


def convert_transformer(net: pp.pandapowerNet, trafo_data: pd.Series, uuid_idx: dict):
    trafo_id = trafo_data["id"]
    hv_bus = uuid_idx[trafo_data["node_a"]]
    lv_bus = uuid_idx[trafo_data["node_b"]]
    sn_mva = trafo_data["s_rated"] / 1000
    vn_hv_kv = trafo_data["v_rated_a"]
    vn_lv_kv = trafo_data["v_rated_b"]
    vk_percent, vkr_percent, pfe_kw, i0_percent = trafo_param_conversion(
        float(trafo_data["r_sc"]),
        float(trafo_data["x_sc"]),
        float(trafo_data["s_rated"]),
        float(trafo_data["v_rated_a"]),
        float(trafo_data["g_m"]),
        float(trafo_data["b_m"]),
    )
    if trafo_data["tap_side"]:
        tap_side = "lv"
    else:
        tap_side = "hv"

    tap_neutral = int(trafo_data["tap_neutr"])
    tap_min = int(trafo_data["tap_min"])
    tap_max = int(trafo_data["tap_max"])
    tap_step_degree = float(trafo_data["d_phi"])
    tap_step_percent = float(trafo_data["d_v"])

    return pp.create_transformer_from_parameters(
        net,
        hv_bus=hv_bus,
        lv_bus=lv_bus,
        name=trafo_id,
        sn_mva=sn_mva,
        vn_hv_kv=vn_hv_kv,
        vn_lv_kv=vn_lv_kv,
        vk_percent=vk_percent,
        vkr_percent=vkr_percent,
        pfe_kw=pfe_kw,
        i0_percent=i0_percent,
        tap_side=tap_side,
        tap_neutral=tap_neutral,
        tap_min=tap_min,
        tap_max=tap_max,
        tap_step_degree=tap_step_degree,
        tap_step_percent=tap_step_percent,
    )


def trafo_param_conversion(
    r_sc: float, x_sc: float, s_rated: float, v_rated_a: float, g_m: float, b_m: float
):

    # Circuit impedance
    z_sc = math.sqrt(r_sc**2 + x_sc**2)

    # Rated current on high voltage side in Ampere
    i_rated = s_rated / (math.sqrt(3) * v_rated_a)

    # Short-circuit voltage
    v_imp = z_sc * i_rated * math.sqrt(3) / 1000

    # Short-circuit voltage in percent
    vk_percent = (v_imp / v_rated_a) * 100

    # Real part of relative short-circuit voltage
    vkr_percent = (r_sc / z_sc) * vk_percent

    # Voltage at the main field admittance in V
    v_m = v_rated_a / math.sqrt(3) * 1e3

    # Recalculating Iron losses in kW
    pfe_kw = (g_m * 3 * v_m**2) / 1e12  # converting to kW

    # No load admittance
    y_no_load = math.sqrt(g_m**2 + b_m**2) / 1e9  # in Siemens

    # No load current in Ampere
    i_no_load = y_no_load * v_m

    # No load current in percent
    i0_percent = (i_no_load / i_rated) * 100

    return vk_percent, vkr_percent, pfe_kw, i0_percent
