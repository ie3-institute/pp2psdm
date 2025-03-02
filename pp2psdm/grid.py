import math
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandapower as pp
import pandas as pd
from pypsdm.models.input.connector import Switches
from pypsdm.models.input.container.raw_grid import RawGridContainer
from pypsdm.models.input.create.grid_elements import (
    create_2w_transformers,
    create_lines,
    create_nodes,
)


@dataclass
class UuidIdxMaps:
    node: dict[str, int] = field(default_factory=dict)
    line: dict[str, int] = field(default_factory=dict)
    trafo: dict[str, int] = field(default_factory=dict)


def convert_grid(grid: pp.pandapowerNet) -> RawGridContainer:
    nodes, node_index_uuid_map = convert_nodes(grid)

    lines = convert_lines(grid, nodes, node_index_uuid_map)

    transformers = convert_transformers(grid, node_index_uuid_map)

    # TODO convert switches

    net = RawGridContainer(nodes, lines, transformers, Switches.create_empty)

    return net


def get_default(value, default):
    return value if not pd.isna(value) else default


def convert_nodes(grid):
    df = grid.bus
    geodata = grid.bus_geodata

    def format_geo_position(row, zone):
        if not pd.isna(row["x"]) and not pd.isna(row["y"]):
            if zone is None:
                geo_name = "EPSG:0"
            else:
                geo_name = str(zone)
            return f'{{"type":"Point","coordinates":[{row["x"]},{row["y"]}],"crs":{{"type":"name","properties":{{"name":"' + geo_name + '"}}}'
        return None

    node_index_uuid_map = {idx: str(uuid4()) for idx in df.index}

    # get slack node
    ext_grid_df = grid.ext_grid

    if len(ext_grid_df) != 1:
        raise ValueError("PSDM grid supports just one slack node!")
    else:
        slack_bus = ext_grid_df.loc[0, "bus"]
        series_slack = pd.DataFrame({"slack": "False"}, index=df.index)
        series_slack.loc[slack_bus, "slack"] = "True"

    data_dict = {
        "id": df["name"].tolist(),
        "uuid": [node_index_uuid_map[idx] for idx in df.index],
        "geo_position": [
            format_geo_position(geodata.iloc[idx], df["zone"].iloc[idx])
            for idx in range(len(df))
        ],
        "subnet": [get_default(row.get("zone"), 101) for _, row in df.iterrows()],
        "v_rated": df["vn_kv"].tolist(),
        "v_target": df["vn_kv"].tolist(),
        "volt_lvl": [
            get_default(row.get("vlt_lvl"), row["vn_kv"]) for _, row in df.iterrows()
        ],
        "slack": series_slack["slack"],
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


def convert_lines(grid, nodes, node_index_uuid_map):
    df = grid.line
    lines_data = []
    line_index_uuid_map = {idx: str(uuid4()) for idx in df.index}

    for idx, row in df.iterrows():

        # Retrieve node_a and node_b UUIDs based on from_bus and to_bus
        node_a_uuid = node_index_uuid_map.get(row["from_bus"])
        node_b_uuid = node_index_uuid_map.get(row["to_bus"])

        # Check if the UUIDs were found
        if node_a_uuid is None or node_b_uuid is None:
            raise KeyError(
                f"UUID not found for from_bus {row['from_bus']} or to_bus {row['to_bus']}"
            )

        # Retrieve v_target for node_a and node_b
        v_target_a = get_v_target_for_node(nodes, node_a_uuid)
        v_target_b = get_v_target_for_node(nodes, node_b_uuid)

        # Ensure v_target_a and v_target_b are the same
        if v_target_a != v_target_b:
            raise ValueError(
                f"v_target mismatch between node_a ({v_target_a}) and node_b ({v_target_b}) for line {row['from_bus']} to {row['to_bus']}"
            )

        # Convert line parameters
        g_us, b_us = line_param_conversion(row["c_nf_per_km"], row["g_us_per_km"])

        # Collect data for each line
        line_data = {
            "id": row["name"],
            "uuid": line_index_uuid_map[idx],
            "geo_position": None,
            "length": row["length_km"],
            "node_a": node_a_uuid,
            "node_b": node_b_uuid,
            "r": row["r_ohm_per_km"],
            "x": row["x_ohm_per_km"],
            "b": b_us,
            "g": g_us,
            "i_max": row["max_i_ka"] * 1000,  # Convert to amperes
            "v_rated": v_target_a,
            "olm_characteristic": "olm:{(0.0,1.0)}",
            "operates_from": get_operation_times(row)[0],
            "operates_until": get_operation_times(row)[1],
            "operator": None,
            "parallel_devices": row["parallel"],
        }
        lines_data.append(line_data)

    data_dict = {key: [d[key] for d in lines_data] for key in lines_data[0]}

    return create_lines(data_dict)


def convert_transformers(grid, node_index_uuid_map):
    df = grid.trafo
    transformers_data = []
    trafo_index_uuid_map = {idx: str(uuid4()) for idx in df.index}

    for idx, row in df.iterrows():
        # Convert trafo parameters
        rSc, xSc, gM, bM = trafo_param_conversion(
            row["vk_percent"],
            row["vkr_percent"],
            row["pfe_kw"],
            row["i0_percent"],
            row["vn_hv_kv"],
            row["sn_mva"],
        )

        tap_side = True if row["tap_side"] == "hv" else False

        # Retrieve node_a and node_b UUIDs based on hv_bus and lv_bus
        node_a_uuid = node_index_uuid_map.get(row["hv_bus"])
        node_b_uuid = node_index_uuid_map.get(row["lv_bus"])

        autoTap = True if row["autoTap"] == 1 else False

        trafo_data = {
            "id": row["name"],
            "uuid": trafo_index_uuid_map[idx],
            "auto_tap": autoTap,
            "node_a": node_a_uuid,
            "node_b": node_b_uuid,
            "b_m": bM,
            "d_phi": row["tap_step_degree"],
            "d_v": row["tap_step_percent"],
            "g_m": gM,
            "r_sc": rSc,
            "s_rated": row["sn_mva"] * 1000,  # Convert to kVA
            "tap_max": row["tap_max"],
            "tap_min": row["tap_min"],
            "tap_neutr": row["tap_neutral"],
            "tap_side": tap_side,
            "v_rated_a": row["vn_hv_kv"],
            "v_rated_b": row["vn_lv_kv"],
            "x_sc": xSc,
            "operates_from": get_operation_times(row)[0],
            "operates_until": get_operation_times(row)[1],
            "operator": None,
            "parallel_devices": row["parallel"],
            "tap_pos": row["tap_pos"],
        }
        transformers_data.append(trafo_data)

    data_dict = {
        key: [d[key] for d in transformers_data] for key in transformers_data[0]
    }

    return create_2w_transformers(data_dict)


def trafo_param_conversion(
    vk_percent, vkr_percent, pfe_kw, i0_percent, vn_hv_kv, sn_mva
):
    # Rated current on high voltage side in Ampere
    i_rated = sn_mva * 1e6 / (math.sqrt(3) * vn_hv_kv * 1e3)

    # Voltage at the main field admittance in V
    vM = vn_hv_kv * 1e3 / math.sqrt(3)

    # No load current in Ampere
    iNoLoad = (i0_percent / 100) * i_rated

    # No load admittance in Ohm
    yNoLoad = iNoLoad / vM

    # No load conductance in Siemens
    gM = pfe_kw * 1e3 / ((vn_hv_kv * 1e3) ** 2)
    # Convert into nano Siemens for psdm
    gM_nS = gM * 1e9

    # No load susceptance in Siemens
    bM = math.sqrt(yNoLoad**2 - gM**2)
    # Convert into nano Siemens for psdm and correct sign
    bm_uS_directed = bM * 1e9 * (-1)

    # Copper losses at short circuit in Watt
    pCU = ((vkr_percent * 1e-3 / 100) * sn_mva * 1e6) * 1e3

    # Resistance at short circuit in Ohm
    rSc = pCU / (3 * i_rated**2)

    # Reference Impedance in Ohm
    z_ref = (vn_hv_kv * 1e3) ** 2 / (sn_mva * 1e6)

    # Short circuit impedance in Ohm
    zSc = (vk_percent / 100) * z_ref

    # Short circuit reactance in Ohm
    xSc = math.sqrt(zSc * zSc - rSc * rSc)

    return rSc, xSc, gM_nS, bm_uS_directed
