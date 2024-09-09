import math
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandapower as pp
import pandas as pd
from pypsdm.models.input.container.raw_grid import RawGridContainer
from pypsdm.models.input.create.participants import create_data


@dataclass
class UuidIdxMaps:
    node: dict[str, int] = field(default_factory=dict)
    line: dict[str, int] = field(default_factory=dict)
    trafo: dict[str, int] = field(default_factory=dict)


def convert_grid(
    grid: pp.pandapowerNet, name: str = "", s_rated_mva: float = 1
) -> Tuple[RawGridContainer, UuidIdxMaps]:

    uuid_idx = UuidIdxMaps()

    net = RawGridContainer.empty(
    )

    for uuid, bus in grid.bus.data.iterrows():
        idx = convert_node(net, bus)
        uuid_idx.bus[uuid] = idx  # type: ignore

    for uuid, line in grid.line.data.iterrows():
        idx = convert_line(net, line, uuid_idx.node)
        uuid_idx.line[uuid] = idx  # type: ignore

    for uuid, trafo in grid.trafo.data.iterrows():
        idx = convert_transformer(net, trafo, uuid_idx.node)
        uuid_idx.trafo[uuid] = idx  # type: ignore

    # TODO convert switches

    return net, uuid_idx


def convert_node(net: pp.pandapowerNet, node_data: pd.Series):
    node_id = node_data["id"]
    vn_kv = node_data["v_rated"]
    xy_coords = [node_data["longitude"], node_data["latitude"]]
    subnet = node_data["subnet"]
    node_type = "b"
    return create_nodes_data(

   )


def convert_line(net: pp.pandapowerNet, line_data: pd.Series, uuid_idx: dict):
    line_id = line_data["id"]
    from_bus = uuid_idx[line_data["node_a"]]
    to_bus = uuid_idx[line_data["node_b"]]
    length_km = line_data["length"]
    r_ohm_per_km = line_data["r"]
    x_ohm_per_km = line_data["x"]
    g_us_per_km, c_nf_per_km = line_param_conversion(
        float(line_data["b"]), float(line_data["g"])
    )
    max_i_ka = line_data["i_max"] / 1000
    return pp.create_line_from_parameters(
        net,
        name=line_id,
        from_bus=from_bus,
        to_bus=to_bus,
        length_km=length_km,
        r_ohm_per_km=r_ohm_per_km,
        x_ohm_per_km=x_ohm_per_km,
        c_nf_per_km=c_nf_per_km,
        max_i_ka=max_i_ka,
        g_us_per_km=g_us_per_km,
    )


def line_param_conversion(b_us: float, g_us: float):
    g_us_per_km = g_us
    f = 50
    c_nf_per_km = b_us / (2 * np.pi * f * 1e-3)
    return g_us_per_km, c_nf_per_km


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
