import math

import numpy as np
import pytest

from pp2psdm.grid import (  # , convert_transformers
    convert_grid,
    convert_lines,
    convert_nodes,
    convert_transformers,
)
from tests.utils import read_psdm_lv, read_sb_lv


@pytest.fixture
def input_data():
    expected = read_psdm_lv()
    input = read_sb_lv()
    return expected, input


def test_convert_grid(input_data):
    _, input = input_data
    net = convert_grid(input)
    assert len(net.nodes) == len(input.bus)
    assert len(net.lines) == len(input.line)
    assert len(net.transformers_2_w) == len(input.trafo)


def test_nodes_conversion(input_data):
    _, input = input_data
    len = input.bus.shape[0]
    net, _ = convert_nodes(input)

    for i in range(len):
        coord_string = f'{{"type":"Point","coordinates":[{input.bus_geodata.iloc[i]["x"]},{input.bus_geodata.iloc[i]["y"]}],"crs":{{"type":"name","properties":{{"name":"EPSG:0"}}}}}}'
        assert net.data.iloc[i]["id"] == input.bus.iloc[i]["name"]
        assert net.data.iloc[i]["v_rated"] == input.bus.iloc[i]["vn_kv"]
        assert net.data.iloc[i]["subnet"] == 101
        assert net.data.iloc[i]["operates_from"] is None
        assert net.data.iloc[i]["operates_until"] is None
        assert net.data.iloc[i]["operator"] is None
        assert net.data.iloc[i]["geo_position"] == coord_string


def test_lines_conversion(input_data):
    expected, input = input_data
    len = input.line.shape[0]
    nodes, node_index_uuid_map = convert_nodes(input)

    net = convert_lines(input, nodes, node_index_uuid_map)

    for i in range(len):

        node_from_bus = node_index_uuid_map.get(input.line.iloc[i]["from_bus"])
        assert net.data.iloc[i]["v_rated"] == nodes.get(node_from_bus)["v_target"]
        node_to_bus = node_index_uuid_map.get(input.line.iloc[i]["to_bus"])
        assert net.data.iloc[i]["v_rated"] == nodes.get(node_to_bus)["v_target"]
        assert math.isclose(net.data.iloc[i]["r"], input.line.iloc[i]["r_ohm_per_km"])
        assert math.isclose(net.data.iloc[i]["x"], input.line.iloc[i]["x_ohm_per_km"])
        assert math.isclose(
            net.data.iloc[i]["b"],
            input.line.iloc[i]["c_nf_per_km"] * 2 * np.pi * 50 * 1e-3,
        )
        assert math.isclose(net.data.iloc[i]["g"], input.line.iloc[i]["g_us_per_km"])
        assert math.isclose(
            net.data.iloc[i]["i_max"], input.line.iloc[i]["max_i_ka"] * 1000
        )
        assert net.data.iloc[i]["id"] == input.line.iloc[i]["name"]
        assert net.data.iloc[i]["node_a"] == node_index_uuid_map.get(
            input.line.iloc[i]["from_bus"]
        )
        assert net.data.iloc[i]["node_b"] == node_index_uuid_map.get(
            input.line.iloc[i]["to_bus"]
        )
        assert net.data.iloc[i]["length"] == input.line.iloc[i]["length_km"]
        assert net.data.iloc[i]["olm_characteristic"] == "olm:{(0.0,1.0)}"
        assert net.data.iloc[i]["operates_from"] is None
        assert net.data.iloc[i]["operates_until"] is None
        assert net.data.iloc[i]["operator"] is None
        assert net.data.iloc[i]["parallel_devices"] == input.line.iloc[i]["parallel"]


def trafo_parameter_test_conversion(
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

    # Reference current in Ampere
    i_ref = sn_mva * 1e6 / (math.sqrt(3) * vn_hv_kv * 1e3)

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
    rSc = pCU / (3 * i_ref**2)

    # Reference Impedance in Ohm
    z_ref = (vn_hv_kv * 1e3) ** 2 / (sn_mva * 1e6)

    # Short circuit impedance in Ohm
    zSc = (vk_percent / 100) * z_ref

    # Short circuit reactance in Ohm
    xSc = math.sqrt(zSc * zSc - rSc * rSc)
    return (rSc, xSc, gM_nS, bm_uS_directed)


def test_trafo_conversion(input_data):
    expected, input = input_data
    len = input.trafo.shape[0]
    _, node_index_uuid_map = convert_nodes(input)

    net = convert_transformers(input, node_index_uuid_map)

    for i in range(len):
        rSc, xSc, gM, bM = trafo_parameter_test_conversion(
            input.trafo.iloc[i]["vk_percent"],
            input.trafo.iloc[i]["vkr_percent"],
            input.trafo.iloc[i]["pfe_kw"],
            input.trafo.iloc[i]["i0_percent"],
            input.trafo.iloc[i]["vn_hv_kv"],
            input.trafo.iloc[i]["sn_mva"],
        )

        tap_side = False if input.trafo.iloc[i]["tap_side"] == "hv" else True

        assert net.data.iloc[i]["tap_max"] == input.trafo.iloc[i]["tap_max"]
        assert net.data.iloc[i]["tap_min"] == input.trafo.iloc[i]["tap_min"]
        assert net.data.iloc[i]["tap_neutr"] == input.trafo.iloc[i]["tap_neutral"]
        assert net.data.iloc[i]["tap_side"] == tap_side
        assert net.data.iloc[i]["v_rated_a"] == input.trafo.iloc[i]["vn_hv_kv"]
        assert net.data.iloc[i]["v_rated_b"] == input.trafo.iloc[i]["vn_lv_kv"]
        assert net.data.iloc[i]["d_phi"] == input.trafo.iloc[i]["tap_step_degree"]
        assert net.data.iloc[i]["d_v"] == input.trafo.iloc[i]["tap_step_percent"]
        assert math.isclose(net.data.iloc[i]["r_sc"], rSc)
        assert math.isclose(net.data.iloc[i]["x_sc"], xSc)
        assert math.isclose(net.data.iloc[i]["g_m"], gM)
        assert math.isclose(net.data.iloc[i]["b_m"], bM)
        assert net.data.iloc[i]["s_rated"] == input.trafo.iloc[i]["sn_mva"] * 1000
        assert net.data.iloc[i]["id"] == input.trafo.iloc[i]["name"]
        assert net.data.iloc[i]["auto_tap"] == input.trafo.iloc[i]["autoTap"]
        assert net.data.iloc[i]["node_a"] == node_index_uuid_map.get(
            input.trafo.iloc[i]["hv_bus"]
        )
        assert net.data.iloc[i]["node_b"] == node_index_uuid_map.get(
            input.trafo.iloc[i]["lv_bus"]
        )
        assert net.data.iloc[i]["operates_from"] is None
        assert net.data.iloc[i]["operates_until"] is None
        assert net.data.iloc[i]["operator"] is None
        assert net.data.iloc[i]["parallel_devices"] == input.trafo.iloc[i]["parallel"]
        assert net.data.iloc[i]["tap_pos"] == input.trafo.iloc[i]["tap_pos"]
