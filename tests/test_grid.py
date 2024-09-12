import math

import numpy as np
import pytest

from pp2psdm.grid import (  # , convert_transformers
    convert_grid,
    convert_line_types,
    convert_lines,
    convert_nodes,
    convert_transformer_types,
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
    s_mva = 5
    name = "test_grid"
    net, uuid_idx = convert_grid(input, name=name, s_rated_mva=s_mva)
    assert net.sn_mva == s_mva
    assert net.name == name
    assert len(net.bus) == len(input.nodes.data)
    for uuid, idx in uuid_idx.node.items():
        assert net.bus.name.iloc[idx] == input.nodes.data.loc[uuid]["id"]
    assert len(net.line) == len(input.lines.data)
    for uuid, idx in uuid_idx.line.items():
        assert net.line.name.iloc[idx] == input.lines.data.loc[uuid]["id"]
    assert len(net.trafo) == len(input.transformers_2_w.data)
    for uuid, idx in uuid_idx.trafo.items():
        assert net.trafo.name.iloc[idx] == input.transformers_2_w.data.loc[uuid]["id"]


def test_nodes_conversion(input_data):
    _, input = input_data
    len = input.bus.shape[0]
    net, _ = convert_nodes(input)
    for i in range(len):
        assert net.data.iloc[i]["id"] == input.bus.iloc[i]["name"]
        assert net.data.iloc[i]["v_rated"] == input.bus.iloc[i]["vn_kv"]
        assert net.data.iloc[i]["subnet"] == 101
        assert net.data.iloc[i]["operates_from"] == None
        assert net.data.iloc[i]["operates_until"] == None
        assert net.data.iloc[i]["operator"] == None
        assert (
            net.data.iloc[i]["geo_position"]
            == f'{{"type":"Point","coordinates":[{input.bus_geodata.iloc[i]["x"]},{input.bus_geodata.iloc[i]["y"]}],"crs":{{"type":"name","properties":{{"name":"EPSG:0"}}}}}}'
        )


def test_line_types_conversion(input_data):
    expected, input = input_data
    len = input.line.shape[0]
    nodes, node_index_uuid_map = convert_nodes(input)
    net, _ = convert_line_types(input, nodes, node_index_uuid_map)

    for i in range(len):
        assert (
            net.iloc[i]["v_rated"]
            == nodes.get(node_index_uuid_map.get(input.line.iloc[i]["from_bus"]))[
                "v_target"
            ]
        )
        assert (
            net.iloc[i]["v_rated"]
            == nodes.get(node_index_uuid_map.get(input.line.iloc[i]["to_bus"]))[
                "v_target"
            ]
        )
        assert math.isclose(net.iloc[i]["r"], input.line.iloc[i]["r_ohm_per_km"])
        assert math.isclose(net.iloc[i]["x"], input.line.iloc[i]["x_ohm_per_km"])
        assert math.isclose(
            net.iloc[i]["b"], input.line.iloc[i]["c_nf_per_km"] * 2 * np.pi * 50 * 1e-3
        )
        assert math.isclose(net.iloc[i]["g"], input.line.iloc[i]["g_us_per_km"])
        assert math.isclose(net.iloc[i]["i_max"], input.line.iloc[i]["max_i_ka"] * 1000)


def test_lines_conversion(input_data):
    expected, input = input_data
    len = input.line.shape[0]
    nodes, node_index_uuid_map = convert_nodes(input)
    line_types, line_index_line_type_uuid_map = convert_line_types(
        input, nodes, node_index_uuid_map
    )

    net = convert_lines(input, line_index_line_type_uuid_map, node_index_uuid_map)

    for i in range(len):
        assert net.data.iloc[i]["id"] == input.line.iloc[i]["name"]
        assert net.data.iloc[i]["node_a"] == node_index_uuid_map.get(
            input.line.iloc[i]["from_bus"]
        )
        assert net.data.iloc[i]["node_b"] == node_index_uuid_map.get(
            input.line.iloc[i]["to_bus"]
        )
        assert net.data.iloc[i]["length"] == input.line.iloc[i]["length_km"]
        assert net.data.iloc[i]["olm_characteristic"] == "olm:{(0.0,1.0)}"
        assert net.data.iloc[i]["operates_from"] == None
        assert net.data.iloc[i]["operates_until"] == None
        assert net.data.iloc[i]["operator"] == None
        assert net.data.iloc[i]["parallel_devices"] == input.line.iloc[i]["parallel"]
        assert net.data.iloc[i]["type"] == line_index_line_type_uuid_map.get(i)


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


def test_tranfo_type_conversion(input_data):
    expected, input = input_data
    len = input.trafo.shape[0]
    net, _ = convert_transformer_types(input)

    for i in range(len):
        rSc, xSc, gM, bM = trafo_parameter_test_conversion(
            input.trafo.iloc[i]["vk_percent"],
            input.trafo.iloc[i]["vkr_percent"],
            input.trafo.iloc[i]["pfe_kw"],
            input.trafo.iloc[i]["i0_percent"],
            input.trafo.iloc[i]["vn_hv_kv"],
            input.trafo.iloc[i]["sn_mva"],
        )

        tap_side = True if input.trafo.iloc[i]["tap_side"] == "hv" else False

        assert net.iloc[i]["tap_max"] == input.trafo.iloc[i]["tap_max"]
        assert net.iloc[i]["tap_min"] == input.trafo.iloc[i]["tap_min"]
        assert net.iloc[i]["tap_neutr"] == input.trafo.iloc[i]["tap_neutral"]
        assert net.iloc[i]["tap_side"] == tap_side
        assert net.iloc[i]["v_rated_a"] == input.trafo.iloc[i]["vn_hv_kv"]
        assert net.iloc[i]["v_rated_b"] == input.trafo.iloc[i]["vn_lv_kv"]
        assert net.iloc[i]["d_phi"] == input.trafo.iloc[i]["tap_step_degree"]
        assert net.iloc[i]["d_v"] == input.trafo.iloc[i]["tap_step_percent"]
        assert math.isclose(net.iloc[i]["r_sc"], rSc)
        assert math.isclose(net.iloc[i]["x_sc"], xSc)
        assert math.isclose(net.iloc[i]["g_m"], gM)
        assert math.isclose(net.iloc[i]["b_m"], bM)
        assert net.iloc[i]["s_rated"] == input.trafo.iloc[i]["sn_mva"] * 1000


def test_trafo_conversion(input_data):
    expected, input = input_data
    len = input.trafo.shape[0]
    _, node_index_uuid_map = convert_nodes(input)
    trafo_types, trafo_index_trafo_type_uuid_map = convert_transformer_types(input)

    net = convert_transformers(
        input, trafo_index_trafo_type_uuid_map, node_index_uuid_map
    )

    for i in range(len):
        assert net.data.iloc[i]["id"] == input.trafo.iloc[i]["name"]
        assert net.data.iloc[i]["auto_tap"] == input.trafo.iloc[i]["autoTap"]
        assert net.data.iloc[i]["node_a"] == node_index_uuid_map.get(
            input.trafo.iloc[i]["hv_bus"]
        )
        assert net.data.iloc[i]["node_b"] == node_index_uuid_map.get(
            input.trafo.iloc[i]["lv_bus"]
        )
        assert net.data.iloc[i]["operates_from"] == None
        assert net.data.iloc[i]["operates_until"] == None
        assert net.data.iloc[i]["operator"] == None
        assert net.data.iloc[i]["parallel_devices"] == input.trafo.iloc[i]["parallel"]
        assert net.data.iloc[i]["tap_pos"] == input.trafo.iloc[i]["tap_pos"]
        assert net.data.iloc[i]["type"] == trafo_index_trafo_type_uuid_map.get(i)
