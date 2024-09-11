import math

import pandapower as pp
import pytest
import numpy as np
from pp2psdm.grid import convert_grid, convert_line_types, convert_lines, convert_nodes#, convert_transformers
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
        assert net.data.iloc[i]['id'] == input.bus.iloc[i]["name"]
        assert net.data.iloc[i]["v_rated"] == input.bus.iloc[i]["vn_kv"]
        assert net.data.iloc[i]["subnet"] == 101
        assert net.data.iloc[i]["operates_from"] == None
        assert net.data.iloc[i]["operates_until"] == None
        assert net.data.iloc[i]["operator"] == None
        assert net.data.iloc[i]["geo_position"] == f'{{"type":"Point","coordinates":[{input.bus_geodata.iloc[i]["x"]},{input.bus_geodata.iloc[i]["y"]}],"crs":{{"type":"name","properties":{{"name":"EPSG:0"}}}}}}'


def test_line_types_conversion(input_data):
    expected, input = input_data
    len = input.line.shape[0]
    nodes, node_index_uuid_map = convert_nodes(input)
    net, _ = convert_line_types(input, nodes, index_uuid_map)

    for i in range(len):
        assert net.iloc[i]["v_rated"] == nodes.get(node_index_uuid_map.get(input.line.iloc[i]["from_bus"]))['v_target']
        assert net.iloc[i]["v_rated"] == nodes.get(node_index_uuid_map.get(input.line.iloc[i]["to_bus"]))['v_target']
        assert math.isclose(net.iloc[i]["r"], input.line.iloc[i]["r_ohm_per_km"])
        assert math.isclose(net.iloc[i]["x"], input.line.iloc[i]["x_ohm_per_km"])
        assert math.isclose(net.iloc[i]["b"], input.line.iloc[i]["c_nf_per_km"] * 2 * np.pi * 50 * 1e-3)
        assert math.isclose(net.iloc[i]["g"], input.line.iloc[i]["g_us_per_km"])
        assert math.isclose(net.iloc[i]["i_max"], input.line.iloc[i]["max_i_ka"] * 1000)



def test_lines_conversion(input_data):
    expected, input = input_data
    len = input.line.shape[0]
    nodes, node_index_uuid_map = convert_nodes(input)
    line_types, line_index_line_type_uuid_map = convert_line_types(input, nodes, node_index_uuid_map)

    net = convert_lines(input, line_index_line_type_uuid_map, node_index_uuid_map)

    for i in range(len):
        assert net.data.iloc[i]['id'] == input.line.iloc[i]["name"]
        assert net.data.iloc[i]["node_a"] == node_index_uuid_map.get(input.line.iloc[i]["from_bus"])
        assert net.data.iloc[i]["node_b"] == node_index_uuid_map.get(input.line.iloc[i]["to_bus"])
        assert net.data.iloc[i]["length"] == input.line.iloc[i]["length_km"]
        assert net.data.iloc[i]["olm_characteristic"] == "olm:{(0.0,1.0)}"
        assert net.data.iloc[i]["operates_from"] == None
        assert net.data.iloc[i]["operates_until"] == None
        assert net.data.iloc[i]["operator"] == None
        assert net.data.iloc[i]["parallel_devices"] == input.line.iloc[i]["parallel"]
        assert net.data.iloc[i]["type"] == line_index_line_type_uuid_map.get(i)


def test_trafo_conversion(input_data):
    expected, input = input_data
    net = pp.create_empty_network()
    input_trafo = input.transformers_2_w.data.iloc[0]
    node_a = input.nodes.data.loc[input_trafo["node_a"]]
    node_b = input.nodes.data.loc[input_trafo["node_b"]]
    noda_a_idx = convert_node(net, node_a)
    noda_b_idx = convert_node(net, node_b)
    uuid_idx = {node_a.name: noda_a_idx, node_b.name: noda_b_idx}
    idx = convert_transformer(net, input_trafo, uuid_idx)

    assert idx == 0
    assert net["trafo"]["name"].iloc[idx] == input_trafo["id"]
    assert net["trafo"]["hv_bus"].iloc[idx] == noda_a_idx
    assert net["trafo"]["lv_bus"].iloc[idx] == noda_b_idx

    sb_trafos = expected.trafo[expected.trafo["name"] == input_trafo["id"]]
    assert len(sb_trafos) == 1
    sb_trafo = sb_trafos.iloc[0]
    assert net["trafo"]["sn_mva"].iloc[idx] == sb_trafo["sn_mva"]
    assert math.isclose(net["trafo"]["vn_hv_kv"].iloc[idx], sb_trafo["vn_hv_kv"])
    assert math.isclose(net["trafo"]["vn_lv_kv"].iloc[idx], sb_trafo["vn_lv_kv"])
    assert math.isclose(net["trafo"]["vk_percent"].iloc[idx], sb_trafo["vk_percent"])
    assert math.isclose(net["trafo"]["vkr_percent"].iloc[idx], sb_trafo["vkr_percent"])
    assert math.isclose(net["trafo"]["pfe_kw"].iloc[idx], sb_trafo["pfe_kw"])
    assert math.isclose(net["trafo"]["i0_percent"].iloc[idx], sb_trafo["i0_percent"])
    assert net["trafo"]["tap_side"].iloc[idx] == sb_trafo["tap_side"]
    assert net["trafo"]["tap_neutral"].iloc[idx] == sb_trafo["tap_neutral"]
    assert net["trafo"]["tap_min"].iloc[idx] == sb_trafo["tap_min"]
    assert net["trafo"]["tap_max"].iloc[idx] == sb_trafo["tap_max"]
    assert math.isclose(
        net["trafo"]["tap_step_degree"].iloc[idx], sb_trafo["tap_step_degree"]
    )
    assert math.isclose(
        net["trafo"]["tap_step_percent"].iloc[idx], sb_trafo["tap_step_percent"]
    )
