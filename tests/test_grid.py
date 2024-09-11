import math

import pandapower as pp
import pytest

from pp2psdm.grid import convert_grid, convert_line, convert_node, convert_transformer
from tests.utils import read_psdm_lv, read_sb_lv


@pytest.fixture
def input_data():
    expected = read_sb_lv()
    input = read_psdm_lv()
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


def test_node_conversion(input_data):
    _, input = input_data
    net = pp.create_empty_network()
    input_node = input.nodes.data.iloc[0]
    idx = convert_node(net, input_node)
    assert idx == 0
    assert net["bus"]["name"].iloc[idx] == input_node["id"]
    assert net["bus"]["vn_kv"].iloc[idx] == input_node["v_rated"]
    assert net["bus"]["subnet"].iloc[idx] == input_node["subnet"]
    assert net["bus"]["in_service"].iloc[idx]
    assert net["bus_geodata"]["x"].iloc[idx] == input_node["longitude"]
    assert net["bus_geodata"]["y"].iloc[idx] == input_node["latitude"]


def test_line_conversion(input_data):
    expected, input = input_data
    net = pp.create_empty_network()
    input_line = input.lines.data.iloc[0]
    node_a = input.nodes.data.loc[input_line["node_a"]]
    node_b = input.nodes.data.loc[input_line["node_b"]]
    noda_a_idx = convert_node(net, node_a)
    noda_b_idx = convert_node(net, node_b)
    uuid_idx = {node_a.name: noda_a_idx, node_b.name: noda_b_idx}
    idx = convert_line(net, input_line, uuid_idx)

    assert idx == 0
    assert net["line"]["name"].iloc[idx] == input_line["id"]
    assert net["line"]["from_bus"].iloc[idx] == noda_a_idx
    assert net["line"]["to_bus"].iloc[idx] == noda_b_idx

    sb_lines = expected.line[expected.line["name"] == input_line["id"]]
    assert len(sb_lines) == 1
    sb_line = sb_lines.iloc[0]
    assert net["line"]["length_km"].iloc[idx] == sb_line["length_km"]
    assert math.isclose(net["line"]["r_ohm_per_km"].iloc[idx], sb_line["r_ohm_per_km"])
    assert math.isclose(net["line"]["x_ohm_per_km"].iloc[idx], sb_line["x_ohm_per_km"])
    assert math.isclose(net["line"]["c_nf_per_km"].iloc[idx], sb_line["c_nf_per_km"])
    assert math.isclose(net["line"]["g_us_per_km"].iloc[idx], sb_line["g_us_per_km"])
    assert math.isclose(net["line"]["max_i_ka"].iloc[idx], sb_line["max_i_ka"])


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
