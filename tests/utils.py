from pathlib import Path

import pandapower as pp
from pypsdm import GridContainer

ROOT_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT_DIR / "tests"
TEST_RESOURCES_DIR = TESTS_DIR / "resources"
SB_DIR = TEST_RESOURCES_DIR / "simbench"
PSDM_DIR = TEST_RESOURCES_DIR / "psdm"
LV_NAME = "1-LV-rural1--2-no_sw_1"


def read_sb_lv():
    return pp.from_json(SB_DIR / (LV_NAME + ".json"))


def read_psdm_lv():
    return GridContainer.from_csv(str(PSDM_DIR / LV_NAME / "input"), delimiter=";")
