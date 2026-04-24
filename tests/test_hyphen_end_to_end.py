"""Regression test: hyphens in entity names must remain supported.

History: a user reported ``pdtProcess[ARG_H2,...] out of domain`` on an
H2-trade scenario with hyphenated connection names like ``BRA_ARG-H2``.
The obvious suspect was the hyphen colliding with MathProg's subtraction
operator — we temporarily added a validator that rejected any entity
name containing ``-``.

This test reproduced the exact conditions (connection with a hyphen in
its name AND a non-zero ``other_operational_cost`` so it actually enters
``pssdt_varCost_eff_connection``) and showed that GLPK solves the model
without complaint.  The validator was therefore a false positive and was
reverted.  This test stays in the suite so anyone who considers
re-adding such a validator has to also delete this regression.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner


def _prepare_hyphenated_db(src_url: str, dst_path: Path) -> str:
    """Clone src DB → dst, rename ``west_east`` → ``west-east`` and attach
    ``other_operational_cost=0.1`` under the ``network`` alternative
    (same alternative the ``network_coal_wind`` scenario pulls in)."""
    src_path = Path(src_url.replace("sqlite:///", ""))
    shutil.copy(src_path, dst_path)
    dst_url = f"sqlite:///{dst_path}"

    with DatabaseMapping(dst_url) as db:
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        # Rename the connection. Relationships (connection__node__node,
        # entity_alternatives, parameter_values) cascade because they
        # reference entity_id, not name.
        conn = next(
            e for e in db.find_entities(entity_class_name="connection")
            if e["entity_byname"][0] == "west_east"
        )
        db.update_item("entity", id=conn["id"], name="west-east")

        # Add other_operational_cost=0.1 in the 'network' alternative so
        # the connection actually enters pssdt_varCost_eff_connection.
        val, vtype = to_database(0.1)
        db.add_update_item(
            "parameter_value",
            entity_class_name="connection",
            parameter_definition_name="other_operational_cost",
            entity_byname=("west-east",),
            alternative_name="network",
            value=val, type=vtype,
        )
        db.commit_session("hyphenate west_east + add VOM")

    return dst_url


def test_glpk_accepts_hyphenated_connection_with_vom(
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> None:
    """Hyphenated connection name + non-zero VOM → solve must still succeed."""

    hyph_db_url = _prepare_hyphenated_db(test_db_url, workdir / "hyph.sqlite")

    runner = FlexToolRunner(
        input_db_url=hyph_db_url,
        scenario_name="network_coal_wind",
        root_dir=workdir,
        bin_dir=test_bin_dir,
    )
    runner.write_input(hyph_db_url, "network_coal_wind")
    return_code = runner.run_model()

    assert return_code == 0, (
        "Solve failed for a model with hyphenated connection 'west-east' "
        "carrying other_operational_cost. If you're about to add a "
        "validator that rejects hyphens, stop — this test has been the "
        "regression guard since 2026-04 and the root cause of similar "
        "user reports lies elsewhere."
    )
