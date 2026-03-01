import logging
import sys
import os
import spinedb_api as api
from spinedb_api import DatabaseMapping
from pathlib import Path

from flextool.flextoolrunner.db_reader import check_version
from flextool.flextoolrunner import input_writer
from flextool.flextoolrunner import orchestration
from flextool.flextoolrunner.solve_config import SolveConfig
from flextool.flextoolrunner.timeline_config import TimelineConfig
from flextool.flextoolrunner.runner_state import PathConfig, RunnerState
from flextool.flextoolrunner.solver_runner import SolverRunner

#return_codes
#0 : Success
#-1: Failure (Defined in the Toolbox)
#1: Infeasible or unbounded problem (not implemented in the toolbox, functionally same as -1. For a possiblity of a graphical depiction)


class FlexToolRunner:
    """
    Define Class to run the model and read and recreate the required config files:
    """

    def __init__(self, input_db_url=None, output_path=None, scenario_name=None, flextool_dir=None, bin_dir=None, root_dir=None):
        logger = logging.getLogger(__name__)
        translation = {39: None}
        # delete highs.log from previous run
        if os.path.exists("./HiGHS.log"):
            os.remove("./HiGHS.log")
        # make a directory for solve data
        if not os.path.exists("./solve_data"):
            os.makedirs("./solve_data")
        # Build PathConfig
        _default_root = Path(__file__).parent.parent.parent
        paths = PathConfig(
            flextool_dir=Path(flextool_dir) if flextool_dir is not None else _default_root / "flextool",
            bin_dir=Path(bin_dir) if bin_dir is not None else _default_root / "bin",
            root_dir=Path(root_dir) if root_dir is not None else _default_root,
            output_path=Path(output_path) if output_path is not None else _default_root,
        )
        # read the data in
        # open connection to input db
        if scenario_name:
            scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
        with (DatabaseMapping(input_db_url) as db):
            if scenario_name:
                api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
            else:
                scen_names = db.get_scenario_items()
                if len(scen_names) == 0:
                    logger.error("No scenario found")
                    sys.exit(-1)
                scenario_name=scen_names[0]['name']
            logger.info(" Work dir: " + str(paths.root_dir) + "\nDB URL: " + str(db.sa_url) + "\nScenario name: " + scenario_name + "\nOutput path: " + str(paths.output_path))
            if len(db.get_scenario_alternative_items(scenario_name=scenario_name)) == 0:
                logger.error("No alternatives in the scenario, i.e. empty scenario.")
                sys.exit(-1)

            db.fetch_all("parameter_value")
            check_version(db=db, logger=logger)
            # Solve-level fields — delegated to SolveConfig
            solve = SolveConfig.load_from_db(db=db, logger=logger)
            # Timeline-level fields — delegated to TimelineConfig
            timeline = TimelineConfig.load_from_db(db=db, logger=logger)

        # Post-DB initialization of timeline
        timeline.create_assumptive_parts(solve)
        timeline.create_timeline_from_timestep_duration()

        # Assemble RunnerState — the single cross-cutting state container
        self.state = RunnerState(paths=paths, solve=solve, timeline=timeline, logger=logger)



    def run_model(self) -> int:
        """Run the full solve loop (delegates to orchestration.run_model)."""
        solver = SolverRunner(self.state)
        return orchestration.run_model(self.state, solver)

    def write_input(self, input_db_url, scenario_name=None) -> None:
        input_writer.write_input(input_db_url, scenario_name, self.state.logger)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.error("Run using run_flextool.py in the root of FlexTool")
    sys.exit(-1)

if __name__ == '__main__':
    main()
