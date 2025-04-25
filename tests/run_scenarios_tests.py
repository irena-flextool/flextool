#!/usr/bin/env python3
import subprocess
import os
import argparse
import sys
import spinedb_api as api
from spinedb_api import DatabaseMapping


def run_scenario(original_dir, flextool_folder, db_url, scenario):
    """
    Run a specific scenario from a database URL
    You'll replace this with your actual scenario execution command

    Args:
        flextool_folder (str): Location of the Flextool to be executed
        db_url (str): Database URL containing the scenario
        scenario_name (str): Name of the scenario to be run

    Returns:
        bool: Success or failure of scenario execution
    """
    print(f"Running scenario {scenario} from {db_url} using {flextool_folder}")

    # Get the absolute path to the run_flextool.py script
    run_flextool_path = os.path.join(os.path.abspath(flextool_folder), "run_flextool.py")

    # Set up environment variables for output directory if needed
    original_env = os.environ.copy()

    try:
        # Change to the flextool directory if needed
        os.chdir(os.path.dirname(run_flextool_path))

        # Execute the script directly
        # This runs the script in the same process
        script_globals = {
            '__file__': run_flextool_path,
            '__name__': '__main__'
        }

        with open(run_flextool_path, 'r') as f:
            script_code = f.read()

        # Pass the arguments through sys.argv
        saved_argv = sys.argv
        sys.argv = [run_flextool_path, db_url, scenario]

        try:
            # Execute the script
            exec(script_code, script_globals)
            return True
        except SystemExit as e:
            # The script called sys.exit()
            if e.code != 0:
                print(f"run_flextool.py exited with code {e.code}")
                return False
            return True
        finally:
            # Restore sys.argv
            sys.argv = saved_argv

    except Exception as e:
        print(f"Error executing run_flextool.py: {str(e)}")
        return False
    finally:
        # Restore environment and working directory
        os.environ.clear()
        os.environ.update(original_env)
        os.chdir(original_dir)



    # This is a placeholder - replace with your actual scenario execution command
    # For example: subprocess.run(["./run_scenario.sh", scenario_id, db_url, output_dir])
    # print(f"Running scenario {scenario_name} from {db_url} using {flextool_folder}")
    #
    # # Simulate running the scenario (replace with your actual command)
    # os.chdir(f"{original_dir}")
    # print(os.getcwd())
    # command = [f"python {flextool_folder}/run_flextool.py {db_url} {scenario_name}"]
    # result = subprocess.run(command, capture_output=True, text=True)
    #
    # if result.returncode != 0:
    #     print(f"Error running scenario {scenario_name}: {result.stderr}")
    #     return False
    #
    # return True


def compare_outputs(comparison_script, run_dir, dir1, dir2, output_file):
    """
    Compare two output directories using the provided comparison script

    Args:
        comparison_script (str): Path to the comparison script
        dir1 (str): First directory to compare
        dir2 (str): Second directory to compare
        output_file (str): Output file for results

    Returns:
        bool: Success or failure of comparison
    """
    sys.path.append(f"{run_dir}")

    cmd = [sys.executable, comparison_script, dir1, dir2, "--output", output_file]

    print(f"Comparing outputs: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error comparing directories: {result.stderr}")
        return False

    print(f"Comparison completed successfully. Results saved to {output_file}")
    return True


def get_scenarios(folder, db_filename):
    """
    Get list of available scenarios from database
    Replace this with your actual method to retrieve scenarios

    Args:
        folder: location of flextool
        db_filename (str): Database filename

    Returns:
        list: List of scenario IDs
    """
    # This is a placeholder - replace with your actual method to get scenarios
    # For example, this might query the database or read from a file

    with DatabaseMapping(f"sqlite:///{folder}/{db_filename}") as db_map:
        scenario_table = db_map.mapped_table("scenario")
        scenarios = db_map.find(scenario_table)
        scenario_names = []
        for scenario in scenarios:
            scenario_names.append(scenario["name"])

    # Return dummy scenario IDs (replace with actual scenario retrieval)
    return scenario_names


def main():
    parser = argparse.ArgumentParser(description='Run and compare scenarios from two databases')
    parser.add_argument('flextool_folder1', help='First folder where to run FlexTool from')
    parser.add_argument('flextool_folder2', help='Second folder where to run FlexTool from')
    parser.add_argument('db_filename1', help='First database filename')
    parser.add_argument('db_filename2', help='Second database filename')
    parser.add_argument('--comparison-script', default='compare_files.py',
                        help='Path to comparison script')

    args = parser.parse_args()

    # Create base output directory
    original_dir = os.getcwd()
    os.makedirs("comparisons_dir", exist_ok=True)

    # Get scenarios from both databases (replace with your actual implementation)
    flextool_folder1 = args.flextool_folder1
    flextool_folder2 = args.flextool_folder2
    db_url1 = f"sqlite:///{args.db_filename1}"
    db_url2 = f"sqlite:///{args.db_filename2}"
    scenarios1 = get_scenarios(flextool_folder1, args.db_filename1)
    scenarios2 = get_scenarios(flextool_folder2, args.db_filename2)

    # Find common scenarios
    common_scenarios = set(scenarios1) & set(scenarios2)

    if not common_scenarios:
        print("No common scenarios found between the two databases.")
        return

    print(f"Found {len(common_scenarios)} common scenarios to process")
    test_dir = "test"
    if not os.path.exists(test_dir):
        os.mkdir("test")

    # Process each common scenario
    for scenario in common_scenarios:
        print(f"\nProcessing scenario: {scenario}")

        # Run scenario from first database
        success1 = run_scenario(original_dir, flextool_folder1, db_url1, scenario)
        if not success1:
            print(f"Skipping scenario {scenario} due to failure in first database")
            continue

        # Run scenario from second database
        success2 = run_scenario(original_dir, flextool_folder2, db_url2, scenario)
        if not success2:
            print(f"Skipping scenario {scenario} due to failure in second database")
            continue

        # Compare outputs
        comparison_file = os.path.join(original_dir, f"test/{scenario}_comparison.txt")
        compare_outputs(f"{flextool_folder1}/flextool/{args.comparison_script}",
                        f"{original_dir}/{flextool_folder1}/flextool/",
                        f"{flextool_folder1}/output",
                        f"{flextool_folder2}/output",
                        comparison_file)

    print("\nAll scenarios processed. See comparison results in:", os.getcwd())


if __name__ == "__main__":
    main()