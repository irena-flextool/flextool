import csv
import math
import subprocess
import logging
import sys
import os
from collections import OrderedDict
from collections import defaultdict


class FlexToolRunner:
    """
    Define Class to run the model and read and recreate the required config files:
    """

    def __init__(self) -> None:
        logging.basicConfig(
            stream=sys.stderr,
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        translation = {39: None}
        # make a directory for model unit tests
        if not os.path.exists("./tests"):
            os.makedirs("./tests")
        # read the data in
        self.timelines = self.get_timelines()
        self.model_solve = self.get_solves()
        self.solve_modes = self.get_solve_modes("solve_mode")
        self.highs_presolve = self.get_solve_modes("highs_presolve")
        self.highs_method = self.get_solve_modes("highs_method")
        self.highs_parallel = self.get_solve_modes("highs_parallel")
        self.solve_period_discount_years = self.get_solve_period_discount_years()
        self.solvers = self.get_solver()
        self.timeblocks = self.get_timeblocks()
        self.timeblocks__timeline = self.get_timeblocks_timelines()
        self.timeblocks_used_by_solves = self.get_timeblocks_used_by_solves()
        self.invest_periods = self.get_list_of_tuples('input/solve__invest_period.csv')
        self.realized_periods = self.get_list_of_tuples('input/solve__realized_period.csv')
        #self.write_full_timelines(self.timelines, 'steps.csv')

    def get_solves(self):
        """
        read in
        the list of solves, return it as a list of strings
        :return:
        """
        with open("input/model__solve.csv", 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            model__solve = defaultdict(list)
            while True:
                try:
                    datain = next(filereader)
                    model__solve[datain[0]].append((datain[1]))
                except StopIteration:
                    break
        return model__solve

    def get_solve_modes(self, parameter):
        """
        read in
        the list of solve modes, return it as a list of strings
        :return:
        """
        with open("input/solve_mode.csv", 'r') as solvefile:
            header = solvefile.readline()
            solves = solvefile.readlines()
            params = []
            right_params = {}
            for solve in solves:
                solve_stripped = solve.rstrip()
                params.append(solve_stripped.split(","))
            for param in params:
                if param[0] == parameter:
                    right_params[param[1]] = param[2]
        return right_params

    def get_solve_period_discount_years(self):
        """
        read in the timelines including step durations for all simulation steps
        timeline is the only inputfile that contains the full timelines for all timeblocks.
        :return: list of tuples in a dict timeblocks : (timestep name, duration)
        """
        with open('input/solve__period__discount_years.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            discount_years = defaultdict(list)
            while True:
                try:
                    datain = next(filereader)
                    discount_years[datain[0]].append((datain[1], datain[2]))
                except StopIteration:
                    break
        return discount_years

    def get_solver(self):
        """
        read in
        the list of solvers for each solve. return it as a list of strings
        :return:
        """
        with open('input/solver.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            solver_dict = defaultdict()
            while True:
                try:
                    datain = next(filereader)
                    solver_dict[datain[0]] = datain[1]
                except StopIteration:
                    break

        #with open("input/solver.csv", 'r') as solvefile:
        #    header = solvefile.readline()
        #    solvers = solvefile.readlines()
        #    for solver in solvers:
        #        solve__period = solver.split(",")
        #        solver_dict[solve__period[0]] = solve__period[1]
        return solver_dict

    def get_timeblocks_used_by_solves(self):
        """
        timeblocks_in_use.csv contains three columns
        solve: name of the solve
        period: name of the time periods used for a particular solve
        timeblocks: timeblocks used by the period

        :return list of tuples in a dict of solves : (period name, timeblock name)
        """
        with open('input/timeblocks_in_use.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            timeblocks_used_by_solves = defaultdict(list)
            while True:
                try:
                    datain = next(filereader)
                    timeblocks_used_by_solves[datain[0]].append((datain[1], datain[2]))
                    # blockname needs to be in both block_start and timeblock_lengths.csv
                    # assert datain[1] in self.starts.keys(), "Block {0} not in block_starts.csv".format(datain[1])
                    # assert datain[1] in self.steps.keys(), "Block {0} not in block_steps.csv".format(datain[1])
                except StopIteration:
                    break
                #except AssertionError as e:
                #    logging.error(e)
                #    sys.exit(-1)
        return timeblocks_used_by_solves

    def get_timelines(self):
        """
        read in the timelines including step durations for all simulation steps
        timeline is the only inputfile that contains the full timelines for all timeblocks.
        :return: list of tuples in a dict timeblocks : (timestep name, duration)
        """
        with open('input/timeline.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            timelines = defaultdict(list)
            while True:
                try:
                    datain = next(filereader)
                    timelines[datain[0]].append((datain[1], datain[2]))
                except StopIteration:
                    break
        return timelines

    def get_timeblocks_timelines(self):
        """
        read in the timelines including step durations for all simulation steps
        timeline is the only inputfile that contains the full timelines for all timeblocks.
        :return: list of tuples in a dict timeblocks : (timestep name, duration)
        """
        with open('input/timeblocks__timeline.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            timeblocks__timeline = defaultdict(list)
            while True:
                try:
                    datain = next(filereader)
                    timeblocks__timeline[datain[0]].append((datain[1]))
                except StopIteration:
                    break
        return timeblocks__timeline

    def get_timeblocks(self):
        """
        read in the timeblock definitions that say what each set of timeblock contains (timeblock start and length)
        :return: list of tuples in a dict of timeblocks : (start timestep name, timeblock length in timesteps)
        :return: list of tuples that hold the timeblock length in timesteps
        """
        with open('input/timeblocks.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            timeblocks = defaultdict(list)
            #timeblock_lengths = []
            while True:
                try:
                    datain = next(filereader)
                    timeblocks[datain[0]].append((datain[1], datain[2]))
                    """ This assert should check the list of timelines inside the dict, but didn't have time to formulate it yet
                    assert timeblocks[datain[0]] in self.timelines[datain[0]], "Block {0} start time {1} not found in timelines".format(
                        datain[0], datain[1])
                    """
                    #timeblock_lengths.append[(datain[0], datain[1])] = datain[2]
                except StopIteration:
                    break
                """ Once the assert works, this can be included
                except AssertionError as e:
                    logging.error(e)
                    sys.exit(-1)
                """
        return timeblocks

    def get_list_of_tuples(self, filename):
        """
        read in invest_period
        :return  a list of tuples that say when it's ok to invest (solve, period):
        """
        with open(filename, 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            tuple_list = []
            while True:
                try:
                    datain = next(filereader)
                    tuple_list.append((datain[0], datain[1]))
                except StopIteration:
                    break
        return tuple_list

    def make_steps(self, start, stop):
        """
        make a list of timesteps available
        :param start: Start index of of the block
        :param stop: Stop index of the block
        :param steplist: list of steps, read from steps.csv
        :return: list of timesteps
        """

        active_step = start
        steps = []
        while active_step <= stop:
            steps.append(self.steplist[active_step])
            active_step += 1
        return steps

    def write_full_timelines(self, period__timeblocks_in_this_solve, timeblocks__timeline, timelines, filename):
        """
        write to file a list of timestep as defined in timelines.
        :param filename: filename to write to
        :param steplist: list of timestep indexes
        :return:
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('period,step\n')
            for period__timeblock in period__timeblocks_in_this_solve:
                for timeline in timelines:
                    for timeblock_in_timeline, tt in timeblocks__timeline.items():
                        if period__timeblock[1] == timeblock_in_timeline:
                            if timeline == tt[0]:
                                for item in timelines[timeline]:
                                    outfile.write(period__timeblock[0] + ',' + item[0] + '\n')

    def write_active_timelines(self, timeline, filename):
        """
        write to file a list of timesteps as defined by the active timeline of the current solve
        :param filename: filename to write to
        :param timeline: list of tuples containing the period and the timestep
        :return: nothing
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('period,step,step_duration\n')
            for period_name, period in timeline.items():
                for item in period:
                    outfile.write(period_name + ',' + item[0] + ',' + item[2] + '\n')


    def write_discount_years(self, discount_years, filename):
        """
        write to file a list of timesteps as defined by the active timeline of the current solve
        :param filename: filename to write to
        :param timeline: list of tuples containing the period and the timestep
        :return: nothing
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('period,p_discount_years\n')
            for period__discount_year in discount_years:
                outfile.write(period__discount_year[0] + ',' + period__discount_year[1] + '\n')

    def make_block_timeline(self, start, length):
        """
        make a block timeline, there might be multiple blocks per solve so these blocks might need to be combined for a run
        :param start: start of block
        :param length: length of block
        :return: block timeline
        """
        steplist = []
        startnum = self.steplist.index(start)
        for i in range(startnum, math.ceil(startnum + float(length))):
            steplist.append(self.steplist[i])
        return steplist

    def model_run(self, current_solve):
        """
        run the model executable once
        :return the output of glpsol.exe:
        """
        try:
            solver = self.solvers[current_solve]
        except KeyError:
            logging.warning(f"No solver defined for {current_solve}. Defaulting to highs.")
            solver = "highs"
        if solver == "glpsol":
            only_glpsol = ['glpsol', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat', '--cbg'] + sys.argv[1:]
            completed = subprocess.run(only_glpsol)
            if completed.returncode != 0:
                logging.error(f'glpsol failed: {completed.returncode}')
                exit(completed.returncode)
        elif solver == "highs":
            highs_step1 = ['glpsol', '--check', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat',
                           '--wfreemps', 'flexModel3.mps'] + sys.argv[1:]
            completed = subprocess.run(highs_step1)
            if completed.returncode != 0:
                logging.error(f'glpsol mps writing failed: {completed.returncode}')
                exit(completed.returncode)
            print("GLPSOL wrote the problem as MPS file\n")
            highs_step2 = "highs flexModel3.mps --options_file=highs.opt --presolve=" \
                + self.highs_presolve.get(current_solve, "on") + " --solver=" \
                + self.highs_method.get(current_solve, "simplex") + " --parallel=" \
                + self.highs_parallel.get(current_solve, "off")
            completed = subprocess.run(highs_step2)
            if completed.returncode != 0:
                logging.error(f'Highs solver failed: {completed.returncode}')
                exit(completed.returncode)
            print("HiGHS solved the problem\n")
            highs_step3 = ['glpsol', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat', '-r',
                           'flexModel3.sol'] + sys.argv[1:]
            completed = subprocess.run(highs_step3)
            if completed.returncode == 0:
                print("GLPSOL wrote the results into csv files\n")
        else:
            logging.error(f"Unknown solver '{solver}'. Currently supported options: highs, glpsol.")
            exit(1)
        return completed.returncode

    def get_active_time(self, current_solve, timeblocks_used_by_solves, timeblocks, timelines, timeblocks__timelines):
        """
        retunr all block codes that are included in solve
        :param solve:
        :param blocklist:
        :return:
        """
        active_time = defaultdict(list)
        for solve in timeblocks_used_by_solves:
            if solve == current_solve:
                for period_timeblock in timeblocks_used_by_solves[solve]:
                    for timeblocks__timeline_key, timeblocks__timeline_value in timeblocks__timelines.items():
                        if timeblocks__timeline_key == period_timeblock[1]:
                            for timeline in timelines:
                                if timeline == timeblocks__timeline_value[0]:
                                    for single_timeblock_def in timeblocks[timeblocks__timeline_key]:
                                        for index, timestep in enumerate(timelines[timeline]):
                                            if timestep[0] == single_timeblock_def[0]:
                                                for block_step in range(int(float(single_timeblock_def[1]))):
                                                    active_time[period_timeblock[0]].append((
                                                                        timelines[timeline][index + block_step][0],
                                                                        index + block_step,
                                                                        timelines[timeline][index + block_step][1]))
                                                break
        return active_time

    def make_step_jump(self, active_time_list):
        """
        make a file that indicates the length of jump from one simulation step to next one.
        the final line should always contain a jump to the first line.

        length of jump is the number of lines needed to advance in the timeline specified in step_duration.csv

        :param steplist: active steps used in the solve
        :param duration: duration of every timestep
        :return:
        """
        step_lengths = []
        period_start_pos = 0
        period_counter = -1
        first_period_name = list(active_time_list)[0]
        last_period_name = list(active_time_list)[-1]
        for period, active_time in reversed(active_time_list.items()):
            period_counter -= 1
            period_last = len(active_time)
            block_last = len(active_time) - 1
            if period == first_period_name:
                previous_period_name = last_period_name
            else:
                previous_period_name = list(active_time_list)[period_counter]
            for i, step in enumerate(reversed(active_time)):
                j = period_last - i - 1
                if j == period_last - 1:
                    store_last_time_step = step[0]
                if j > 0:  # handle the first element of the period separately below
                    jump = active_time[j][1] - active_time[j - 1][1]
                    if jump > 1:
                        step_lengths.insert(period_start_pos, (period, step[0], active_time[j - 1][0], active_time[block_last][0], period, active_time[j - 1][0]))
                        block_last = j - 1
                    else:
                        step_lengths.insert(period_start_pos, (period, step[0], active_time[j - 1][0], active_time[j - 1][0], period, active_time[j - 1][0]))
                else:  # first time step of the period is handled here
                    step_lengths.insert(period_start_pos, (period, step[0], active_time[j - 1][0], active_time[block_last][0], previous_period_name, store_last_time_step))
        return step_lengths

    def write_step_jump(self, step_lengths):
        """
        write step_jump.csv according to spec.

        :param step_lengths:
        :return:
        """

        headers = ("period", "time", "previous", "previous_within_block", "previous_period", "previous_within_solve")
        with open("solve_data/step_previous.csv", 'w', newline='\n') as stepfile:
            writer = csv.writer(stepfile, delimiter=',')
            writer.writerow(headers)
            writer.writerows(step_lengths)

    def get_first_steps(self, steplists):
        """
        get the first step of the current solve and the next solve in execution order.
        :param steplists: Dictionary containg steplist for each solve, in order
        :return: Return a dictionary containing tuples of current_first, next first
        """
        solve_names = list(steplists.keys())
        starts = OrderedDict()
        for index, name in enumerate(solve_names):
            # last key is a different case
            if index == (len(solve_names) - 1):
                starts[name] = (steplists[name][0],)
            else:
                starts[name] = (steplists[solve_names[index]][0], steplists[solve_names[index + 1]][0])
        return starts

    def write_first_steps(self, timeline, filename):
        """
        write to file the first step of each period
        
        :param steps: a tuple containing the period and the timestep
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('period,step\n')
            for period_name, period in timeline.items():
                for item in period[:1]:
                    outfile.write(period_name + ',' + item[0] + '\n')

    def write_last_steps(self, timeline, filename):
        """
        write to file the last step of each period

        :param steps: a tuple containing the period and the timestep
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('period,step\n')
            for period_name, period in timeline.items():
                for item in period[-1:]:
                    outfile.write(period_name + ',' + item[0] + '\n')

    def write_periods(self, solve, periods, filename):
        """
        write to file a list of periods based on the current solve and
        a list of tuples with the solve as the first element in the tuple
        :param solve: current solve
        :param filename: filename to write to
        :param periods: list of tuples with solve and periods to be printed to the file
        :return: nothing
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('period\n')
            for item in periods:
                if item[0] == solve:
                    outfile.write(item[1] + '\n')

    def write_solve_status(self, first_state, last_state):
        """
        make a file solve_first.csv that contains information if the current solve is the first to be run

        :param first_state: boolean if the current run is the first

        """
        with open("input/p_model.csv", 'w') as p_model_file:
            p_model_file.write("modelParam,p_model\n")
            if first_state:
                p_model_file.write("solveFirst,1\n")
            else:
                p_model_file.write("solveFirst,0\n")
            if last_state:
                p_model_file.write("solveLast,1\n")
            else:
                p_model_file.write("solveLast,0\n")

    def write_currentSolve(self, solve, filename):
        """
        make a file with the current solve name
        """
        with open(filename, 'w') as solvefile:
            solvefile.write("solve\n")
            solvefile.write(solve + "\n")

    def write_empty_investment_file(self):
        """
        make a file p_entity_invested.csv that will contain capacities of invested and divested processes. For the first solve it will be empty.

        :param first_state: boolean if the current run is the first

        """
        with open("solve_data/p_entity_invested.csv", 'w') as firstfile:
            firstfile.write("entity,p_entity_invested\n")


def main():
    """
    first read the solve configuration from the input files, then for each solve write the files that are needed
    By that solve into disk. separate the reading into a separate step since the input files need knowledge of multiple solves.
    """
    runner = FlexToolRunner()
    active_time_lists = OrderedDict()
    jump_lists = OrderedDict()
    try:
        os.mkdir('solve_data')
    except FileExistsError:
        print("solve_data folder existed")

    if not runner.model_solve:
        logging.error("No model. Make sure the 'model' class defines solves.")
        sys.exit(1)
    solves = next(iter(runner.model_solve.values()))
    if not solves:
        logging.error("No solves in model.")
        sys.exit(1)
    for solve in solves:
        active_time_list = runner.get_active_time(solve, runner.timeblocks_used_by_solves, runner.timeblocks,
                                              runner.timelines, runner.timeblocks__timeline)
        active_time_lists[solve] = active_time_list
        jumps = runner.make_step_jump(active_time_list)
        jump_lists[solve] = jumps

    first = True
    for i, solve in enumerate(solves):
        runner.write_full_timelines(runner.timeblocks_used_by_solves[solve], runner.timeblocks__timeline, runner.timelines, 'solve_data/steps_in_timeline.csv')
        runner.write_active_timelines(active_time_lists[solve], 'solve_data/steps_in_use.csv')
        runner.write_step_jump(jump_lists[solve])
        runner.write_periods(solve, runner.realized_periods, 'solve_data/realized_periods_of_current_solve.csv')
        runner.write_periods(solve, runner.invest_periods, 'solve_data/invest_periods_of_current_solve.csv')
        runner.write_discount_years(runner.solve_period_discount_years[solve], 'solve_data/p_discount_years.csv')
        runner.write_currentSolve(solve, 'solve_data/solve_current.csv')
        runner.write_first_steps(active_time_lists[solve], 'solve_data/first_timesteps.csv')
        runner.write_last_steps(active_time_lists[solve], 'solve_data/last_timesteps.csv')
        last = i == len(solves) - 1
        runner.write_solve_status(first, last)
        if i == 0:
            first = False
            runner.write_empty_investment_file()

        exit_status = runner.model_run(solve)
        if exit_status == 0:
            logging.info('Success!')
        else:
            logging.error(f'Error: {exit_status}')
    if len(runner.model_solve) > 1:
        logging.error(
            f'Trying to run more than one model - not supported. The results of the first model are retained.')
        sys.exit(1)


if __name__ == '__main__':
    main()
