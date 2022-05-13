import csv
import math
import subprocess
import itertools
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
        self.solves = self.get_solves()
        self.timeblocks = self.get_timeblocks()
        self.timeblocks__timeline = self.get_timeblocks_timelines()
        self.timeblocks_used_by_solves = self.get_timeblocks_used_by_solves()
        self.invest_periods = self.get_list_of_tuples('input/solve__invest_period.csv')
        self.realized_periods = self.get_list_of_tuples('input/solve__realized_period.csv')
        #self.write_full_timelines(self.timelines, 'steps.csv')

    def get_solves(self):
        """
        read in
        the list of solves return it as a list of strings
        :return:
        """
        with open("input/solve_mode.csv", 'r') as solvefile:
            header = solvefile.readline()
            solves = solvefile.readlines()
        return [solve.split(",")[0] for solve in solves]

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

    @property
    def model_run(self):
        """
        run the model executable once
        :return the output of glpsol.exe:
        """
        foo = ['glpsol', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat', '--cbg'] + sys.argv[1:]
        #highs_step1 = ['glpsol', '--check', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat', '--wmps', 'instance.mps']
        #highs_step2 = ['highs instance.mps']
        #highs_step3 = ['glpsol', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat', '--wmps', 'instance.mps']
        modelout = subprocess.Popen(['glpsol.exe', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat'] +
                                    sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = modelout.communicate()
        # print(stdout.decode("utf-8"))
        # print(stderr)
        return stdout, stderr

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
        for period, active_time in active_time_list.items():
            period_last = len(active_time)
            block_last = len(active_time) - 1
            for i, step in enumerate(reversed(active_time)):
                j = period_last - i - 1
                if j > 0:  # handle the first element of the period separately below
                    jump = active_time[j][1] - active_time[j - 1][1]
                    if jump > 1:
                        jump_back = active_time[j][1] - active_time[block_last][1]
                        step_lengths.insert(period_start_pos, (period, step[0], active_time[j - 1][0], active_time[block_last][0]))
                        block_last = j - 1
                    else:
                        step_lengths.insert(period_start_pos, (period, step[0], active_time[j - 1][0], active_time[j - 1][0]))
                else:  # first time step of the period is handled here
                    jump = active_time[j][1] - active_time[len(active_time) - 1][1]
                    step_lengths.insert(period_start_pos, (period, step[0], active_time[j - 1][0], active_time[block_last][0]))
            period_start_pos = period_start_pos + period_last
        return step_lengths

    def write_step_jump(self, step_lengths):
        """
        write step_jump.csv according to spec.

        :param step_lengths:
        :return:
        """

        headers = ("period", "time", "previous", "previous_within_block")
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

    def write_first_steps(self, steps):
        """
        write to file the first step of the model run
        
        write info into two separate files "solve_start.csv" & "solve_startNext.csv"

        :param steps: a tuple containg the first step of current solve and the first step of next solve
                        in case the current solve is the last one the second item is empty
        """
        with open("solve_start.csv", "w") as startfile:
            startfile.write("start\n")
            startfile.write(steps[0])
            startfile.write("\n")

        with open("solve_startNext.csv", 'w') as nextfile:
            nextfile.write("startNext\n")
            if len(steps) == 2:
                nextfile.write(steps[1])
                nextfile.write("\n")

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

    def write_first_status(self, first_state):
        """
        make a file solve_first.csv that contains information if the current solve is the first to be run

        :param first_state: boolean if the current run is the first

        """
        with open("input/p_model.csv", 'w') as firstfile:
            firstfile.write("modelParam,p_model\n")
            if first_state:
                firstfile.write("solveFirst,1\n")
            else:
                firstfile.write("solveFirst,0\n")

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

    for solve in runner.solves:
        active_time_list = runner.get_active_time(solve, runner.timeblocks_used_by_solves, runner.timeblocks,
                                                  runner.timelines, runner.timeblocks__timeline)
        active_time_lists[solve] = active_time_list
        jumps = runner.make_step_jump(active_time_list)
        jump_lists[solve] = jumps

    #first_steps = runner.get_first_steps(active_time_lists)

    first = True
    for solve in runner.solves:
        runner.write_full_timelines(runner.timeblocks_used_by_solves[solve], runner.timeblocks__timeline, runner.timelines, 'solve_data/steps_in_timeline.csv')
        runner.write_active_timelines(active_time_lists[solve], 'solve_data/steps_in_use.csv')
        runner.write_step_jump(jump_lists[solve])
        runner.write_periods(solve, runner.realized_periods, 'solve_data/realized_periods_of_current_solve.csv')
        runner.write_periods(solve, runner.invest_periods, 'solve_data/invest_periods_of_current_solve.csv')
        runner.write_currentSolve(solve, 'solve_data/solve_current.csv')
        if first:
            runner.write_first_status(first)
            first = False
            runner.write_empty_investment_file()
        else:
            runner.write_first_status(first)

        model_out, model_err = runner.model_run
        logging.info(model_out.decode("utf-8"))


if __name__ == '__main__':
    main()
