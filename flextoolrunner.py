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
        # make a directory for model unit tests
        if not os.path.exists("./tests"):
            os.makedirs("./tests")
        # read the data in
        self.timelines = self.get_timelines()
        self.solves = self.get_solves()
        self.timeblocks = self.get_timeblocks()
        self.timeblocks_used_by_periods = self.get_timeblocks_used_by_periods()
        self.invest = self.get_invest_period()
        self.output_invest_period = self.get_output_invest_period()
        #self.write_full_timeline(timelines, 'steps.csv')

    def get_solves(self):
        """
        read in the list of solves return it as a list of strings
        :return:
        """
        with open("solves.csv", 'r') as solvefile:
            header = solvefile.readline()
            solves = solvefile.readlines()
        return [solve.strip() for solve in solves]

    def get_timeblocks_used_by_periods(self):
        """
        timeblocks_in_use.csv contains three columns
        solve: name of the solve
        period: name of the time periods used for a particular solve
        timeblocks: timeblocks used by the period

        :return list of tuples in a dict of solves : (period name, timeblock name)
        """
        with open('timeblocks_in_use.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            timeblocks_used_by_periods = defaultdict(list)
            while True:
                try:
                    datain = next(filereader)
                    timeblocks_used_by_periods[datain[0]].append((datain[1], datain[2]))
                    # blockname needs to be in both block_start and timeblock_lengths.csv
                    # assert datain[1] in self.starts.keys(), "Block {0} not in block_starts.csv".format(datain[1])
                    # assert datain[1] in self.steps.keys(), "Block {0} not in block_steps.csv".format(datain[1])
                except StopIteration:
                    break
                #except AssertionError as e:
                #    logging.error(e)
                #    sys.exit(-1)
        return timeblocks_used_by_periods

    def get_timelines(self):
        """
        read in the timelines including step durations for all simulation steps
        timeline is the only inputfile that contains the full timelines for all timeblocks.
        :return: list of tuples in a dict timeblocks : (timestep name, duration)
        """
        with open('timeline.csv', 'r') as blk:
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

    def get_timeblocks(self):
        """
        read in the timeblock definitions that say what each set of timeblock contains (timeblock start and length)
        :return: list of tuples in a dict of timeblocks : (start timestep name, timeblock length in timesteps)
        :return: list of tuples that hold the timeblock length in timesteps
        """
        with open('timeblocks.csv', 'r') as blk:
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

    def get_invest_period(self):
        """
        read in invest_period
        :return  a list of tuples that say when it's ok to invest (solve, period):
        """
        with open('invest_period.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            invest_period = []
            while True:
                try:
                    datain = next(filereader)
                    invest_period.append((datain[0], datain[1]))
                except StopIteration:
                    break
        return invest_period

    def get_output_invest_period(self):
        """
        read the investment periods to be output in each solve
        :return: dict : (solve, period)
        """
        with open('output_invest_period.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            solve_period = defaultdict(list)
            while True:
                try:
                    datain = next(filereader)
                    solve_period[datain[0]].append(datain[1])
                except StopIteration:
                    break
        return solve_period

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

    def write_full_timelines(self, timelines, filename):
        """
        write to file a list of timestep as defined in timelines. Use the same function to write
        the total set and the scenario dependent one
        :param filename: filename to write to
        :param steplist: list of timestep indexes
        :return:
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('step\n')
            for timeline in timelines:
                outfile.write(timeline + ',' + timelines[timeline][0] + '\n')

    def write_active_timelines(self, timeline, filename):
        """
        write to file a list of timesteps as defined by the active timeline of the current solve
        :param filename: filename to write to
        :param timeline: list of tuples containing the period and the timestep
        :return: nothing
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('step\n')
            for item in timeline:
                outfile.write(item[0] + ',' + item[1] + '\n')

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

    def write_step_invest(self, solve_code, step_invest):
        """
        make new step_invest2.csv that has only the timestamps defined by solve_code
        :param solve_code:
        :param step_invest:
        :return:
        """
        with open("step_invest.csv", 'w', newline='\n') as stepfile:
            headers = ["solve", "step_invest"]
            writer = csv.writer(stepfile, delimiter=',')
            writer.writerow(headers)
            for line in step_invest:
                if line[0] == solve_code:
                    writer.writerow(line)

    def model_run(self):
        """
        run the model executable once
        :return the output of glpsol.exe:
        """
        modelout = subprocess.Popen(['glpsol.exe', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = modelout.communicate()
        # print(stdout.decode("utf-8"))
        # print(stderr)
        return stdout, stderr

    def get_active_time(self, solve, timeblocks_used_by_periods, timeblocks, timelines):
        """
        retunr all block codes that are included in solve
        :param solve:
        :param blocklist:
        :return:
        """
        active_time = []
        for solve_timeblock in timeblocks_used_by_periods:
            if solve_timeblock == solve:
                for timeline in timelines:
                    for period_timeblock in timeblocks_used_by_periods[solve_timeblock]:
                        if timeline == period_timeblock[1]:
                            #timeblocks[datain[0]].append((datain[1], datain[2]))
                            for timeblocks_def in timeblocks:
                                if timeblocks_def == timeline:
                                    for single_timeblock_def in timeblocks[timeblocks_def]:
                                        for index, timestep in enumerate(timelines[timeline]):
                                            if timestep[0] == single_timeblock_def[0]:
                                                for block_step in range(int(float(single_timeblock_def[1]))):
                                                    active_time.append((period_timeblock[0], timelines[timeline][index + block_step][0]))
                                                break
        return active_time

    def make_step_jump(self, active_time_list, timelines, timeblocks_used_by_periods, solve):
        """
        make a file that indicates the length of jump from one simulation step to next one.
        the final line should always contain a jump to the first line.

        length of jump is the number of lines needed to advance in the timeline specified in step_duration.csv

        :param steplist: active steps used in the solve
        :param duration: duration of every timestep
        :return:
        """
        # define index for every step from the reference list
        active_steps = []
        for timeline in timelines:
            for period_timeblock in timeblocks_used_by_periods[solve]:
                if timeline == period_timeblock[1]:
                    for i, line in enumerate(timelines[timeline]):
                        # duration lines (stepcode, duration)
                        if line[0] in active_time_list:
                            active_steps.append((line[0], i))
        # calculate jump length based on step index
        step_lengths = []
        for j, line in enumerate(active_steps):
            # last step is different, last step index length -1
            if j < len(active_steps) - 1:
                jump = active_steps[j + 1][1] - active_steps[j][1]
                step_lengths.append((line[0], jump))
            else:
                jump = active_steps[0][1] - active_steps[j][1]
                step_lengths.append((line[0], jump))
        return step_lengths

    def write_step_jump(self, step_lengths):
        """
        write step_jump.csv according to spec.

        :param step_lengths:
        :return:
        """

        headers = ("time", "step_jump")
        with open("step_jump.csv", 'w', newline='\n') as stepfile:
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

    def write_realized_invest_period(self, output_invest_period):
        """
        write to file a list of timesteps as defined by the active timeline of the current solve
        :param filename: filename to write to
        :param timeline: list of tuples containing the period and the timestep
        :return: nothing
        """
        with open("output_periods.csv", 'w') as outfile:
            # prepend with a header
            outfile.write('period\n')
            for item in output_invest_period:
                outfile.write(item + '\n')

    def write_first_status(self, first_state):
        """
        make a file solve_first.csv that contains information if the current solve is the first to be run

        :param first_state: boolean if the current run is the first

        """
        with open("p_model.csv", 'w') as firstfile:
            firstfile.write("modelParam,p_model\n")
            if first_state:
                firstfile.write("solveFirst,1\n")
            else:
                firstfile.write("solveFirst,0\n")

    def write_empty_investment_file(self):
        """
        make a file p_process_invested.csv that will contain capacities of invested and divested processes. For the first solve it will be empty.

        :param first_state: boolean if the current run is the first

        """
        with open("p_process_invested.csv", 'w') as firstfile:
            firstfile.write("process,p_process_invested\n")


def main():
    """
    first read the solve configuration from the input files, then for each solve write the files that are needed
    By that solve into disk. separate the reading into a separate step since the input files need knowledge of multiple solves.
    """
    runner = FlexToolRunner()
    active_time_lists = OrderedDict()
    jump_lists = OrderedDict()
    for solve in runner.solves:
        active_time_list = runner.get_active_time(solve, runner.timeblocks_used_by_periods, runner.timeblocks, runner.timelines)
        active_time_lists[solve] = active_time_list
        jumps = runner.make_step_jump(active_time_list, runner.timelines, runner.timeblocks_used_by_periods, solve)
        jump_lists[solve] = jumps

    #first_steps = runner.get_first_steps(active_time_lists)

    first = True
    for solve in runner.solves:
        foo = active_time_lists[solve]
        runner.write_active_timelines(active_time_lists[solve], 'step_in_use.csv')
        runner.write_step_jump(jump_lists[solve])
        runner.write_step_invest(solve, runner.invest)
        runner.write_realized_invest_period(runner.output_invest_period[solve])
        if first:
            runner.write_first_status(first)
            first = False
            runner.write_empty_investment_file()
        else:
            runner.write_first_status(first)

        model_out, model_err = runner.model_run()
        logging.info(model_out.decode("utf-8"))


if __name__ == '__main__':
    main()
