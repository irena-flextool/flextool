import csv
import math
import subprocess
import itertools
import logging
import sys
import os
from collections import OrderedDict

class FlexToolRunner:
    """
    Define Class to run the model and read and recreate the requierd config files:
    """
    def __init__(self) -> None:
        logging.basicConfig(
            stream=sys.stderr,
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        #make a directory for model unit tests
        if not os.path.exists("./tests"):
            os.makedirs("./tests")
        #read the data in
        self.durations, self.steplist = self.get_step_dur()
        self.solves = self.get_solves()
        self.starts = self.get_block_start()
        self.steps = self.get_block_steps()
        self.blocklist = self.get_blocklist()
        self.jumps = self.get_step_jump()
        self.invest = self.get_step_invest()
        self.write_steps(self.steplist, 'steps.csv')
        

        


    def get_solves(self):
        """
        read in the list of solves return it as a list of strings
        :return:
        """
        with open("solves.csv", 'r') as solvefile:
            header = solvefile.readline()
            solves = solvefile.readlines()
        return [solve.strip() for solve in solves]

    def get_blocklist(self):
        """
        block_in_use.csv contains two columns
        solve: name of the solve
        block: name of the block used for a particular solve

        :return list of tuples of solve-block pairs:
        """
        with open('block_in_use.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            block_id = []
            while True:
                try:
                    datain = next(filereader)
                    block_id.append((datain[0], datain[1]))
                    # blockname needs to be in both block_start and block_steps.csv
                    assert datain[1] in self.starts.keys(), "Block {0} not in block_starts.csv".format(datain[1])
                    assert datain[1] in self.steps.keys(), "Block {0} not in block_steps.csv".format(datain[1])
                except StopIteration:
                    break
                except AssertionError as e:
                    logging.error(e)
                    sys.exit(-1)
        return block_id


    def get_block_start(self):
        """
        fetch the start time for each block form the file
        if the start timestep is not found from the list of available 
        steps raise an error and stop execution
        :return: dictionary containing block name:starttime
        """
        with open('block_start.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            block_strt = {}
            while True:
                try:
                    datain = next(filereader)
                    block_strt[datain[0]] = datain[1]
                    assert datain[1] in self.steplist, "Block {0} start time {1} not found in steplist".format(datain[0], datain[1])
                except StopIteration:
                    break
                except AssertionError as e:
                    logging.error(e)
                    sys.exit(-1)


        return block_strt

    def get_block_steps(self):
        """
        fetch the block step count
        :return: dictionary blockname:stepcount
        """
        with open('block_steps.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            block_steps = {}
            while True:
                try:
                    datain = next(filereader)
                    block_steps[datain[0]] = datain[1]
                except StopIteration:
                    break
        return block_steps


    def get_step_dur(self):
        """
        read in step durations for all simulation steps
        step_durations is the only inputfile that contains the full timeline
        return also a list with just the timestep names. 
        Both are needed so this is the easiest shortcut for this.
        :return: list of tuples (timestep name, duration)
        :return: timestep names
        """
        with open('step_duration.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            step_dur = []
            step_names = []
            while True:
                try:
                    datain = next(filereader)
                    step_dur.append((datain[0], datain[1]))
                    step_names.append(datain[0])
                except StopIteration:
                    break
        return step_dur, step_names

    def get_step_invest(self):
        """
        read in step_invest
        :return  a list of tuples (casename, timestep):
        """
        with open('step_invest.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            invest_step = []
            while True:
                try:
                    datain = next(filereader)
                    invest_step.append((datain[0], datain[1]))
                except StopIteration:
                    break
        return invest_step

    def get_step_jump(self):
        """
        get the jump vaue for every timestep,
        :return a list of tuples (step, jump):
        """
        with open('step_jump.csv', 'r') as blk:
            filereader = csv.reader(blk, delimiter=',')
            headers = next(filereader)
            step_jump = []
            while True:
                try:
                    datain = next(filereader)
                    step_jump.append((datain[0], float(datain[1])))
                except StopIteration:
                    break
        return step_jump

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

    def write_steps(self, steplist, filename):
        """
        write to file a list of timestep as defined in steplist. Use the same function to write
        the total set and the scenario dependent one
        :param filename: filename to write to
        :param steplist: list of timestep indexes
        :return:
        """
        with open(filename, 'w') as outfile:
            # prepend with a header
            outfile.write('step\n')
            for item in steplist:
                outfile.write(item)
                outfile.write('\n')


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

    def make_solve_timeline(self, blocks):
        """
        make a timeline for the full solve, single solve can consist of multiple blocks
        :param blocks: list of block timelines as given by make_block_timeline
        :return list of stpes included in the solve:
        """
        steplist =  list(itertools.chain(*blocks))
        #self.write_steps(steplist, 'step_in_use.csv')
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
        modelout = subprocess.Popen(['glpsol.exe', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        stdout, stderr = modelout.communicate()
        #print(stdout.decode("utf-8"))
        #print(stderr)
        return stdout, stderr

    def get_blocks(self, solve, blocklist):
        """
        retunr all block codes that are included in solve
        :param solve:
        :param blocklist:
        :return:
        """
        return [block[1] for block in blocklist if block[0] == solve]

    def make_step_jump(self, steplist, duration):
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
        for i, line in enumerate(duration):
            # duration lines (stepcode, duration)
            if line[0] in steplist:
                active_steps.append((line[0], i))
        # calculate jump length based on step index
        step_lengths = []
        for j, line in enumerate(active_steps):
            # last step is different, last step index length -1
            if j < len(active_steps)-1:
                jump = active_steps[j+1][1] - active_steps[j][1]
                step_lengths.append((line[0], jump))
            else:
                jump = active_steps[0][1]- active_steps[j][1]
                step_lengths.append((line[0], jump))
        return step_lengths

    def write_step_jump(self, step_lengths):
        """
        write step_jump.csv according to spec.

        :param step_lengths:
        :return:
        """

        headers = ("time","step_jump")
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
                starts[name] = (steplists[solve_names[index]][0], steplists[solve_names[index+1]][0])
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


    def write_first_status(self, first_state):
        """
        make a file solve_first.csv that contains information if the current solve is the first to be run

        :param first_state: boolean if the current run is the first

        """
        with open("solve_first.csv",'w') as firstfile:
            firstfile.write("solve_first\n")
            if first_state:
                firstfile.write("true\n")

def main():
    """
    first read the solve configuration from the input files, then for each solve write the files that are needed
    By that solve into disk. separate the reading into a separate step since the input files need knowledge of multiple solves.
    """
    runner = FlexToolRunner()
    steplists = OrderedDict()
    jumplists = OrderedDict()
    for solve in runner.solves:
        active_blocks = runner.get_blocks(solve, runner.blocklist)
        steplist = []
        for block in active_blocks:
            steplist.append(runner.make_block_timeline(runner.starts[block], runner.steps[block]))
        steplist = runner.make_solve_timeline(steplist)
        steplists[solve] = steplist
        jumps  = runner.make_step_jump(steplist, runner.durations)
        jumplists[solve] = jumps
    
    first_steps = runner.get_first_steps(steplists)

    first = True
    for solve in runner.solves:
        runner.write_steps(steplists[solve], 'step_in_use.csv')
        runner.write_step_jump(jumplists[solve])
        runner.write_step_invest(solve, runner.invest)
        runner.write_first_steps(first_steps[solve])
        if first:
            runner.write_first_status(first)
            first = False
        else:
            runner.write_first_status(first)
        
        
        model_out, model_err = runner.model_run()
        logging.info(model_out.decode("utf-8"))
        

if __name__ == '__main__':
    main()


