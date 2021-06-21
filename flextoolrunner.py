import csv
import re
import math
import subprocess
import itertools
import logging
import sys

# [10.44] Kiviluoma Juha
#     Sain viimein työnnettyä päivitykset. Se oli vähän monimutkaisempaa kuin eka ajattelin, mutta ehkä tämä on nyt lähempänä lopullista.
#
#
#
# solves: tässä on lista 'solve' nimistä, jotka pitää ajaa yksittäisinä malliajoina käyttäen Python koodia
# 	block_in_use: kertoo mitkä aikablokit ovat käytössä missäkin solvessa
# 	block_start: kertoo ajanhetken jolloin block alkaa
# 	block_steps: kertoo montako aika-askelta blockiin kuuluu
# 	step_duration: kertoo jokaisen aika-askeleen pituuden tunteina
# 	step_invest: kertoo kullekin solvelle ajanhetket jolloin malli saa investoida uuteen kapasiteettiin (voisi olla tyhjä, jolloin ei saa investoida)
# 	step_jump: montako tuntia pitää edetä että pääsee seuraavan aika-askeleen alkuun. Tämä tulee myöhemmin laskettavaksi Python koodin sisällä, koska sen voi päätellä noista muista tiedoista, mutta mennään eka vaan läpi syötöllä.
#
# Eli näiden pohjalta pitäisi viedä seuraavat tiedostot jokaiselle solvelle:
#
#
#
# steps: kaikki aika-askeleet
# 	steps_in_use: ajanaskeleet jotka ovat kyseisessä malliajossa käytössä (nämä päätellään block_start ja block_steps pohjalta)
# 	step_jump: läpivientinä toistaiseksi
# 	step_duration: läpivienti
# 	[Yesterday 14.26] Kiviluoma Juha
#     step_invest: läpivienti ei riitä, vaan pitää valita se solve ja tulostaa uusi versio tiedostosta jossa on vain sen kyseisen solven 'step_invest' leimat
#
# Varmaan vain steps_in_use tarvitsee päivittää solvejen välillä

def get_solves():
    """
    read in the list of solves return it as a list of strings
    :return:
    """
    with open("solves.csv", 'r') as solvefile:
        header = solvefile.readline()
        solves = solvefile.readlines()
    return [solve.strip() for solve in solves]

def get_blocklist():
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
            except StopIteration:
                break
    return block_id


def get_block_start():
    """
    fetch the start time for each block form the file
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
            except StopIteration:
                break
    return block_strt

def get_block_steps():
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

def get_step_dur():
    """
    read in step durations for all simulation steps
    """
    with open('step_duration.csv', 'r') as blk:
        filereader = csv.reader(blk, delimiter=',')
        headers = next(filereader)
        step_dur = []
        while True:
            try:
                datain = next(filereader)
                step_dur.append((datain[0], datain[1]))
            except StopIteration:
                break
    return step_dur

def get_step_invest():
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

def get_step_jump():
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

def make_steps(start, stop):
    """
    make a list of timesteps available
    :return: list of timesteps
    """
    step_code = "t{0:02d}"
    active_step = start
    steps = []
    while active_step <= stop:
        steps.append(step_code.format(active_step))
        active_step += 1
    return steps

def write_steps(steplist, filename):
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

def get_index_int(a):
    """
    helper function to convert timestamp index in format "tddd" to in ddd
    :param a:
    :return:
    """
    return int(re.search("t(\d+)",a).group(1))

def make_full_timeline(starts, steps):
    """
    full timeline is from the smallest start index to the last possible stop index
    just write directly to file
    :param starts: starts for different solves
    :param steps: lengths of different solves
    :return:
    """
    # find the latest stop and earliest start
    latest_stop = 0
    earliest_start = None
    for item in starts.keys():
        starttime = get_index_int(starts[item])
        stoptime = starttime + float(steps[item])
        if stoptime >= latest_stop:
            latest_stop = stoptime
        if earliest_start is None:
            earliest_start = starttime
        elif (earliest_start is not None) and (starttime < earliest_start):
            earliest_start = starttime
    steplist = []
    step_code = "t{0:02d}"

    for i in range(earliest_start, math.ceil(latest_stop)):
        steplist.append(step_code.format(i))
    write_steps(steplist, 'steps.csv')

def make_block_timeline(start, length):
    """
    make a block timeline, there might be multiple blocks per solve so these blocks might need to be combined for a run
    :param start: start of block
    :param length: length of block
    :return: block timeline
    """
    steplist = []
    step_code = "t{0:02d}"
    startnum = get_index_int(start)
    for i in range(startnum, math.ceil(startnum + float(length))):
        steplist.append(step_code.format(i))
    return steplist

def make_solve_timeline(blocks):
    """
    make a timeline for the full solve, single solve can consist of multiple blocks
    :param blocks: list of block timelines as given by make_block_timeline
    :return:
    """
    steplist =  list(itertools.chain(*blocks))
    write_steps(steplist, 'steps_in_use.csv')


def make_step_invest(solve_code, step_invest):
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

def model_run():
    """
    run the model executable once
    :return:
    """
    modelout = subprocess.Popen(['glpsol.exe', '--model', 'flexModel3.mod', '-d', 'FlexTool3_base_sets.dat'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    stdout, stderr = modelout.communicate()
    #print(stdout.decode("utf-8"))
    #print(stderr)
    return stdout, stderr

def get_blocks(solve, blocklist):
    """
    retunr all block codes that are included in solve
    :param solve:
    :param blocklist:
    :return:
    """
    return [block[1] for block in blocklist if block[0] == solve]


def main():
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    #read the data in
    solves = get_solves()
    blocklist = get_blocklist()
    starts = get_block_start()
    steps = get_block_steps()
    durations = get_step_dur()
    jumps = get_step_jump()
    invest = get_step_invest()
    make_full_timeline(starts, steps)
    for solve in solves:
        active_blocks = get_blocks(solve, blocklist)
        steplist = []
        for block in active_blocks:
            steplist.append(make_block_timeline(starts[block], steps[block]))
        make_solve_timeline(steplist)
        make_step_invest(solve, invest)
        model_out, model_err = model_run()
        logging.info(model_out.decode("utf-8"))

    #model_run()

if __name__ == '__main__':
    main()


