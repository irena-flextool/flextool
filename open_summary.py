import subprocess
import os


def open_summary(filepath):
    os.startfile(filepath)

if __name__ == '__main__':
    open_summary("output\summary_solve.csv")