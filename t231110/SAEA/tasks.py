"""SAEA tasks"""
import sys
sys.path.append(r"/home/robin/projects/t231101/t231110")

import os
import itertools
import subprocess
from tqdm import tqdm
from SAEA.Exp import Exp
from SAEA.utils.common1 import get_id, get_time_str

def get_all_exp_settings():
    # problems = [
    #     "Ackley01",
    #     "Rastrigin",
    #     "F12017",
    #     "F32017",
    #     "F42017",
    #     "F52017",
    #     "F62017",
    #     "Himmelblau",
    #     "Griewank",
    # ]
    problems = [
        # "F62005",
        "Ackley01",
        # "F72005",
    ]

    algs = [
        "HSAEA",
        # "DE_SAEA_Base",
        # "FWA_SAEA",
        # "SAEA",
    ]

    dims = [
        10,
    ]

    fit_max = [
        1000,
    ]

    selection_strategies = [
        # "JustGlobal",
        "JustLocal",
        # "HalfHalf",
        # "MixedLinearly",
        # "MixedNonlinearly",
    ]

    return itertools.product(problems, algs, dims, fit_max, selection_strategies)

def init():
    os.makedirs("./exp", exist_ok=True)

def run_tasks():
    init()
    exps = get_all_exp_settings()
    exp_list = list(exps)

    with tqdm(total=len(exp_list), desc="Processing", unit="it") as pbar:
        for exp in exp_list:
            problem, alg, dim, fit_max, ss = exp
            id = get_id()
            os.makedirs(f"./exp/{id}", exist_ok=True)
            command = f"python script.py -p {problem} -a {alg} -i {id} -d {dim} -m {fit_max} -s {ss}"
            with open(f"./exp/{id}/output.txt", "w", encoding="utf-8") as output_file, open(
                "./log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{get_time_str()} {command}\n")
                process = subprocess.Popen(
                    command, shell=True, stdout=output_file, stderr=subprocess.STDOUT)
                stdout, stderr = process.communicate()
                if stderr:
                    print(stderr)
                    log_file.write(stderr)
            pbar.update(1)

run_tasks()