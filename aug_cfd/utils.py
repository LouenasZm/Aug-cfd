"""
    Module containinig utility functions for AugCfd, taken from Aero-Optim
    (see https://github.com/mschouler/aero-optim/).

"""
import importlib.util
import glob
import json
import logging
import math
import os.path
import shutil
import signal
import subprocess
import time
import pickle
from types import FrameType
import numpy as np


logger = logging.getLogger(__name__)

def check_file(filename: str):
    """
    Makes sure an existing file was given.
    """
    if not os.path.isfile(filename):
        raise Exception(f"ERROR -- <{filename}> could not be found")


def check_dir(dirname: str):
    """
    Makes sure the directory exists and create one if not.
    """
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        logger.info(f"created {dirname} repository")


def configure_logger(logger: logging.Logger, log_filename: str, log_level: int = logging.INFO):
    """
    Configures logger.
    """
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_filename, mode="w")
    formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_log_level_from_verbosity(verbosity: int) -> int:
    """
    Logger level setting taken from:
    https://gitlab.inria.fr/melissa/melissa/-/blob/develop/melissa/utility/logger.py
    """
    if verbosity >= 3:
        return logging.DEBUG
    elif verbosity == 2:
        return logging.INFO
    elif verbosity == 1:
        return logging.WARNING
    elif verbosity == 0:
        return logging.ERROR
    else:
        return logging.DEBUG


def set_logger(logger: logging.Logger, outdir: str, log_name: str, verb: int = 3) -> logging.Logger:
    """
    Returns the set logger.
    """
    log_level = get_log_level_from_verbosity(verb)
    configure_logger(logger, os.path.join(outdir, log_name), log_level)
    return logger


def handle_signal(signo: int, frame: FrameType | None):
    """
    Raises exception in case of interruption signal.
    """
    signame = signal.Signals(signo).name
    logger.info(f"clean handling of {signame} signal")
    raise Exception("Program interruption")


def catch_signal():
    """
    Makes sure interruption signals are catched.
    """
    signals = [signal.SIGINT, signal.SIGPIPE, signal.SIGTERM]
    for s in signals:
        signal.signal(s, handle_signal)


def get_custom_class(filename: str, module_name: str):
    """
    Returns a customized object (evolution, optimizer, simulator or mesh).
    """
    try:
        spec_class = importlib.util.spec_from_file_location(module_name, filename)
        if spec_class and spec_class.loader:
            custom_class = importlib.util.module_from_spec(spec_class)
            spec_class.loader.exec_module(custom_class)
            MyClass = getattr(custom_class, module_name)
            logger.info(f"successfully recovered {module_name}")
            return MyClass
    except Exception:
        logger.warning(f"could not find {module_name} in {filename}")
        return None


def run(cmd: list[str], output: str, timeout: float | None = None):
    """
    Wrapper around subprocess.run that executes cmd and redirects stdout/stderr to output.
    """
    with open(output, "wb") as out:
        subprocess.run(cmd, stdout=out, stderr=out, check=True, timeout=timeout)


def replace_in_file(fname: str, sim_args: dict):
    """
    Updates the file fname by replacing specific strings with others.
    """
    with open(fname, 'r') as file:
        filedata = file.read()
    # Replace the target string
    for key, value in sim_args.items():
        filedata = filedata.replace(key, value)
    # Write the file out again
    with open(fname, 'w') as file:
        file.write(filedata)


def custom_input(fname: str, args: dict):
    """
    Writes a customized input file.

    Args:
        fname: name of the file to modify
        args: dictionary of patterns and their replacement values
    """
    for key, value in args.items():
        modify_next_line_in_file(fname, key, str(value))


def modify_next_line_in_file(fname: str, pattern: str, modif: str):
    """
    Locates the line in fname containing pattern and replaces the next line with modif.
    """
    try:
        with open(fname, 'r') as file:
            filedata = file.readlines()
        # Iterate through the lines and find the line containing pattern
        for i, line in enumerate(filedata):
            if pattern in line:
                # Ensure the next line exists
                if i + 1 < len(filedata):
                    filedata[i + 1] = modif + '\n'
        # Write the modified content back to the file
        with open(fname, 'w') as file:
            file.writelines(filedata)
    except Exception as e:
        logger.error(f"error reading file: {e}")


def read_next_line_in_file(fname: str, pattern: str) -> str:
    """
    Returns the next line of fname containing pattern.
    """
    with open(fname, "r") as file:
        filedata = file.readlines()
    # Iterate through the lines and find the line containing pattern
    for i, line in enumerate(filedata):
        if pattern in line:
            # Ensure the next line exists
            if i + 1 < len(filedata):
                return filedata[i + 1].strip()  # Remove any extra newlines
    raise Exception(f"{pattern} not found in {fname}")


def rm_filelist(deletion_list: list[str]):
    """
    Wrapper around os.remove that deletes all files specified in deletion_list.
    """
    [os.remove(f) for f_pattern in deletion_list for f in glob.glob(f_pattern)]  # type: ignore


def cp_filelist(in_files: list[str], out_files: list[str], move: bool = False):
    """
    Wrapper around shutil.copy that mimics bash cp command.
    It copies all files specified in in_files to out_files if move is set to False.
    If move is set to True, the behavior is changed to the bash mv command.
    """
    for in_f, out_f in zip(in_files, out_files):
        try:
            shutil.copy(in_f, out_f) if not move else shutil.move(in_f, out_f)
        except FileNotFoundError:
            print(f"WARNING -- {in_f} not found")
        except shutil.SameFileError:
            print(f"WARNING -- {in_f} same file as {out_f}")


def mv_filelist(*args):
    """
    Wrapper around shutil.move that mimics bash mv command.
    """
    return cp_filelist(*args, move=True)


def ln_filelist(in_files: list[str], out_files: list[str]):
    """
    Wrapper around os.symlink that mimics bash ln -s command.
    If the symbolic link already exists, the behavior is changed to ln -sf.
    """
    for in_f, out_f in zip(in_files, out_files):
        try:
            os.symlink(in_f, out_f)
        except FileExistsError as e:
            print(f"WARNING -- {e}, symlink will be forced")
            os.symlink(in_f, "tmplink")
            os.rename("tmplink", out_f)


def find_closest_index(range_value: np.ndarray, target_value: float) -> int:
    """
    Returns the index of the closest element to targe_value within range.
    """
    closest_index = 0
    closest_difference = abs(range_value[0] - target_value)

    for i in range(1, len(range_value)):
        difference = abs(range_value[i] - target_value)
        if difference < closest_difference:
            closest_difference = difference
            closest_index = i
    return closest_index


def round_number(n: int | float, direction: str = "", decimals: int = 0) -> int | float:
    """
    Returns the ceiling/floor rounded value of a given number.
    """
    multiplier = 10**decimals
    if direction == "up":
        return math.ceil(n * multiplier) / multiplier
    elif direction == "down":
        return math.floor(n * multiplier) / multiplier
    else:
        return round(n, decimals)

def save(model, filename):
    """
    Save the surrogate model to a file.

    Parameters:
    filename : str
        The name of the file to save the model to.
    """
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def load(filename):
    """
    Load a surrogate model from a file.
    Parameters:
    filename : str
        The name of the file to load the model from.
    """
    try:
        with open(filename, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        logger.error("Surrogate model file not found. Try generating a new surrogate model.")
        return None
