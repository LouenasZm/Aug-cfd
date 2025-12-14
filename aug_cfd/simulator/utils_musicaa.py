"""
    This module contains some useful functions to run and monitor Musicaa simulations.

    Author: PhD. Camille Matar (March 2025)
    Modified by: L.Zemmour (June 2025)
    Refactored for Hybrid HPC/Local Execution
"""
import os
import re
import subprocess
import logging
import time
import numpy as np

# Ensure these imports match your project structure
from ppModule.iniFiles.read_ini import FeosReader, InfoReader
from ..utils import (custom_input, read_next_line_in_file)

logger = logging.getLogger(__name__)

# =========================================================
# NEW: Standalone Tasks for Executor (Local or Slurm)
# =========================================================
def run_musicaa_task(config: dict, sim_outdir: str,
                     exec_cmd: list[str], computation_type: str) -> bool:
    """
    **Task** that runs on the worker (Compute Node or Local Thread).
    It launches the solver, monitors it, and returns the convergence status.
    
    Returns:
        bool: True if converged, False otherwise.
    """
    # 1. Prepare logging
    log_file = os.path.join(sim_outdir, "musicaa_solver.log")

    with open(log_file, "w", encoding="utf-8") as out_f:
        # 2. Start the process
        # Note: On HPC, exec_cmd should ideally be ['srun', 'exec'] or similar
        logger.info("Starting solver in %s", sim_outdir)
        proc = subprocess.Popen(exec_cmd, cwd=sim_outdir, stdout=out_f, stderr=out_f)

        # 3. Monitor (Blocking call)
        # This function loops until process finishes or we kill it due to convergence
        monitor_sim_progress(proc, config, sim_outdir, exec_cmd, computation_type)

    # 4. Final check
    converged, _ = check_convergence(config, sim_outdir, computation_type)
    return converged

def run_stats_task(config: dict, sim_outdir: str, exec_cmd: list[str]) -> bool:
    """
    **Task** to run the statistics phase.
    """
    # 1. Update INI files for stats
    pre_process_stats(config, sim_outdir, "steady")

    # 2. Run
    log_file = os.path.join(sim_outdir, "musicaa_stats.log")
    with open(log_file, "w", encoding="utf-8") as out_f:
        proc = subprocess.Popen(exec_cmd, cwd=sim_outdir, stdout=out_f, stderr=out_f)
        monitor_sim_progress(proc, config, sim_outdir, exec_cmd, "steady")

    return True

# =========================================================
# Existing Helper Functions
# =========================================================

def monitor_sim_progress(proc: subprocess.Popen,
                         config: dict,
                         sim_outdir: str,
                         computation_type: str,
                         unsteady_step: str = ""):
    """
    **Monitors** a simulation (Running Process).
    Checks for convergence periodically and kills the process if converged.
    """
    # get simulation arguments
    if unsteady_step == "init_2D":
        max_niter = config["simulator"].get("max_niter_init_2D", 200000)
    elif unsteady_step == "init_3D":
        max_niter = config["simulator"].get("max_niter_init_3D", 200000)
    elif config.get("is_stats", False):
        max_niter = config["simulator"].get("max_niter_stats", 200000)
    else:
        config.update({"max_niter_steady_reached": False})
        max_niter = config["simulator"]["convergence_criteria"]["max_niter_steady"]

    while True:
        returncode = proc.poll()

        # computation still running
        if returncode is None:
            converged, niter = check_convergence(config, sim_outdir, computation_type)

            if converged or niter >= max_niter:
                logger.info("Stopping simulation in %s (Converged: %s, Iter: %s)",
                            sim_outdir, converged, niter)
                stop_musicaa(sim_outdir)

                # Wait a bit for the code to write final files and exit cleanly
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()

                if computation_type == "steady" and not config.get("is_stats", False):
                    config.update({"max_niter_steady_reached": True})

                if "n_convergence_check" in config:
                    del config["n_convergence_check"]
                break

            # Sleep to avoid trashing the CPU/FS while polling
            time.sleep(5)

        # computation has completed
        elif returncode == 0:
            break
        else:
            # Crashed
            break

def pre_process_stats(config: dict, sim_outdir: str, computation_type: str):
    """
    **Pre-processes** computation for statistics.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]

    # get current iteration from time.ini
    time_info = get_time_info(sim_outdir)
    ndeb_stats = time_info.get("niter_total", 0)

    # modify param.ini file
    args = {}
    args.update({"from_field": "2"})
    args.update({"Iteration number to start statistics": f"{ndeb_stats + 1}"})
    param_ini = os.path.join(sim_outdir, "param.ini")

    if computation_type == "steady":
        # if steady computation did not fully converge
        if config.get("max_niter_steady_reached", False):
            niter_stats = convergence_criteria.get("niter_stats_steady", 10000)
        else:
            niter_stats = 2
            logger.warning("Steady computation did not fully converge," \
                            "stats computed from last iteration")

    elif computation_type == "unsteady":
        niter_stats = 999999
        # add frequency for QoI convergence check
        try:
            niter_ftt = get_niter_ftt(sim_outdir, config["gmsh"]["chord_length"])
            freqs = read_next_line_in_file(param_ini,
                                    "Output frequencies: screen / stats / fields").split()
            args.update({"Output frequencies: screen / stats / fields":
                         f"{freqs[0]} {freqs[1]} {niter_ftt}"})
        except Exception as e:
            logger.warning("Could not calculate FTT for unsteady stats: %s", e)

    args.update({"Max number of temporal iterations": f"{niter_stats} 3000.0"})
    custom_input(param_ini, args)

def get_niter_ftt(sim_outdir: str, l_ref: float) -> int:
    """
    **Returns** the number of iterations per flow-through time (ftt)
    """
    try:
        pp_feos = FeosReader(file_path=sim_outdir, fluid="air")
        info_reader = InfoReader(file_path=os.path.join(sim_outdir, "info.ini"))

        mach_ref = float(read_next_line_in_file(os.path.join(sim_outdir, "param.ini"),
                                                "Reference Mach"))
        temp_ref = float(read_next_line_in_file(os.path.join(sim_outdir, "param.ini"),
                                                "Reference temperature"))

        c_ref = np.sqrt(pp_feos.feos["Equivalent gamma"]
                        * pp_feos.feos["Gas constant"] * temp_ref)
        u_ref = c_ref * mach_ref

        l_grid = float(read_next_line_in_file(os.path.join(sim_outdir, "param.ini"),
                                             "Scaling value for the grid l_grid"))
        if l_grid != 0:
            l_ref = l_ref * l_grid

        ftt = l_ref / u_ref
        dt = info_reader.get_value(key="dt")
        return int(ftt / dt)
    except Exception as e:
        logger.error("Error calculating FTT: %s", e)
        return 1000

def check_convergence(config: dict, sim_outdir: str, computation_type: str) -> tuple[bool, int]:
    """
    Function to check convergence of a simulation

    Only calls residuals for steady simulation at the moment
    """
    if computation_type == "steady":
        return check_residuals(config, sim_outdir)
    return False, 0

def stop_musicaa(sim_outdir: str):
    """
    **Stops** MUSICAA during execution by writing 'stop' file.
    """
    with open(os.path.join(sim_outdir, "stop"), "w") as stop:
        stop.write("stop")

def get_time_info(sim_outdir: str) -> dict:
    pattern = re.compile(r"^(\d{4}_\d{4})\s*=\s*(\d+)\s*([\d.E+-]+)\s*([\d.E+-]+)$")
    time_info = {"niter_total": 0}

    try:
        with open(os.path.join(sim_outdir, "time.ini"), "r") as file:
            iter_val = 0
            for line in file:
                match = pattern.search(line)
                if match:
                    timestamp = match.group(1)
                    iter_val = int(match.group(2))
                    cputot = float(match.group(3))
                    time_val = float(match.group(4))
                    time_info[timestamp] = {'iter': iter_val, 'cputot': cputot, 'time': time_val}
            time_info["niter_total"] = iter_val
    except FileNotFoundError:
        pass

    return time_info

def check_residuals(config: dict, sim_outdir: str) -> tuple[bool, int]:
    """
    Function to check convergence of residuals in a RANS simulation using Musicaa
    Stores highest value of residual and check if orders of magnitudes have been reached
    
    :param config: Description
    :type config: dict
    :param sim_outdir: Description
    :type sim_outdir: str
    :return: Description
    :rtype: tuple[bool, int]
    """
    convergence_criteria = config["simulator"]["convergence_criteria"]
    residual_convergence_order = convergence_criteria.get("residual_convergence_order", 4)

    if config.get("is_stats", False):
        return False, 0

    param_ini = os.path.join(sim_outdir, "param.ini")
    try:
        nprint = int(re.findall(r'\b\d+\b', read_next_line_in_file(param_ini, "screen"))[0])
        ndeb_rans = int(read_next_line_in_file(param_ini, "ndeb_RANS"))
    except (ValueError, IndexError):
        return False, 0

    if "n_convergence_check" not in config:
        config["n_convergence_check"] = 1

    try:
        res = get_residuals(sim_outdir, nprint, ndeb_rans)
        unwanted = ["nn"]
        try:
            niter = int(res["nn"][-1])
            if niter <= ndeb_rans:
                nvars = 4
                unwanted.append("RANS_var")
            else:
                nvars = 5
        except IndexError:
            return False, 0

        if niter // nprint >= config["n_convergence_check"]:
            config["n_convergence_check"] += 1
            nvars_converged = 0
            for var in set(res) - set(unwanted):
                if not res[var]:
                    return False, niter
                if max(res[var]) - min(res[var]) > residual_convergence_order:
                    nvars_converged += 1

            if niter > ndeb_rans and nvars_converged == nvars:
                return True, niter
            return False, niter
        else:
            return False, niter

    except FileNotFoundError:
        return False, 0

def get_residuals(sim_outdir: str, nprint: int, ndeb_rans: int) -> dict:
    """
    Read residuals from residuals.bin file and returns it for each quantity
    so convergence can be checked
    """

    res = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhoe': [], 'RANS_var': []}
    try:
        f = open(f"{sim_outdir}/residuals.bin", "rb")
        i4_dtype = np.dtype('<i4')
        i8_dtype = np.dtype('<i8')
        f8_dtype = np.dtype('<f8')
        it = 1
        while True:
            try:
                if it == 1:
                    np.fromfile(f, dtype=i4_dtype, count=1)
                else:
                    np.fromfile(f, dtype=i8_dtype, count=1)

                # Check for EOF or read data
                data = np.fromfile(f, dtype=i4_dtype, count=1)
                if len(data) == 0: break
                res['nn'].append(data[0])

                # Read variables
                np.fromfile(f, dtype=f8_dtype, count=1) # skip arg
                res['Rho'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
                np.fromfile(f, dtype=f8_dtype, count=1)
                res['Rhou'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
                np.fromfile(f, dtype=f8_dtype, count=1)
                res['Rhov'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
                np.fromfile(f, dtype=f8_dtype, count=1)
                res['Rhoe'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])

                if it * nprint > ndeb_rans:
                    np.fromfile(f, dtype=f8_dtype, count=1)
                    res['RANS_var'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
                it += 1
            except IndexError:
                break
        f.close()
    except Exception:
        pass
    return res