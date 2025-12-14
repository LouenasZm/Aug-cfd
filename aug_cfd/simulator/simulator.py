"""
    Docstring to write some day
"""

import logging
import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import submitit

from ppModule.interface import PostProcessMusicaa
from ppModule.iniFiles.read_ini import ParamBlockReader
from .utils_musicaa import (
    run_musicaa_task, 
    run_stats_task, 
)

from ..utils import (custom_input, check_dir, cp_filelist, rm_filelist)

logger = logging.getLogger(__name__)

class Simulator(ABC):
    def __init__(self, config: dict):
        self.cwd = os.getcwd()
        self.config = config
        self.outdir = config["study"]["outdir"]
        self._process_config()
        self._set_solver_name()
        self.exec_cmd = config["simulator"]["exec_cmd"].split(" ")
        self.files_to_cp = config["simulator"].get("files_to_cp", [])
        self.df_dict = {}


    @abstractmethod
    def kill_all(self): pass

    @abstractmethod
    def execute_sim(self, *args): pass

    @abstractmethod
    def _set_solver_name(self): pass

    @abstractmethod
    def _process_config(self): pass


class MusicaaSimulator(Simulator):
    """
    Refactored to support both Local (ThreadPool) and HPC (Slurm/Submitit) execution.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.active_jobs = [] 
        self.restart = config["simulator"].get("restart", 0)
        self.computation_type = "steady"

        # --- EXECUTOR SETUP ---
        # We determine the platform (Local or HPC)
        self.mode = config.get("platform", "local") 

        if self.mode == "hpc":
            logger.info("Initializing HPC Executor (submitit)")
            # 1. Initialize the Executor in __init__ as requested
            log_folder = os.path.join(self.outdir, "slurm_logs")
            self.executor = submitit.AutoExecutor(folder=log_folder)

            # 2. Set default parameters (Queues, Time limits)
            # Note: Resources (cpus/tasks) will be updated dynamically in get_nproc_simu
            slurm_conf = config.get("slurm", {})
            self.executor.update_parameters(
                timeout_min=slurm_conf.get("time_min", 60),
                slurm_partition=slurm_conf.get("partition", "standard"),
                # Default fallback if get_nproc_simu is never called
                slurm_ntasks=1,
                cpus_per_task=1 
            )
        else:
            logger.info("Initializing Local Executor (ThreadPool)")
            self.executor = ThreadPoolExecutor(
                                max_workers=config["simulator"].get("max_concurrent", 4))

    def get_nproc_simu(self):
        """
        Reads the param_blocks.ini file to get the number of processors for the simulation.
        UPDATES the executor resources dynamically.
        """
        # Set up the ParamBlockReader to read the param_block.ini file
        pp_blocks = ParamBlockReader(self.outdir+"/param_blocks.ini")
        pp_blocks.read_block_info()

        # Calculate Total MPI Ranks (n_procs)
        n_procs = 0
        for block_id in range(1, pp_blocks.block_info["nbloc"]+1):
            n_procs += (pp_blocks.block_info[block_id]['Nb procs']["I"]
                       *pp_blocks.block_info[block_id]['Nb procs']["J"]
                       *pp_blocks.block_info[block_id]['Nb procs']["K"])

        logger.info(f"Calculated required processors: {n_procs}")
        self.config["simulator"]["nproc"] = n_procs

        # --- DYNAMIC RESOURCE UPDATE ---
        if self.mode == "hpc":
            # We assume n_procs corresponds to MPI Ranks (ntasks).
            # If your code uses Threads (OpenMP), change 'slurm_ntasks' to 'cpus_per_task'
            logger.info(f"Updating Slurm Executor to use {n_procs} tasks.")
            self.executor.update_parameters(
                slurm_ntasks=n_procs,
                nodes=1  # Assuming 1 node, or remove to let Slurm decide based on ntasks
            )

        # Update the command string replacing 'nproc'
        exec_cmd = [cmd.replace("nproc", str(n_procs)) for cmd in self.exec_cmd]
        return exec_cmd

    def execute_sim(self, *args):
        """
        Preprocess and executes a simulation using Musicaa
        """
        if len(args) != 4:
            raise ValueError("Expected exactly 4 arguments: candidate, gid, cid, and restart")
        candidate, gid, cid, restart = args

        if gid not in self.df_dict:
            self.df_dict[gid] = {}

        try:
            sim_outdir = self.get_sim_outdir(gid, cid)
            dict_id = {"gid": gid, "cid": cid}

            # Simple check if results exist (customize as needed)
            if os.path.exists(os.path.join(sim_outdir, "stats.bin")):
                # self.df_dict[gid][cid] = self.post_process(dict_id, sim_outdir)
                logger.info(f"g{gid}, c{cid}: results exist (skipped)")
                return

            # Force exception to trigger run logic if files missing
            raise FileNotFoundError

        except (FileNotFoundError, TypeError):
            # 1. Prepare Directory
            sim_outdir = self._pre_process(candidate, gid, cid)

            # 2. Get Resources & Update Executor
            # This calls get_nproc_simu, which updates self.executor.update_parameters()
            exec_cmd = self.get_nproc_simu()

            # 3. Submit
            self.submit_job(sim_outdir, exec_cmd,
                            {"candidate": candidate, "gid": gid, "cid": cid},
                            restart,
                            phase="steady")

    def submit_job(self, sim_outdir, exec_cmd, dict_id, restart, phase="steady"):
        """
        Submits a job to the configured executor.
        """
        logger.info(f"Submitting {phase} job for g{dict_id['gid']}c{dict_id['cid']}")

        if phase == "steady":
            job = self.executor.submit(
                run_musicaa_task,
                self.config,
                sim_outdir,
                exec_cmd,
                "steady"
            )
        elif phase == "stats":
            # Stats might require fewer resources?
            # Optional: self.executor.update_parameters(slurm_ntasks=1) 
            job = self.executor.submit(
                run_stats_task,
                self.config,
                sim_outdir,
                exec_cmd
            )

        self.active_jobs.append({
            "dict_id": dict_id,
            "job": job,
            "restart": restart,
            "phase": phase,
            "sim_outdir": sim_outdir,
            "exec_cmd": exec_cmd
        })

    def monitor_ressources(self) -> int:
        """
        Checks status of submitted jobs and manages workflow.
        """
        still_running = []

        for job_info in self.active_jobs:
            job = job_info["job"]

            # Check if running
            if not job.done():
                still_running.append(job_info)
                continue

            # JOB FINISHED
            dict_id = job_info["dict_id"]
            phase = job_info["phase"]

            try:
                success = job.result() # Blocks if not done, but we checked .done()

                if phase == "steady":
                    if success:
                        logger.info(f"{dict_id} Converged. Submitting stats...")
                        self.submit_job(job_info["sim_outdir"], job_info["exec_cmd"],
                                        dict_id, job_info["restart"], phase="stats")
                    else:
                        logger.warning(f"{dict_id} did not fully converge.")
                        # Logic: Run stats anyway or fail?
                        self.submit_job(job_info["sim_outdir"], job_info["exec_cmd"],
                                        dict_id, job_info["restart"], phase="stats")

                elif phase == "stats":
                    logger.info(f"{dict_id} Stats finished.")
                    # Post process here if needed

            except Exception as e:
                logger.error(f"Job failed {dict_id}: {e}")
                # Restart logic
                if job_info["restart"] < self.restart:
                    logger.info(f"Restarting {dict_id}...")
                    rm_filelist([os.path.join(job_info["sim_outdir"], f)
                                 for f in ["plane*", "restart*"]])

                    self.submit_job(job_info["sim_outdir"], job_info["exec_cmd"],
                                    dict_id, job_info["restart"] + 1, phase="steady")

        self.active_jobs = still_running
        return len(self.active_jobs)

    def kill_all(self):
        for info in self.active_jobs:
            info["job"].cancel()

    # ... (Keep _pre_process and other private methods from your original code) ...
    def _pre_process(self, candidate, gid: int = 0, cid: int = 0):
        """
        Preprocess the simulation execution and returns the execution
        command and directory
        """
        # Get the simulation directory:
        sim_outdir = self.get_sim_outdir(gid=gid, cid=cid)
        check_dir(sim_outdir)
        # Get outidr to fine files to copy:
        files_to_cp = [os.path.join(self.outdir, file) for file in self.files_to_cp]
        # Copy files and executable to directory:
        cp_filelist(files_to_cp,
                    [sim_outdir] * len(self.files_to_cp))
        # Create local config file:
        sim_config = {
            "simulator": self.config["simulator"],
        }
        with open(os.path.join(sim_outdir, "sim_config.json"), "w", encoding="utf-8") as jfile:
            json.dump(sim_config, jfile)
        # =========== Update param_rans.ini file and correction.dat:
        # Create a correction.dat file with vector of coefficient written in it:
        filename = os.path.join(sim_outdir, "correction.dat")
        with open(filename, "w", encoding="utf-8") as f:
            for param in candidate:
                f.write(f"{param}\n")
        logger.info("correction.dat written to %s", filename)
        # Modify param_rans.ini file to read correction.dat:²
        args = {}
        args.update({"file where the correction coefficient is stored": "correction.dat"})
        custom_input(sim_outdir+"/param_rans.ini", args)

        # =========== Update param.ini file to write the directory where the grid is stored:
        filename = os.path.join(sim_outdir, "param.ini")
        args = {}
        args.update({"Directory for grid files": '"../../grid"'})
        custom_input(filename, args)

        logger.info("Directory for candidate %d, generation %d has been set up", cid, gid)
        #tory for grid files": '"../../grid"'})
        return sim_outdir

    def _set_solver_name(self): self.solver_name = "musicaa"
    def _process_config(self): pass