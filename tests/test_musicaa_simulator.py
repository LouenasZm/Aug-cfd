"""
    Unit tests for the MusicaaSimulator class in aug_cfd.simulator.simulator module.
"""


import pytest
from unittest.mock import MagicMock, mock_open, call
import os

# Absolute import as requested
from aug_cfd.simulator.simulator import MusicaaSimulator

@pytest.fixture
def base_config():
    return {
        "study": {"outdir": "/tmp/test_study"},
        "simulator": {
            "exec_cmd": "mpirun -np nproc ./solver",
            "files_to_cp": ["input.dat"],
            "restart": 2,
            "max_concurrent": 4
        },
        "platform": "local",
        "slurm": {"partition": "debug", "time_min": 30}
    }

@pytest.fixture
def simulator(base_config, mocker):
    """Fixture for a local simulator instance with mocked dependencies."""
    # Mock os.getcwd to avoid side effects
    mocker.patch("os.getcwd", return_value="/tmp")
    # Mock ThreadPoolExecutor for default local init
    mocker.patch("aug_cfd.simulator.simulator.ThreadPoolExecutor")

    sim = MusicaaSimulator(base_config)

    # Mock the get_sim_outdir method (likely from base class not shown in excerpt)
    sim.get_sim_outdir = MagicMock(side_effect=lambda gid, cid: f"/tmp/test_study/g{gid}/c{cid}")
    return sim

def test_init_local(base_config, mocker):
    """Test initialization in local mode."""
    mock_tpe = mocker.patch("aug_cfd.simulator.simulator.ThreadPoolExecutor")
    sim = MusicaaSimulator(base_config)

    assert sim.mode == "local"
    mock_tpe.assert_called_with(max_workers=4)
    assert isinstance(sim.executor, MagicMock)

def test_init_hpc(base_config, mocker):
    """Test initialization in HPC mode."""
    config = base_config.copy()
    config["platform"] = "hpc"
    mock_submitit = mocker.patch("aug_cfd.simulator.simulator.submitit")

    sim = MusicaaSimulator(config)

    assert sim.mode == "hpc"
    mock_submitit.AutoExecutor.assert_called_once()

    # Verify default parameters update
    sim.executor.update_parameters.assert_called_with(
        timeout_min=30,
        slurm_partition="debug",
        slurm_ntasks=1,
        cpus_per_task=1
    )

def test_get_nproc_simu(simulator, mocker):
    """Test processor calculation from param blocks."""
    # Mock ParamBlockReader
    mock_reader_cls = mocker.patch("aug_cfd.simulator.simulator.ParamBlockReader")
    mock_reader = mock_reader_cls.return_value

    # Mock block info: 
    # Block 1: 2*2*1 = 4 procs
    # Block 2: 1*1*2 = 2 procs
    # Total = 6
    mock_reader.block_info = {
        "nbloc": 2,
        1: {'Nb procs': {"I": 2, "J": 2, "K": 1}},
        2: {'Nb procs': {"I": 1, "J": 1, "K": 2}}
    }

    cmd = simulator.get_nproc_simu()

    assert simulator.config["simulator"]["nproc"] == 6
    assert cmd == ["mpirun", "-np", "6", "./solver"]
    mock_reader.read_block_info.assert_called_once()

def test_get_nproc_simu_hpc_update(base_config, mocker):
    """Test that HPC executor parameters are updated dynamically."""
    config = base_config.copy()
    config["platform"] = "hpc"
    mocker.patch("aug_cfd.simulator.simulator.submitit")

    sim = MusicaaSimulator(config)
    sim.get_sim_outdir = MagicMock()

    # Mock reader for 10 procs
    mock_reader_cls = mocker.patch("aug_cfd.simulator.simulator.ParamBlockReader")
    mock_reader = mock_reader_cls.return_value
    mock_reader.block_info = {
        "nbloc": 1,
        1: {'Nb procs': {"I": 10, "J": 1, "K": 1}}
    }

    sim.get_nproc_simu()

    # Verify dynamic update
    sim.executor.update_parameters.assert_called_with(
        slurm_ntasks=10,
        nodes=1
    )

def test_execute_sim_results_exist(simulator, mocker):
    """Test that simulation is skipped if results exist."""
    mocker.patch("os.path.exists", return_value=True) # stats.bin exists
    spy_submit = mocker.spy(simulator, "submit_job")

    simulator.execute_sim([1.0], 0, 0, 0)

    spy_submit.assert_not_called()
    assert len(simulator.active_jobs) == 0

def test_execute_sim_new_run(simulator, mocker):
    """Test execution flow for a new simulation."""
    mocker.patch("os.path.exists", return_value=False) # stats.bin missing

    # Mock internal methods to isolate flow
    mocker.patch.object(simulator, "_pre_process", return_value="/path/to/sim")
    mocker.patch.object(simulator, "get_nproc_simu", return_value=["cmd"])
    spy_submit = mocker.spy(simulator, "submit_job")

    candidate = [1.0]
    simulator.execute_sim(candidate, 0, 0, 0)

    simulator._pre_process.assert_called_with(candidate, 0, 0)
    simulator.get_nproc_simu.assert_called()
    spy_submit.assert_called_with(
        "/path/to/sim", ["cmd"], 
        {"candidate": candidate, "gid": 0, "cid": 0}, 
        0, phase="steady"
    )

def test_execute_sim_invalid_args(simulator):
    """Test argument validation."""
    with pytest.raises(ValueError):
        simulator.execute_sim("not enough args")

def test_submit_job_steady(simulator, mocker):
    """Test submitting a steady phase job."""
    mock_run_musicaa = mocker.patch("aug_cfd.simulator.simulator.run_musicaa_task")
    simulator.executor = MagicMock()

    simulator.submit_job("/out", ["cmd"], {"gid":1, "cid":1}, 0, "steady")

    simulator.executor.submit.assert_called_with(
        mock_run_musicaa, simulator.config, "/out", ["cmd"], "steady"
    )
    assert len(simulator.active_jobs) == 1
    assert simulator.active_jobs[0]["phase"] == "steady"

def test_submit_job_stats(simulator, mocker):
    """Test submitting a stats phase job."""
    mock_run_stats = mocker.patch("aug_cfd.simulator.simulator.run_stats_task")
    simulator.executor = MagicMock()

    simulator.submit_job("/out", ["cmd"], {"gid":1, "cid":1}, 0, "stats")

    simulator.executor.submit.assert_called_with(
        mock_run_stats, simulator.config, "/out", ["cmd"]
    )
    assert simulator.active_jobs[0]["phase"] == "stats"

def test_monitor_resources_running(simulator):
    """Test monitoring when job is still running."""
    mock_job = MagicMock()
    mock_job.done.return_value = False

    simulator.active_jobs.append({
        "job": mock_job,
        "dict_id": {"gid": 0, "cid": 0}
    })

    count = simulator.monitor_ressources()
    assert count == 1
    assert len(simulator.active_jobs) == 1

def test_monitor_resources_steady_success(simulator, mocker):
    """Test monitoring when steady job succeeds (should submit stats)."""
    mock_job = MagicMock()
    mock_job.done.return_value = True
    mock_job.result.return_value = True # Success

    job_info = {
        "job": mock_job,
        "dict_id": {"gid": 0, "cid": 0},
        "phase": "steady",
        "sim_outdir": "/out",
        "exec_cmd": ["cmd"],
        "restart": 0
    }
    simulator.active_jobs.append(job_info)

    spy_submit = mocker.spy(simulator, "submit_job")

    remaining = simulator.monitor_ressources()

    assert remaining == 0
    spy_submit.assert_called_with("/out", ["cmd"], {"gid": 0, "cid": 0}, 0, phase="stats")

def test_monitor_resources_failure_restart(simulator, mocker):
    """Test monitoring when job fails and restart is allowed."""
    mock_rm = mocker.patch("aug_cfd.simulator.simulator.rm_filelist")

    mock_job = MagicMock()
    mock_job.done.return_value = True
    mock_job.result.side_effect = Exception("Crash")

    # Tell the executor to return this specific mock when submit is called
    simulator.executor.submit.return_value = mock_job

    job_info = {
        "job": mock_job,
        "dict_id": {"gid": 0, "cid": 0},
        "phase": "steady",
        "sim_outdir": "/out",
        "exec_cmd": ["cmd"],
        "restart": 0 # < max restart (2)
    }
    simulator.active_jobs.append(job_info)

    spy_submit = mocker.spy(simulator, "submit_job")

    remaining = simulator.monitor_ressources()

    assert remaining == 0
    mock_rm.assert_called()
    # Should submit steady again with restart + 1
    spy_submit.assert_called_with("/out", ["cmd"], {"gid": 0, "cid": 0}, 2, phase="steady")

def test_monitor_resources_failure_no_restart(simulator, mocker):
    """Test monitoring when job fails and max restarts reached."""
    mock_job = MagicMock()
    mock_job.done.return_value = True
    mock_job.result.side_effect = Exception("Crash")

    job_info = {
        "job": mock_job,
        "dict_id": {"gid": 0, "cid": 0},
        "phase": "steady",
        "sim_outdir": "/out",
        "exec_cmd": ["cmd"],
        "restart": 2 # == max restart (2)
    }
    simulator.active_jobs.append(job_info)

    spy_submit = mocker.spy(simulator, "submit_job")

    remaining = simulator.monitor_ressources()

    assert remaining == 0
    spy_submit.assert_not_called()

def test_pre_process(simulator, mocker):
    """Test pre-processing logic (files, config, custom_input)."""
    mocker.patch("aug_cfd.simulator.simulator.check_dir")
    mocker.patch("aug_cfd.simulator.simulator.cp_filelist")
    mock_custom = mocker.patch("aug_cfd.simulator.simulator.custom_input")

    mock_open_obj = mock_open()
    mocker.patch("builtins.open", mock_open_obj)

    outdir = simulator._pre_process([0.5, 0.6], 0, 0)

    assert outdir == "/tmp/test_study/g0/c0"
    assert mock_custom.call_count == 2

    # Check file writing for correction.dat
    handle = mock_open_obj()
    handle.write.assert_any_call("0.5\n")
    handle.write.assert_any_call("0.6\n")

def test_kill_all(simulator):
    """Test killing all active jobs."""
    mock_job = MagicMock()
    simulator.active_jobs.append({"job": mock_job})

    simulator.kill_all()

    mock_job.cancel.assert_called_once()
