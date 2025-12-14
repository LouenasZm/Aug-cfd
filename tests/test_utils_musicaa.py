# language: python
import os
import pytest
from types import SimpleNamespace
import numpy as np

from aug_cfd.simulator import utils_musicaa as um

class FakePopen:
    """
    Minimal fake Popen replacement that records call args and provides
    basic methods so mocked monitor_sim_progress can accept it.
    """
    last_instance = None
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # store for tests to inspect
        FakePopen.last_instance = self
        # emulate a running process
        self._returncode = None

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._returncode = -9

def test_run_musicaa_task_converged(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    exec_cmd = ["echo", "hello"]
    computation_type = "steady"
    config = {"simulator": {"convergence_criteria": {"max_niter_steady": 10}}}

    # Patch subprocess.Popen to FakePopen
    monkeypatch.setattr(um, "subprocess", SimpleNamespace(Popen=FakePopen))

    # monitor_sim_progress should be called with the proc returned by FakePopen
    def fake_monitor(proc, cfg, outdir, exec_cmd_arg, comp_type, *args, **kwargs):
        # ensure it's the same proc instance created by FakePopen
        assert proc is FakePopen.last_instance
        assert outdir == sim_outdir
        assert exec_cmd_arg == exec_cmd
        assert comp_type == computation_type
        return None

    monkeypatch.setattr(um, "monitor_sim_progress", fake_monitor)

    # check_convergence returns converged
    monkeypatch.setattr(um, "check_convergence", lambda cfg, outdir, comp_type: (True, 42))

    # Run the task
    result = um.run_musicaa_task(config, sim_outdir, exec_cmd, computation_type)

    # Assertions
    assert result is True
    # Ensure Popen was called with the exec_cmd
    popen_inst = FakePopen.last_instance
    assert popen_inst is not None
    # first positional arg should be the command list
    assert popen_inst.args[0] == exec_cmd
    # cwd should be set to sim_outdir
    assert popen_inst.kwargs.get("cwd") == sim_outdir
    # stdout/stderr should be provided (file handles) in kwargs
    assert "stdout" in popen_inst.kwargs and "stderr" in popen_inst.kwargs

    # Check log file was created
    log_path = os.path.join(sim_outdir, "musicaa_solver.log")
    assert os.path.exists(log_path)

def test_run_musicaa_task_not_converged(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    exec_cmd = ["./musicaa"]
    computation_type = "steady"
    config = {"simulator": {"convergence_criteria": {"max_niter_steady": 5}}}

    # Patch subprocess.Popen to FakePopen
    monkeypatch.setattr(um, "subprocess", SimpleNamespace(Popen=FakePopen))

    # monitor_sim_progress should be invoked; make a no-op
    def fake_monitor(proc, cfg, outdir, exec_cmd_arg, comp_type, *args, **kwargs):
        assert proc is FakePopen.last_instance
        return None

    monkeypatch.setattr(um, "monitor_sim_progress", fake_monitor)

    # check_convergence returns not converged
    monkeypatch.setattr(um, "check_convergence", lambda cfg, outdir, comp_type: (False, 0))

    # Run the task
    result = um.run_musicaa_task(config, sim_outdir, exec_cmd, computation_type)

    # Assertions
    assert result is False
    popen_inst = FakePopen.last_instance
    assert popen_inst is not None
    assert popen_inst.args[0] == exec_cmd
    assert popen_inst.kwargs.get("cwd") == sim_outdir

    # Check log file was created
    log_path = os.path.join(sim_outdir, "musicaa_solver.log")
    assert os.path.exists(log_path)

def test_run_stats_task_calls_preprocess_and_runs(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    exec_cmd = ["./musicaa", "--stats"]
    config = {"simulator": {"convergence_criteria": {"max_niter_steady": 10}}}

    # Patch subprocess.Popen to FakePopen
    monkeypatch.setattr(um, "subprocess", SimpleNamespace(Popen=FakePopen))

    # Ensure pre_process_stats is invoked with expected args
    called = {}

    def fake_pre_process(cfg, outdir, comp_type):
        called["args"] = (cfg, outdir, comp_type)
        # no-op

    monkeypatch.setattr(um, "pre_process_stats", fake_pre_process)

    # monitor_sim_progress receives (proc, config, sim_outdir, exec_cmd, "steady")
    def fake_monitor(proc, cfg, outdir, exec_cmd_arg, comp_type, *args, **kwargs):
        assert proc is FakePopen.last_instance
        assert outdir == sim_outdir
        assert exec_cmd_arg == exec_cmd
        assert comp_type == "steady"

    monkeypatch.setattr(um, "monitor_sim_progress", fake_monitor)

    result = um.run_stats_task(config, sim_outdir, exec_cmd)

    assert result is True
    assert called.get("args") == (config, sim_outdir, "steady")

    popen_inst = FakePopen.last_instance
    assert popen_inst is not None
    assert popen_inst.args[0] == exec_cmd
    assert popen_inst.kwargs.get("cwd") == sim_outdir

    log_path = os.path.join(sim_outdir, "musicaa_stats.log")
    assert os.path.exists(log_path)

# language: python
def test_monitor_sim_progress_converged_writes_stop_and_updates_config(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    # Use existing FakePopen from the test module
    proc = FakePopen()
    # No waiting
    monkeypatch.setattr(um.time, "sleep", lambda s: None)
    # Force check_convergence to indicate convergence immediately
    monkeypatch.setattr(um, "check_convergence", lambda cfg, outdir, comp: (True, 1))

    config = {"simulator": {"convergence_criteria": {"max_niter_steady": 10}}}
    stop_path = os.path.join(sim_outdir, "stop")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    um.monitor_sim_progress(proc, config, sim_outdir, "steady")

    # stop file should be created
    assert os.path.exists(stop_path)
    # config flag should be set to True for steady
    assert config.get("max_niter_steady_reached", False) is True
    # n_convergence_check should be removed if present
    assert "n_convergence_check" not in config

def test_monitor_sim_progress_returns_immediately_if_already_completed(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    proc = FakePopen()
    proc._returncode = 0  # simulate finished process
    monkeypatch.setattr(um.time, "sleep", lambda s: None)

    config = {"simulator": {"convergence_criteria": {"max_niter_steady": 5}}}
    stop_path = os.path.join(sim_outdir, "stop")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    # If poll() == 0, function should exit without creating stop
    um.monitor_sim_progress(proc, config, sim_outdir, "steady")
    assert not os.path.exists(stop_path)

def test_monitor_sim_progress_exits_on_crash_without_writing_stop(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    proc = FakePopen()
    proc._returncode = -1  # simulate crashed process
    monkeypatch.setattr(um.time, "sleep", lambda s: None)

    config = {"simulator": {"convergence_criteria": {"max_niter_steady": 5}}}
    stop_path = os.path.join(sim_outdir, "stop")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    um.monitor_sim_progress(proc, config, sim_outdir, "steady")
    assert not os.path.exists(stop_path)

def test_monitor_sim_progress_stops_when_niter_reaches_max(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    proc = FakePopen()
    monkeypatch.setattr(um.time, "sleep", lambda s: None)
    # Configure check_convergence to report niter equal to max_niter
    def fake_check(cfg, outdir, comp):
        return False, cfg["simulator"]["convergence_criteria"]["max_niter_steady"]
    monkeypatch.setattr(um, "check_convergence", fake_check)

    config = {"simulator": {"convergence_criteria": {"max_niter_steady": 7}}}
    stop_path = os.path.join(sim_outdir, "stop")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    um.monitor_sim_progress(proc, config, sim_outdir, "steady")

    assert os.path.exists(stop_path)
    # Ensure the flag was set for steady termination
    assert config.get("max_niter_steady_reached", False) is True
    assert "n_convergence_check" not in config


def test_pre_process_stats_steady_converged(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    # ensure param.ini exists (function calls custom_input only, but presence is realistic)
    (tmp_path / "param.ini").write_text("dummy")

    # capture custom_input call
    recorded = {}
    def fake_custom_input(param_ini, args):
        recorded["param_ini"] = param_ini
        recorded["args"] = args

    monkeypatch.setattr(um, "custom_input", fake_custom_input)
    monkeypatch.setattr(um, "get_time_info", lambda d: {"niter_total": 5})

    config = {
        "simulator": {"convergence_criteria": {"niter_stats_steady": 123}},
        "max_niter_steady_reached": True,
    }

    um.pre_process_stats(config, sim_outdir, "steady")

    assert recorded["param_ini"] == os.path.join(sim_outdir, "param.ini")
    assert recorded["args"]["from_field"] == "2"
    assert recorded["args"]["Iteration number to start statistics"] == "6"
    assert recorded["args"]["Max number of temporal iterations"] == "123 3000.0"


def test_pre_process_stats_steady_not_converged_uses_small_stats(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    (tmp_path / "param.ini").write_text("dummy")

    recorded = {}
    def fake_custom_input(param_ini, args):
        recorded["args"] = args

    monkeypatch.setattr(um, "custom_input", fake_custom_input)
    monkeypatch.setattr(um, "get_time_info", lambda d: {"niter_total": 0})

    config = {"simulator": {"convergence_criteria": {}}} 
                    # no max_niter flag -> treated as not reached

    um.pre_process_stats(config, sim_outdir, "steady")

    assert recorded["args"]["from_field"] == "2"
    assert recorded["args"]["Iteration number to start statistics"] == "1"
    assert recorded["args"]["Max number of temporal iterations"] == "2 3000.0"

def test_pre_process_stats_unsteady_updates_output_freqs_and_max_iter(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)
    (tmp_path / "param.ini").write_text("Output frequencies: screen / stats / fields\ndummy\n")

    recorded = {}
    def fake_custom_input(param_ini, args):
        recorded["args"] = args

    # Simulate get_time_info, get_niter_ftt and read_next_line_in_file
    monkeypatch.setattr(um, "custom_input", fake_custom_input)
    monkeypatch.setattr(um, "get_time_info", lambda d: {"niter_total": 7})
    monkeypatch.setattr(um, "get_niter_ftt", lambda outdir, chord: 777)
    monkeypatch.setattr(um, "read_next_line_in_file", lambda param_ini, key: "10 20 30")

    config = {
        "simulator": {"convergence_criteria": {}},
        "gmsh": {"chord_length": 0.12},
    }

    um.pre_process_stats(config, sim_outdir, "unsteady")

    assert recorded["args"]["from_field"] == "2"
    assert recorded["args"]["Iteration number to start statistics"] == "8"
    assert recorded["args"]["Output frequencies: screen / stats / fields"] == "10 20 777"
    assert recorded["args"]["Max number of temporal iterations"] == "999999 3000.0"

def test_get_niter_ftt_computes_expected_value(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)

    # Provide deterministic FeosReader and InfoReader replacements
    class FakeFeosReader:
        def __init__(self, file_path, fluid):
            self.feos = {"Equivalent gamma": 1.4, "Gas constant": 287.0}

    class FakeInfoReader:
        def __init__(self, file_path):
            pass
        def get_value(self, key="dt"):
            return 0.001  # dt

    monkeypatch.setattr(um, "FeosReader", FakeFeosReader)
    monkeypatch.setattr(um, "InfoReader", FakeInfoReader)

    # Control param values returned by read_next_line_in_file
    def fake_read_next_line(param_ini, key):
        if "Reference Mach" in key:
            return "0.5"
        if "Reference temperature" in key:
            return "300"
        if "Scaling value for the grid l_grid" in key:
            return "0"  # no grid scaling
        raise KeyError(key)

    monkeypatch.setattr(um, "read_next_line_in_file", fake_read_next_line)

    l_ref = 1.23
    result = um.get_niter_ftt(sim_outdir, l_ref)

    # replicate expected calculation
    mach_ref = 0.5
    temp_ref = 300.0
    gamma = 1.4
    R = 287.0
    c_ref = np.sqrt(gamma * R * temp_ref)
    u_ref = c_ref * mach_ref
    ftt = l_ref / u_ref
    dt = 0.001
    expected = int(ftt / dt)

    assert isinstance(result, int)
    assert result == expected

def test_get_niter_ftt_returns_1000_on_error(tmp_path, monkeypatch):
    sim_outdir = str(tmp_path)

    # Make FeosReader raise to trigger exception path
    def raising_feos(*args, **kwargs):
        raise RuntimeError("feos failure")

    monkeypatch.setattr(um, "FeosReader", raising_feos)

    # read_next_line_in_file can be left untouched or mocked; it won't be reached
    res = um.get_niter_ftt(sim_outdir, 0.5)
    assert res == 1000

import os
import numpy as np
import pytest
from aug_cfd.simulator import utils_musicaa as um


def write_sample_residuals_bin(path, nn_values, rho_vals, rhou_vals, rhov_vals,
                               rhoe_vals, rans_vals=None, nprint_first_header=0):
    """
    Helper to write a residuals.bin compatible with get_residuals reader.
    Only supports a single-record-per-iteration layout matching the reader.
    """
    fpath = os.path.join(path, "residuals.bin")
    with open(fpath, "wb") as f:
        it = 1
        for idx, nn in enumerate(nn_values):
            # header: first iterate uses i4, others i8 (we only create small set)
            if it == 1:
                np.array([nprint_first_header], dtype='<i4').tofile(f)
            else:
                np.array([nprint_first_header], dtype='<i8').tofile(f)

            # write nn (i4)
            np.array([nn], dtype='<i4').tofile(f)

            # write sequence of doubles per reader expectations:
            # skip, Rho, skip, Rhou, skip, Rhov, skip, Rhoe
            np.array([0.0], dtype='<f8').tofile(f)
            np.array([rho_vals[idx]], dtype='<f8').tofile(f)
            np.array([0.0], dtype='<f8').tofile(f)
            np.array([rhou_vals[idx]], dtype='<f8').tofile(f)
            np.array([0.0], dtype='<f8').tofile(f)
            np.array([rhov_vals[idx]], dtype='<f8').tofile(f)
            np.array([0.0], dtype='<f8').tofile(f)
            np.array([rhoe_vals[idx]], dtype='<f8').tofile(f)

            # optionally include RANS_var when reader checks (it * nprint > ndeb_rans)
            if rans_vals is not None:
                np.array([0.0], dtype='<f8').tofile(f)   # skip
                np.array([rans_vals[idx]], dtype='<f8').tofile(f)
            it += 1


def test_get_residuals_reads_written_binary(tmp_path):
    sim_outdir = str(tmp_path)

    # Create sample data for two iterations
    nn_vals = [1, 4]
    rho = [1.1, 1.2]
    rhou = [2.1, 2.2]
    rhov = [3.1, 3.2]
    rhoe = [4.1, 4.2]
    rans = [5.1, 5.2]

    # Write residuals.bin that includes RANS_var entries
    write_sample_residuals_bin(sim_outdir, nn_vals, rho, rhou, rhov, rhoe, rans_vals=rans)

    # Call reader with nprint and ndeb_rans ensuring the branch that adds RANS_var is exercised
    res = um.get_residuals(sim_outdir, nprint=1, ndeb_rans=0)

    # Validate keys and values
    assert "nn" in res and res["nn"] == nn_vals
    # floats compared with approximate equality
    assert pytest.approx(res["Rho"]) == rho
    assert pytest.approx(res["Rhou"]) == rhou
    assert pytest.approx(res["Rhov"]) == rhov
    assert pytest.approx(res["Rhoe"]) == rhoe
    assert pytest.approx(res["RANS_var"]) == rans


def test_check_residuals_converged_and_n_convergence_check_increment(monkeypatch, tmp_path):
    sim_outdir = str(tmp_path)
    # Prepare config and convergence criteria
    config = {"simulator": {"convergence_criteria": {"residual_convergence_order": 4}}}

    # Make read_next_line_in_file return desired nprint and ndeb_RANS
    def fake_read_next_line(param_ini, key):
        if key == "screen":
            return "1"         # nprint -> 1
        if key == "ndeb_RANS":
            return "0"         # ndeb_rans -> 0
        return ""
    monkeypatch.setattr(um, "read_next_line_in_file", fake_read_next_line)

    # Provide a fake get_residuals that reports a last nn = 5 (> ndeb_rans)
    # and for each variable returns ranges exceeding:
    fake_res = {
        "nn": [1, 2, 3, 5],
        "Rho": [0.0, 10.0],
        "Rhou": [0.0, 10.0],
        "Rhov": [0.0, 10.0],
        "Rhoe": [0.0, 10.0],
        "RANS_var": [0.0, 10.0],
    }
    monkeypatch.setattr(um, "get_residuals", lambda outdir, nprint, ndeb: fake_res)

    converged, niter = um.check_residuals(config, sim_outdir)

    assert converged is True
    assert niter == 5
    # n_convergence_check should have been created and incremented from 1 to 2
    assert config.get("n_convergence_check", None) == 2


def test_check_residuals_not_converged_due_to_insufficient_variable_spread(monkeypatch, tmp_path):
    sim_outdir = str(tmp_path)
    config = {"simulator": {"convergence_criteria": {"residual_convergence_order": 4}}}

    def fake_read_next_line(param_ini, key):
        if key == "screen":
            return "1"
        if key == "ndeb_RANS":
            return "0"
        return ""
    monkeypatch.setattr(um, "read_next_line_in_file", fake_read_next_line)

    # One variable has too small spread -> not counted as converged -> overall not converged
    fake_res = {
        "nn": [1, 2, 3, 6],
        "Rho": [0.0, 10.0],
        "Rhou": [0.0, 10.0],
        "Rhov": [0.0, 10.0],
        "Rhoe": [0.0, 1.0],      # small spread -> does not exceed threshold 4
        "RANS_var": [0.0, 10.0],
    }
    monkeypatch.setattr(um, "get_residuals", lambda outdir, nprint, ndeb: fake_res)

    converged, niter = um.check_residuals(config, sim_outdir)

    assert converged is False
    assert niter == 6
    # n_convergence_check should exist and have been incremented
    assert config.get("n_convergence_check", None) == 2


def test_check_residuals_returns_false_when_is_stats_set():
    # ensure early return when is_stats True
    config = {"simulator": {"convergence_criteria": {}}, "is_stats": True}
    conv, niter = um.check_residuals(config, "/nonexistent")
    assert conv is False
    assert niter == 0

def _write_residuals_bin(sim_outdir, nn_vals, rho_vals, rhou_vals, rhov_vals, rhoe_vals, rans_vals=None):
    """Write a residuals.bin compatible with get_residuals reader."""
    fpath = os.path.join(sim_outdir, "residuals.bin")
    i4_dtype = np.dtype('<i4')
    i8_dtype = np.dtype('<i8')
    f8_dtype = np.dtype('<f8')

    with open(fpath, "wb") as f:
        it = 1
        for idx, nn in enumerate(nn_vals):
            # header: i4 for first record, i8 for subsequent (reader discards it)
            if it == 1:
                np.array([0], dtype=i4_dtype).tofile(f)
            else:
                np.array([0], dtype=i8_dtype).tofile(f)

            # write nn (i4)
            np.array([nn], dtype=i4_dtype).tofile(f)

            # write sequence of doubles in order expected by reader:
            # skip, Rho, skip, Rhou, skip, Rhov, skip, Rhoe
            np.array([0.0], dtype=f8_dtype).tofile(f)
            np.array([rho_vals[idx]], dtype=f8_dtype).tofile(f)
            np.array([0.0], dtype=f8_dtype).tofile(f)
            np.array([rhou_vals[idx]], dtype=f8_dtype).tofile(f)
            np.array([0.0], dtype=f8_dtype).tofile(f)
            np.array([rhov_vals[idx]], dtype=f8_dtype).tofile(f)
            np.array([0.0], dtype=f8_dtype).tofile(f)
            np.array([rhoe_vals[idx]], dtype=f8_dtype).tofile(f)

            # optionally include RANS_var (skip then value)
            if rans_vals is not None:
                np.array([0.0], dtype=f8_dtype).tofile(f)
                np.array([rans_vals[idx]], dtype=f8_dtype).tofile(f)

            it += 1


def test_get_residuals_reads_binary_with_rans(tmp_path):
    sim_outdir = str(tmp_path)
    # prepare two iterations where second iteration should include RANS_var depending on nprint/ndeb
    nn_vals = [1, 3]
    rho = [1.0, 1.1]
    rhou = [2.0, 2.1]
    rhov = [3.0, 3.1]
    rhoe = [4.0, 4.1]
    rans = [5.0, 5.1]

    _write_residuals_bin(sim_outdir, nn_vals, rho, rhou, rhov, rhoe, rans_vals=rans)

    # use nprint=1 and ndeb_rans=0 so both records should include RANS_var (it * nprint > ndeb_rans)
    res = um.get_residuals(sim_outdir, nprint=1, ndeb_rans=0)

    assert res["nn"] == nn_vals
    assert np.allclose(res["Rho"], rho)
    assert np.allclose(res["Rhou"], rhou)
    assert np.allclose(res["Rhov"], rhov)
    assert np.allclose(res["Rhoe"], rhoe)
    assert np.allclose(res["RANS_var"], rans)


def test_get_residuals_handles_missing_file(tmp_path):
    sim_outdir = str(tmp_path)
    # no residuals.bin -> should return dict with empty lists
    res = um.get_residuals(sim_outdir, nprint=1, ndeb_rans=0)
    assert isinstance(res, dict)
    assert res["nn"] == []
    assert res["Rho"] == []
    assert res["RANS_var"] == []


def test_check_residuals_converged_increments_n_convergence_check(monkeypatch, tmp_path):
    sim_outdir = str(tmp_path)

    # config with convergence threshold default 4
    config = {"simulator": {"convergence_criteria": {"residual_convergence_order": 4}}}

    # make param parsing succeed: screen -> "1", ndeb_RANS -> "0"
    def fake_read_next_line(param_ini, key):
        if "screen" in key:
            return "1"
        if "ndeb_RANS" in key:
            return "0"
        return ""
    monkeypatch.setattr(um, "read_next_line_in_file", fake_read_next_line)

    # make get_residuals return values with sufficient spread (>4) for all variables
    fake_res = {
        "nn": [1, 2, 5],
        "Rho": [0.0, 10.0],
        "Rhou": [0.0, 10.0],
        "Rhov": [0.0, 10.0],
        "Rhoe": [0.0, 10.0],
        "RANS_var": [0.0, 10.0],
    }
    monkeypatch.setattr(um, "get_residuals", lambda outdir, nprint, ndeb: fake_res)

    converged, niter = um.check_residuals(config, sim_outdir)

    assert converged is True
    assert niter == 5
    # n_convergence_check should have been initialized to 1 then incremented to 2
    assert config.get("n_convergence_check") == 2


def test_check_residuals_not_converged_due_to_small_spread(monkeypatch, tmp_path):
    sim_outdir = str(tmp_path)
    config = {"simulator": {"convergence_criteria": {"residual_convergence_order": 4}}}

    def fake_read_next_line(param_ini, key):
        if "screen" in key:
            return "1"
        if "ndeb_RANS" in key:
            return "0"
        return ""
    monkeypatch.setattr(um, "read_next_line_in_file", fake_read_next_line)

    # one variable has too small spread -> not all variables converge
    fake_res = {
        "nn": [1, 2, 6],
        "Rho": [0.0, 10.0],
        "Rhou": [0.0, 10.0],
        "Rhov": [0.0, 10.0],
        "Rhoe": [0.0, 1.0],   # small spread
        "RANS_var": [0.0, 10.0],
    }
    monkeypatch.setattr(um, "get_residuals", lambda outdir, nprint, ndeb: fake_res)

    converged, niter = um.check_residuals(config, sim_outdir)

    assert converged is False
    assert niter == 6
    assert config.get("n_convergence_check") == 2


def test_check_residuals_early_return_when_is_stats_true():
    config = {"simulator": {"convergence_criteria": {}}, "is_stats": True}
    conv, niter = um.check_residuals(config, "/does/not/matter")
    assert conv is False
    assert niter == 0


def test_check_residuals_param_parse_failure_returns_false(monkeypatch, tmp_path):
    sim_outdir = str(tmp_path)
    config = {"simulator": {"convergence_criteria": {}}}

    # make read_next_line_in_file return a non-digit so int(...) fails -> ValueError branch
    monkeypatch.setattr(um, "read_next_line_in_file", lambda param_ini, key: "no-digits-here")

    conv, niter = um.check_residuals(config, sim_outdir)
    assert conv is False
    assert niter == 0

def test_get_time_info_parses_time_ini(tmp_path):
    sim_outdir = str(tmp_path)
    content = (
        "2025_0001 = 10  123.45  67.89\n"
        "some irrelevant line that should be ignored\n"
        "2025_0002=20 1.23E+02 3.45E+01\n"
    )
    (tmp_path / "time.ini").write_text(content)

    info = um.get_time_info(sim_outdir)

    assert "2025_0001" in info
    assert "2025_0002" in info
    assert info["2025_0001"]["iter"] == 10
    assert info["2025_0002"]["iter"] == 20
    assert info["2025_0001"]["cputot"] == pytest.approx(123.45)
    assert info["2025_0001"]["time"] == pytest.approx(67.89)
    # scientific notation parsed correctly
    assert info["2025_0002"]["cputot"] == pytest.approx(123.0)
    assert info["2025_0002"]["time"] == pytest.approx(34.5)
    # niter_total should be the last parsed iter value
    assert info["niter_total"] == 20


def test_get_time_info_missing_file_returns_default(tmp_path):
    sim_outdir = str(tmp_path)  # no time.ini created
    info = um.get_time_info(sim_outdir)
    assert isinstance(info, dict)
    assert info == {"niter_total": 0}