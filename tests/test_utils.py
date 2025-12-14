# python
import signal
import pathlib
import logging

import numpy as np
import pytest

from aug_cfd import utils

# --- file / dir helpers ----------------------------------------------------

def test_check_file_and_check_dir(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    # check_file should not raise for existing file
    utils.check_file(str(f))
    # missing file should raise
    with pytest.raises(Exception):
        utils.check_file(str(tmp_path / "nofile"))

    d = tmp_path / "newdir"
    assert not d.exists()
    utils.check_dir(str(d))
    assert d.exists()
    # idempotent
    utils.check_dir(str(d))


def test_cp_mv_ln_rm_filelist(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("A")
    utils.cp_filelist([str(a)], [str(b)])
    assert b.exists()
    # move (mv_filelist wrapper)
    c = tmp_path / "c.txt"
    utils.mv_filelist([str(b)], [str(c)])
    assert c.exists() and not b.exists()

    # symlink
    link = tmp_path / "link.txt"
    utils.ln_filelist([str(c)], [str(link)])
    assert link.exists()
    # rm_filelist (supports glob)
    utils.rm_filelist([str(tmp_path / "*.txt")])
    assert not any(p.exists() for p in [a, c, link])


# --- logger & levels ------------------------------------------------------

def test_get_and_set_logger(tmp_path):
    l = logging.getLogger("aug_cfd_test")
    logfile = tmp_path / "log.txt"
    utils.configure_logger(l, str(logfile), logging.INFO)
    # handler added
    assert any(isinstance(h, logging.FileHandler) for h in l.handlers)
    # set_logger returns same logger and creates file handler with path under outdir
    l2 = logging.getLogger("aug_cfd_test2")
    out = tmp_path / "outdir"
    out.mkdir()
    utils.set_logger(l2, str(out), "o.log", verb=2)
    assert any(isinstance(h, logging.FileHandler) for h in l2.handlers)


def test_get_log_level_from_verbosity():
    assert utils.get_log_level_from_verbosity(3) == logging.DEBUG
    assert utils.get_log_level_from_verbosity(2) == logging.INFO
    assert utils.get_log_level_from_verbosity(1) == logging.WARNING
    assert utils.get_log_level_from_verbosity(0) == logging.ERROR
    assert utils.get_log_level_from_verbosity(-1) == logging.DEBUG


# --- signal handling ------------------------------------------------------

def test_handle_and_catch_signal_restore():
    # store old handlers to restore
    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    try:
        utils.catch_signal()
        # handler should be our function
        assert signal.getsignal(signal.SIGINT) == utils.handle_signal
        # calling handle_signal raises Exception
        with pytest.raises(Exception):
            utils.handle_signal(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


# --- dynamic import -------------------------------------------------------

def test_get_custom_class(tmp_path):
    module_file = tmp_path / "MyClass.py"
    module_file.write_text(
        "class MyClass:\n"
        "    def __init__(self):\n"
        "        self.v = 3\n"
    )
    cls = utils.get_custom_class(str(module_file), "MyClass")
    assert cls is not None
    inst = cls()
    assert inst.v == 3



def test_run_captures_output(tmp_path):
    out = tmp_path / "out.txt"
    # simple echo via sh -c to work cross-shell
    utils.run(["/bin/echo", "hello"], str(out))
    content = out.read_text().strip()
    assert "hello" in content

# --- file content replacement ---------------------------------------------
def test_replace_in_file_replaces_all(tmp_path):
    f = tmp_path / "replace.txt"
    f.write_text("Hello KEY world. KEY!")
    utils.replace_in_file(str(f), {"KEY": "there"})
    assert f.read_text() == "Hello there world. there!"

def test_replace_in_file_no_change_when_no_match(tmp_path):
    f = tmp_path / "nochange.txt"
    original = "This file has nothing to replace."
    f.write_text(original)
    utils.replace_in_file(str(f), {"NON_EXISTENT": "X"})
    assert f.read_text() == original

def test_custom_input_changes_next_lines_multiple_occurrences(tmp_path: pathlib.Path):
    p = tmp_path / "input.txt"
    p.write_text("PARAM_A\noldA\nmiddle\nPARAM_A\noldA2\n")
    utils.custom_input(str(p), {"PARAM_A": "newA"})
    assert p.read_text() == "PARAM_A\nnewA\nmiddle\nPARAM_A\nnewA\n"

def test_custom_input_no_change_when_pattern_missing(tmp_path: pathlib.Path):
    p = tmp_path / "input2.txt"
    original = "LINE1\nvalue1\nLINE2\nvalue2\n"
    p.write_text(original)
    utils.custom_input(str(p), {"NON_EXISTENT": "42"})
    assert p.read_text() == original

def test_read_next_line_basic(tmp_path):
    p = tmp_path / "basic.txt"
    content = "HEADER\nPATTERN\nnext_line_value\nTAIL\n"
    p.write_text(content)
    assert utils.read_next_line_in_file(str(p), "PATTERN") == "next_line_value"

def test_read_next_line_multiple_occurrences_returns_first(tmp_path):
    p = tmp_path / "multi.txt"
    content = "PAT\nfirst_next\nmiddle\nPAT\nsecond_next\n"
    p.write_text(content)
    # should return the next line after the first occurrence
    assert utils.read_next_line_in_file(str(p), "PAT") == "first_next"

def test_read_next_line_pattern_at_end_raises(tmp_path):
    p = tmp_path / "end.txt"
    content = "LINE1\nLINE2\nPAT_AT_END\n"
    p.write_text(content)
    with pytest.raises(Exception):
        utils.read_next_line_in_file(str(p), "PAT_AT_END")

def test_read_next_line_missing_pattern_raises(tmp_path):
    p = tmp_path / "missing.txt"
    content = "A\nB\nC\n"
    p.write_text(content)
    with pytest.raises(Exception):
        utils.read_next_line_in_file(str(p), "NO_SUCH_PATTERN")

# --- utils numeric --------------------------------------------------------

def test_find_closest_index_and_round_number():
    arr = np.array([0.0, 1.5, 3.2, 7.9])
    assert utils.find_closest_index(arr, 3.0) == 2
    assert utils.round_number(3.14159, "", 2) == round(3.14159, 2)
    assert utils.round_number(3.14159, "up", 2) == pytest.approx(3.15)
    assert utils.round_number(3.14159, "down", 2) == pytest.approx(3.14)

# --- pickle save/load ----------------------------------------------------
def test_save_and_load(tmp_path):
    obj = {"a": 1, "b": [1, 2, 3]}
    fn = tmp_path / "model.pkl"
    utils.save(obj, str(fn))
    loaded = utils.load(str(fn))
    assert loaded == obj
    # missing file returns None
    assert utils.load(str(tmp_path / "no-such-file.pkl")) is None
