"""Post-freeze smoke tests for the bundled ome_tiff_converter executable.

Invoked by ``ome_tiff_converter.exe --smoke-test`` from CI. A failure here
means the .exe cannot import a critical dependency or the bundled
ome_tiff_converter module is structurally broken; either way the build is
not safe to ship.

Each test is a small, fast check. The goal is to verify that PyInstaller
included everything we need, not to re-run the full converter test suite.
"""
import os
import sys
import tempfile

import numpy as np


def _run_one(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name} -- {type(e).__name__}: {e}")
        return False


def run():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    results = []

    def t_pyqt6():
        from PyQt6.QtWidgets import QApplication, QWidget  # noqa: F401
        from PyQt6.QtCore import Qt  # noqa: F401
        from PyQt6.QtGui import QFont  # noqa: F401

    def t_pyqt6_app_instance():
        # Confirm Qt can actually find its platform plugin and start an
        # offscreen QApplication. This is the most common Windows-bundle
        # failure mode (missing platform DLL).
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        assert app is not None

    def t_tifffile_roundtrip():
        import tifffile
        arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = f.name
        try:
            tifffile.imwrite(path, arr)
            back = tifffile.imread(path)
            assert back.shape == (8, 8)
            assert (back == arr).all()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def t_import_converter():
        import ome_tiff_converter as otc
        for sym in ("AcquisitionConverter", "OMETiffWriter", "main", "gui_main"):
            assert hasattr(otc, sym), f"ome_tiff_converter missing {sym!r}"

    def t_parse_stem_fallback():
        # Without a configurations.xml the parser falls back to positional
        # split. This verifies the fallback path is reachable in the bundle
        # and that the static logic still parses a typical Squid filename.
        from ome_tiff_converter import AcquisitionConverter
        c = AcquisitionConverter("/nonexistent")  # configurations.xml absent
        parsed = c._parse_stem("manual_0_0_Fluorescence_488_nm_Ex")
        assert parsed == ("manual", "0", 0, "Fluorescence_488_nm_Ex"), parsed

    def t_end_to_end_convert():
        # Synthesize a tiny acquisition and run a real conversion through
        # the bundled converter. If this passes, the whole pipeline
        # (filename parse -> tifffile memmap write -> read-back) works
        # in the frozen environment.
        import json
        import shutil
        import tifffile
        from ome_tiff_converter import main as converter_main

        work = tempfile.mkdtemp(prefix="otc_smoke_")
        try:
            acq = os.path.join(work, "acq")
            out = os.path.join(work, "out")
            os.makedirs(acq)
            os.makedirs(out)
            with open(os.path.join(acq, "acquisition parameters.json"), "w") as f:
                json.dump({
                    "sensor_pixel_size_um": 6.5,
                    "objective": {"name": "10x", "magnification": 10},
                    "dz(um)": 1.0, "dt(s)": 1.0,
                }, f)
            for t in range(3):
                tp = os.path.join(acq, str(t))
                os.makedirs(tp)
                tifffile.imwrite(
                    os.path.join(tp, "manual_0_0_Fluorescence_488_nm_Ex.tiff"),
                    np.full((4, 4), t, dtype=np.uint16),
                )
            converter_main(acq, out, mode="imagej")
            outputs = [n for n in os.listdir(out) if n.endswith(".tif")]
            assert len(outputs) == 1, outputs
            arr = tifffile.imread(os.path.join(out, outputs[0]))
            # 3 timepoints x 1 z x 1 ch x 4 x 4 -> singletons collapse
            assert arr.shape == (3, 4, 4), arr.shape
            for i in range(3):
                assert int(arr[i, 0, 0]) == i, (i, arr[i, 0, 0])
        finally:
            shutil.rmtree(work, ignore_errors=True)

    tests = [
        ("PyQt6 imports", t_pyqt6),
        ("PyQt6 QApplication starts (platform plugin found)", t_pyqt6_app_instance),
        ("tifffile read/write round-trip", t_tifffile_roundtrip),
        ("import ome_tiff_converter (all public symbols)", t_import_converter),
        ("AcquisitionConverter._parse_stem fallback", t_parse_stem_fallback),
        ("end-to-end conversion of synthetic acquisition", t_end_to_end_convert),
    ]
    for name, fn in tests:
        results.append(_run_one(name, fn))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} smoke tests passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    run()
