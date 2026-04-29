"""Frozen entry point for the PyInstaller-built ome_tiff_converter.exe.

Two responsibilities:

1. When running from a PyInstaller bundle (``sys.frozen`` is set), point Qt
   at the plugins shipped inside the bundle. Without this, PyQt6 cannot find
   the platform plugin (``windowsvista``/``windows``) and the app fails to
   launch with "could not find or load the Qt platform plugin".

2. Route ``--smoke-test`` to the post-build smoke tests (used by CI to verify
   the bundled binary actually loads its dependencies); otherwise launch the
   GUI via :func:`ome_tiff_converter.gui_main`.

A crash log is written next to the .exe so the user has something to send
back if the GUI dies before it can show an error dialog.
"""
import os
import sys
import traceback

if getattr(sys, "frozen", False):
    os.environ["QT_PLUGIN_PATH"] = os.path.join(
        sys._MEIPASS, "PyQt6", "Qt6", "plugins"
    )
    _log_path = os.path.join(os.path.dirname(sys.executable), "crash.log")
else:
    _log_path = None

if "--smoke-test" in sys.argv:
    from installer.smoke_test import run
    run()
else:
    try:
        from ome_tiff_converter import gui_main
        gui_main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if _log_path:
            try:
                with open(_log_path, "w") as f:
                    f.write(tb)
                print(f"\nCrash log written to: {_log_path}", file=sys.stderr)
            except OSError:
                pass
        raise
