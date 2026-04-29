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

def _diagnose():
    """Print proof that this process is using the BUNDLED Python, not the
    host machine's Python install.

    Run with: ome_tiff_converter.exe --diagnose

    Output goes to three places (whichever is visible on your setup):
      1. Parent cmd window (when launched from cmd.exe; we attach to it)
      2. ome_tiff_converter_diagnose.txt next to the .exe (always)
      3. A Windows MessageBox (so double-click users see where the file is)
    """
    # On Windows, a console=False (windowed) PyInstaller exe has no stdout
    # attached when launched from cmd.exe. AttachConsole hooks us up to the
    # parent console so print() works. Silent no-op on non-Windows / no parent.
    if sys.platform == "win32":
        try:
            import ctypes
            ATTACH_PARENT_PROCESS = -1
            ctypes.windll.kernel32.AttachConsole(ATTACH_PARENT_PROCESS)
        except Exception:
            pass

    frozen = getattr(sys, "frozen", False)
    meipass = getattr(sys, "_MEIPASS", None)
    lines = [
        "== ome_tiff_converter.exe self-contained-Python diagnostic ==",
        f"sys.frozen      : {frozen}",
        f"sys.executable  : {sys.executable}",
        f"sys._MEIPASS    : {meipass}",
        f"sys.prefix      : {sys.prefix}",
        f"sys.exec_prefix : {sys.exec_prefix}",
        f"sys.version     : {sys.version.splitlines()[0]}",
        "sys.path[:5]    :",
    ]
    for p in sys.path[:5]:
        lines.append(f"    {p}")
    lines.append("")
    if frozen and meipass:
        inside = sys.prefix.startswith(meipass) or sys.prefix == meipass
        if inside:
            lines.append("VERIFIED: the Python prefix is inside the bundle's temp dir,")
            lines.append("          so the host machine's Python install is NOT in use.")
        else:
            lines.append("WARNING: sys.prefix is NOT inside sys._MEIPASS.")
            lines.append("         The bundle may be picking up host Python; send this output.")
    else:
        lines.append("Not running from a PyInstaller bundle (sys.frozen=False).")
        lines.append("This diagnostic is only meaningful from the built .exe.")
    text = "\n".join(lines)

    # Always print (visible in CI logs and in any console we manage to attach).
    print(text)

    # Always write to a file next to the .exe so double-click users have a
    # tangible artifact even if no console output is visible.
    out_path = None
    if frozen:
        try:
            out_path = os.path.join(
                os.path.dirname(sys.executable), "ome_tiff_converter_diagnose.txt"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            out_path = None

    # On Windows, also pop a MessageBox so a double-click user sees something.
    if sys.platform == "win32":
        try:
            import ctypes
            msg = text + (f"\n\nDiagnostic file: {out_path}" if out_path else "")
            ctypes.windll.user32.MessageBoxW(
                0, msg, "OME-TIFF Converter -- Diagnostic", 0x40
            )
        except Exception:
            pass


if "--diagnose" in sys.argv:
    _diagnose()
elif "--smoke-test" in sys.argv:
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
