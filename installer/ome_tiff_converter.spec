# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ome_tiff_converter.

Build from the project root with:
    python -m PyInstaller installer/ome_tiff_converter.spec --noconfirm

Or from the installer/ directory:
    cd installer && python -m PyInstaller ome_tiff_converter.spec --noconfirm
"""
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# tifffile lazy-loads several optional codec modules. Pull every submodule so
# nothing breaks at runtime when the user opens a slightly less common TIFF.
tifffile_imports = collect_submodules("tifffile")

# imagecodecs is tifffile's compressed-codec backend. It's optional, but if
# the build environment installed it we want it bundled. Tolerate absence.
try:
    imagecodecs_imports = collect_submodules("imagecodecs")
except Exception:
    imagecodecs_imports = []

hidden = (
    tifffile_imports
    + imagecodecs_imports
    + [
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.sip",
        "numpy",
        "numpy.core._methods",
        "numpy.lib.format",
        "xml.etree.ElementTree",
        "ome_tiff_converter",
        "installer.smoke_test",
    ]
)

a = Analysis(
    ["entry.py"],
    pathex=[os.path.abspath("..")],
    binaries=[],
    datas=[],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ome_tiff_converter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # disabled: UPX-packed exes get false-flagged by some AV vendors
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ome_tiff_converter",
)
