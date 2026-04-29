@echo off
rem Local Windows build. Run from the repo root.
rem Output: dist\ome_tiff_converter.exe (one-file)
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
python -m PyInstaller installer\ome_tiff_converter.spec --noconfirm --distpath dist --workpath build --clean
echo.
echo Build complete. Run dist\ome_tiff_converter.exe
