#!/usr/bin/env python3
"""
create_desktop_shortcut.py – generate Desktop launchers for NDViewer
on Ubuntu (XDG-compliant) and Windows that activate the conda environment.

Run from the project root:
    python3 create_desktop_shortcut.py
"""

from __future__ import annotations
import platform
import subprocess
import sys
from pathlib import Path
import os
import shutil

# ────────────────────────────── helpers ───────────────────────────────────

def find_conda() -> tuple[Path, str]:
    """Find conda/mamba executable and get conda base path."""
    # Check for mamba first, then conda
    for cmd in ['mamba', 'conda']:
        conda_exe = shutil.which(cmd)
        if conda_exe:
            break
    else:
        sys.exit("[ERROR] Neither conda nor mamba found in PATH")
    
    # Get conda info
    result = subprocess.run([conda_exe, 'info', '--json'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"[ERROR] Failed to get conda info: {result.stderr}")
    
    import json
    conda_info = json.loads(result.stdout)
    conda_base = Path(conda_info['conda_prefix'])
    
    return conda_base, cmd

def check_environment(env_name: str = "ndv") -> bool:
    """Check if the ndv environment exists."""
    result = subprocess.run(['conda', 'env', 'list'], 
                          capture_output=True, text=True)
    return env_name in result.stdout

def find_icon(project_root: Path) -> Path:
    """Return the icon PNG in project root or create a simple one."""
    icon_path = project_root / "ndviewer_icon.png"
    
    # If icon doesn't exist, create a simple one
    if not icon_path.exists():
        print("[INFO] Creating default icon...")
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a 256x256 transparent icon
            size = 256
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Try to load a clean, modern font
            font_size = 160
            font = None
            for font_name in ["Arial-Bold", "DejaVuSans-Bold", "LiberationSans-Bold", "arial.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, font_size)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
            
            text = "ND"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (size - text_width) // 2
            y = (size - text_height) // 2
            
            # Draw simple black outline
            outline_width = 2
            for offset_x in range(-outline_width, outline_width + 1):
                for offset_y in range(-outline_width, outline_width + 1):
                    draw.text((x + offset_x, y + offset_y), text, fill=(0, 0, 0, 255), font=font)
            
            # Draw white text
            draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
            
            img.save(icon_path, optimize=True)
            print(f"[OK] Created simple icon at {icon_path}")
            
        except ImportError:
            print("[WARNING] PIL not available, icon creation skipped")
            
    return icon_path

# Windows --------------------------------------------------------------

def windows_shortcuts(project_root: Path, icon_png: Path, desktop: Path) -> None:
    """Create .bat launchers and .lnk shortcuts on Windows."""
    
    # Convert PNG to ICO if possible
    ico_path = None
    if icon_png.exists():
        try:
            from PIL import Image
            ico_path = project_root / "ome_tiff_converter_icon.ico"
            img = Image.open(icon_png)
            img.save(ico_path, format='ICO', sizes=[(256, 256)])
        except ImportError:
            print("[WARNING] PIL not available, using PNG icon")
    
    # Create batch file for ome_tiff_converter
    script_name = "ome_tiff_converter.py"
    shortcut_name = "OME-TIFF Converter"
    
    if not (project_root / script_name).exists():
        print(f"[WARNING] {script_name} not found, skipping...")
        return
            
    # Create batch file without conda activation
    bat_path = project_root / f"start_{script_name.replace('.py', '')}.bat"
    bat_content = (
        "@echo off\r\n"
        f"cd /d \"{project_root}\"\r\n"
        f"python {script_name}\r\n"
        "pause\r\n"
    )
    bat_path.write_text(bat_content, encoding="utf-8")
    
    # Create shortcut
    lnk_path = desktop / f"{shortcut_name}.lnk"
    
    ps_cmd = (
        "$WshShell = New-Object -ComObject WScript.Shell; "
        f"$Shortcut = $WshShell.CreateShortcut('{lnk_path}'); "
        f"$Shortcut.TargetPath = '{bat_path}'; "
        f"$Shortcut.IconLocation = '{ico_path or icon_png}'; "
        f"$Shortcut.WorkingDirectory = '{project_root}'; "
        "$Shortcut.Save()"
    )
    
    subprocess.run([
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", ps_cmd,
    ], check=True)
    
    print(f"[OK] Windows shortcut created → {lnk_path}")

# Ubuntu / Linux -------------------------------------------------------

def ubuntu_shortcuts(project_root: Path, icon_png: Path, desktop: Path) -> None:
    """Create XDG .desktop launchers on Ubuntu/Linux."""
    
    # Create wrapper script and desktop file for ome_tiff_converter
    script_name = "ome_tiff_converter.py"
    app_name = "OME-TIFF Converter"
    comment = "Convert microscopy acquisition folder to OME-TIFF format."
    
    if not (project_root / script_name).exists():
        print(f"[WARNING] {script_name} not found, skipping...")
        return
            
    # Create wrapper script without conda activation
    wrapper_path = project_root / f"launch_{script_name.replace('.py', '')}.sh"
    wrapper_content = f"""#!/bin/bash
cd \"{project_root}\"
python {script_name}
"""
    wrapper_path.write_text(wrapper_content)
    wrapper_path.chmod(0o755)
    
    # Create desktop file
    desktop_file = desktop / f"{script_name.replace('.py', '')}.desktop"
    
    desktop_content = "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={app_name}",
        f"Comment={comment}",
        f"Path={project_root}",
        f"Exec={wrapper_path}",
        f"Icon={icon_png}",
        "Terminal=false",
        "Categories=Graphics;Science;Converter;",
        ""
    ])
    
    desktop_file.write_text(desktop_content, encoding="utf-8")
    desktop_file.chmod(0o755)
    
    print(f"[OK] Ubuntu shortcut created → {desktop_file}")

# ─────────────────────────────── main ───────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parent
    
    # Remove conda detection and environment check
    icon_png = find_icon(project_root)
    
    desktop = Path.home() / "Desktop"
    desktop.mkdir(exist_ok=True)
    
    os_name = platform.system().lower()
    
    print(f"\nCreating desktop shortcuts for OME-TIFF Converter...")
    print(f"Project root: {project_root}")
    print(f"Desktop: {desktop}")
    
    if os_name == "windows":
        windows_shortcuts(project_root, icon_png, desktop)
    elif os_name == "linux":
        ubuntu_shortcuts(project_root, icon_png, desktop)
    else:
        sys.exit("[ERROR] Unsupported OS: only Windows and Ubuntu are handled.")
    
    print("\n[SUCCESS] Desktop shortcuts created!")
    print("The shortcut will launch ome_tiff_converter.py directly.")

if __name__ == "__main__":
    main()
