#!/usr/bin/env python3
"""
Create desktop shortcut for OME-TIFF converter
"""

import platform
import subprocess
import sys
from pathlib import Path


def create_windows_shortcut():
    """Create Windows desktop shortcut."""
    desktop = Path.home() / "Desktop"
    script_path = Path(__file__).parent / "ome_tiff_converter.py"
    shortcut_path = desktop / "OME-TIFF Converter.lnk"
    
    # Create PowerShell command to create shortcut
    ps_script = f'''
    $WScriptShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WScriptShell.CreateShortcut("{shortcut_path}")
    $Shortcut.TargetPath = "python"
    $Shortcut.Arguments = '"{script_path}"'
    $Shortcut.WorkingDirectory = "{script_path.parent}"
    $Shortcut.IconLocation = "python.exe"
    $Shortcut.Description = "Convert microscopy acquisitions to OME-TIFF"
    $Shortcut.Save()
    '''
    
    subprocess.run(["powershell", "-Command", ps_script], check=True)
    print(f"Created Windows shortcut: {shortcut_path}")


def create_macos_shortcut():
    """Create macOS desktop alias/shortcut."""
    desktop = Path.home() / "Desktop"
    script_path = Path(__file__).parent / "ome_tiff_converter.py"
    
    # Create a command file that can be double-clicked
    command_file = desktop / "OME-TIFF Converter.command"
    
    content = f'''#!/bin/bash
cd "{script_path.parent}"
echo "OME-TIFF Converter"
echo "=================="
echo ""
echo "Drag and drop your acquisition folder here and press Enter:"
read -r acquisition_folder

# Remove quotes if present
acquisition_folder="${{acquisition_folder%\\"}}"\
acquisition_folder="${{acquisition_folder#\\"}}"\

if [ -d "$acquisition_folder" ]; then
    python3 "{script_path}" "$acquisition_folder"
    echo ""
    echo "Conversion complete! Press Enter to exit."
    read
else
    echo "Error: Not a valid directory"
    echo "Press Enter to exit."
    read
fi
'''
    
    command_file.write_text(content)
    command_file.chmod(0o755)
    
    print(f"Created macOS shortcut: {command_file}")
    print("Double-click it to run the converter with drag-and-drop support")


def create_linux_shortcut():
    """Create Linux desktop entry."""
    desktop = Path.home() / "Desktop"
    desktop.mkdir(exist_ok=True)
    
    script_path = Path(__file__).parent / "ome_tiff_converter.py"
    desktop_file = desktop / "ome-tiff-converter.desktop"
    
    content = f'''[Desktop Entry]
Type=Application
Name=OME-TIFF Converter
Comment=Convert microscopy acquisitions to OME-TIFF format
Exec=gnome-terminal -- bash -c "echo 'OME-TIFF Converter'; echo '=================='; echo ''; echo 'Enter acquisition folder path:'; read folder; python3 '{script_path}' \\"\\$folder\\"; echo ''; echo 'Press Enter to exit'; read"
Icon=applications-science
Terminal=false
Categories=Science;Graphics;
'''
    
    desktop_file.write_text(content)
    desktop_file.chmod(0o755)
    
    print(f"Created Linux shortcut: {desktop_file}")


def main():
    """Create appropriate shortcut for the current OS."""
    system = platform.system()
    
    if system == "Windows":
        create_windows_shortcut()
    elif system == "Darwin":  # macOS
        create_macos_shortcut()
    elif system == "Linux":
        create_linux_shortcut()
    else:
        print(f"Unsupported operating system: {system}")
        sys.exit(1)
    
    print("\nShortcut created successfully!")
    print("You can now run the OME-TIFF converter from your desktop.")


if __name__ == "__main__":
    main()
