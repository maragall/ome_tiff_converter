# ome_tiff_converter
Convert microscopy acquisition folder to OME-TIFF format.
One OME-TIFF per FOV containing all Z, channels, and timepoints.

## Installation
```
pip install -r requirements.txt
```

## Usage

### GUI (drag and drop)
```
python ome_tiff_converter.py
```
Drop an acquisition folder onto the window. A dialog will let you select which channels and FOVs to include before converting.

### CLI
```python
from ome_tiff_converter import main

# Convert everything
main("/path/to/acquisition_folder")

# Convert specific channels and FOVs only
main("/path/to/acquisition_folder", channels=["488", "561"], fovs=["R0_0", "R1_0"])

# ImageJ mode
main("/path/to/acquisition_folder", mode="imagej")
```

## Create Desktop Shortcut
```
python create_desktop_shortcut.py
```
