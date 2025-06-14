#!/usr/bin/env python3
"""
Convert microscopy acquisition folder to OME-TIFF format.
One OME-TIFF per FOV containing all Z, channels, and timepoints.
"""

import os
import json
import colorsys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import tifffile
from ome_types import OME
from ome_types.model import Image, Pixels, Channel, Plane, TiffData
from ome_types.model.simple_types import Color, PixelType, UnitsLength
import uuid


def load_acquisition_parameters(folder: Path) -> Tuple[float, float]:
    """Load pixel size and dz from acquisition parameters."""
    with open(folder / "acquisition parameters.json", "r") as f:
        params = json.load(f)
    
    pixel_size = params["sensor_pixel_size_um"] / params["objective"]["magnification"]
    dz = params["dz(um)"]
    
    return pixel_size, dz


def get_unique_fovs(acquisition_path: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """Find all unique FOVs across all timepoints."""
    fov_map = {}  # fov_id -> [(timepoint, file_path), ...]
    
    # Get all timepoint directories
    timepoint_dirs = sorted([
        d for d in acquisition_path.iterdir() 
        if d.is_dir() and d.name.isdigit()
    ])
    
    for tp_dir in timepoint_dirs:
        for tiff_file in tp_dir.glob("*.tiff"):
            parts = tiff_file.stem.split("_")
            if len(parts) >= 6:
                well = parts[0]
                fov = parts[1]
                fov_id = f"{well}_{fov}"
                
                if fov_id not in fov_map:
                    fov_map[fov_id] = []
                
                fov_map[fov_id].append((tp_dir.name, tiff_file))
    
    return fov_map


def organize_fov_files(files: List[Tuple[str, Path]]) -> Dict:
    """Organize files by timepoint, z-level, and channel."""
    organized = {}
    
    for timepoint, filepath in files:
        parts = filepath.stem.split("_")
        z_level = int(parts[2])
        wavelength = parts[4]
        
        if timepoint not in organized:
            organized[timepoint] = {}
        if z_level not in organized[timepoint]:
            organized[timepoint][z_level] = {}
        
        organized[timepoint][z_level][wavelength] = filepath
    
    return organized


def get_channels(organized_files: Dict) -> List[str]:
    """Extract unique channels from organized files."""
    channels = set()
    for tp_data in organized_files.values():
        for z_data in tp_data.values():
            channels.update(z_data.keys())
    return sorted(list(channels))


def generate_channel_color(idx: int, total: int) -> int:
    """Generate a color for channel based on position in spectrum."""
    hue = idx / total
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
    return (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)


def get_pixel_type_from_dtype(dtype: np.dtype) -> PixelType:
    """Map numpy dtype to OME PixelType."""
    dtype_map = {
        np.dtype('uint8'): PixelType.UINT8,
        np.dtype('uint16'): PixelType.UINT16,
        np.dtype('uint32'): PixelType.UINT32,
        np.dtype('int8'): PixelType.INT8,
        np.dtype('int16'): PixelType.INT16,
        np.dtype('int32'): PixelType.INT32,
        np.dtype('float32'): PixelType.FLOAT,
        np.dtype('float64'): PixelType.DOUBLE,
    }
    
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported data type: {dtype}")
    
    return dtype_map[dtype]


def convert_fov(fov_id: str, files: List[Tuple[str, Path]], output_dir: Path, pixel_size: float, dz: float):
    """Convert one FOV to OME-TIFF with all timepoints."""
    print(f"\nProcessing FOV {fov_id}...")
    
    # Organize files
    organized = organize_fov_files(files)
    channels = get_channels(organized)
    timepoints = sorted(organized.keys())
    
    # Get dimensions from first file
    first_tp = timepoints[0]
    first_z = sorted(organized[first_tp].keys())[0]
    first_ch = channels[0]
    first_file = organized[first_tp][first_z][first_ch]
    
    first_img = tifffile.imread(first_file)
    img_shape = first_img.shape
    img_dtype = first_img.dtype
    
    # Determine dimensions
    n_t = len(timepoints)
    n_z = max(len(tp_data) for tp_data in organized.values())
    n_c = len(channels)
    
    print(f"  Dimensions: X={img_shape[1]}, Y={img_shape[0]}, Z={n_z}, C={n_c}, T={n_t}")
    
    # Create 5D array (TZCYX)
    data_array = np.zeros((n_t, n_z, n_c, img_shape[0], img_shape[1]), dtype=img_dtype)
    
    # Load all images
    for t_idx, tp in enumerate(timepoints):
        for z_idx in range(n_z):
            for c_idx, ch in enumerate(channels):
                if z_idx in organized[tp] and ch in organized[tp][z_idx]:
                    img = tifffile.imread(organized[tp][z_idx][ch])
                    data_array[t_idx, z_idx, c_idx] = img
    
    # Create OME metadata
    ome = OME(uuid="urn:uuid:" + str(uuid.uuid4()))
    
    pixels = Pixels(
        id="Pixels:0",
        dimension_order="XYZCT",
        size_x=img_shape[1],
        size_y=img_shape[0],
        size_z=n_z,
        size_c=n_c,
        size_t=n_t,
        type=get_pixel_type_from_dtype(img_dtype),
        physical_size_x=pixel_size,
        physical_size_x_unit=UnitsLength.MICROMETER,
        physical_size_y=pixel_size,
        physical_size_y_unit=UnitsLength.MICROMETER,
        physical_size_z=dz,
        physical_size_z_unit=UnitsLength.MICROMETER
    )
    
    # Add channels
    for c_idx, ch in enumerate(channels):
        channel = Channel(
            id=f"Channel:{c_idx}",
            name=f"Channel_{ch}nm",
            color=Color(generate_channel_color(c_idx, n_c)),
            samples_per_pixel=1
        )
        pixels.channels.append(channel)
    
    # Add planes
    for t in range(n_t):
        for z in range(n_z):
            for c in range(n_c):
                plane = Plane(
                    the_z=z,
                    the_c=c,
                    the_t=t
                )
                pixels.planes.append(plane)
    
    # Add TiffData
    for i in range(n_t * n_z * n_c):
        t = i // (n_z * n_c)
        z = (i % (n_z * n_c)) // n_c
        c = i % n_c
        
        tiff_data = TiffData(
            ifd=i,
            plane_count=1,
            first_z=z,
            first_c=c,
            first_t=t
        )
        pixels.tiff_data_blocks.append(tiff_data)
    
    # Create image
    image = Image(
        id="Image:0",
        name=fov_id,
        pixels=pixels
    )
    
    ome.images.append(image)
    
    # Write OME-TIFF
    output_file = output_dir / f"{fov_id}.ome.tiff"
    ome_xml = ome.to_xml()
    
    # Write as OME-TIFF (not ImageJ format)
    tifffile.imwrite(
        output_file,
        data_array,
        description=ome_xml,
        metadata={'axes': 'TZCYX'},
        resolution=(1./pixel_size, 1./pixel_size),
        tile=(256, 256),
        compression='lzw',
        photometric='minisblack',
        ome=True  # Force OME-TIFF format
    )
    
    print(f"  Saved {output_file.name}")


def main(acquisition_folder: str, output_folder: str = None):
    """Convert acquisition folder to OME-TIFFs."""
    acquisition_path = Path(acquisition_folder)
    
    if output_folder is None:
        output_path = acquisition_path / "ome_output"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(exist_ok=True)
    
    # Load parameters
    print("Loading acquisition parameters...")
    pixel_size, dz = load_acquisition_parameters(acquisition_path)
    
    print(f"Pixel size: {pixel_size:.3f} µm")
    print(f"Z step: {dz} µm")
    
    # Find all FOVs across timepoints
    fov_map = get_unique_fovs(acquisition_path)
    print(f"\nFound {len(fov_map)} FOVs across all timepoints")
    
    # Convert each FOV
    for fov_id, files in sorted(fov_map.items()):
        convert_fov(fov_id, files, output_path, pixel_size, dz)
    
    print(f"\nConversion complete! Output saved to: {output_path}")
    print("Each OME-TIFF contains all timepoints, Z-slices, and channels for one FOV")


if __name__ == "__main__":
    import sys
    try:
        from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPalette, QBrush, QColor, QFont
        import threading
    except ImportError:
        print("PyQt6 is required for the GUI. Please install it with 'pip install PyQt6'.")
        sys.exit(1)

    class DropBox(QWidget):
        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setWindowTitle("OME-TIFF Converter")
            self.setFixedSize(400, 200)
            layout = QVBoxLayout()
            self.label = QLabel("Drop your acquisition folder here")
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setFont(QFont("Arial", 14))
            self.label.setStyleSheet("border: 2px dashed #888; padding: 40px; color: #444;")
            layout.addWidget(self.label)
            self.setLayout(layout)

        def dragEnterEvent(self, event: QDragEnterEvent):
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if len(urls) == 1 and urls[0].isLocalFile():
                    import os
                    if os.path.isdir(urls[0].toLocalFile()):
                        event.acceptProposedAction()
                        self.label.setStyleSheet("border: 2px dashed #0078d7; padding: 40px; color: #0078d7;")
                        return
            event.ignore()

        def dragLeaveEvent(self, event):
            self.label.setStyleSheet("border: 2px dashed #888; padding: 40px; color: #444;")

        def dropEvent(self, event: QDropEvent):
            self.label.setStyleSheet("border: 2px dashed #888; padding: 40px; color: #444;")
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                folder = urls[0].toLocalFile()
                import os
                if os.path.isdir(folder):
                    self.label.setText("Converting...")
                    self.setEnabled(False)
                    def run():
                        try:
                            main(folder)
                            self.label.setText("Done! Output in ome_output folder.")
                        except Exception as e:
                            self.label.setText(f"Error: {str(e)}")
                        self.setEnabled(True)
                    threading.Thread(target=run).start()

    app = QApplication(sys.argv)
    win = DropBox()
    win.show()
    sys.exit(app.exec())
