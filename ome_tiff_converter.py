#!/usr/bin/env python3
"""
Convert microscopy acquisition folder to OME-TIFF format.
Replicating Talley's OMETiffWriter architecture exactly.
"""

import os
import json
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import tifffile
import threading

# Talley's constants
IMAGEJ_AXIS_ORDER = "tzcyxs"


class _NULL:
    """Sentinel for missing metadata (Talley's pattern)."""
    pass


_NULL = _NULL()


class _5DWriterBase:
    """Base class replicating Talley's _5DWriterBase architecture."""
    
    def __init__(self) -> None:
        self.current_sequence: Optional['MDASequence'] = None
        self.position_sizes: List[Dict[str, int]] = []
        self._arrays: Dict[str, np.memmap] = {}
    
    def sequenceStarted(self, seq: 'MDASequence', meta: Any = _NULL) -> None:
        """Initialize sequence (Talley's exact method signature)."""
        self.current_sequence = seq
        self.position_sizes = [seq.sizes.copy()]
    
    def write_frame(self, ary: np.memmap, index: tuple, frame: np.ndarray) -> None:
        """Write a frame to the file (Talley's exact method)."""
        ary[index] = frame
    
    def new_array(self, position_key: str, dtype: np.dtype, sizes: Dict[str, int]) -> np.memmap:
        """Create new array (Talley's abstract method - to be implemented)."""
        raise NotImplementedError


class MDASequence:
    """Minimal MDA sequence to match Talley's interface."""
    
    def __init__(self, sizes: Dict[str, int], pixel_size: float, params: dict, channels: List[str]):
        self.sizes = sizes
        self.pixel_size = pixel_size
        self.params = params
        self.channels = channels
        
        # Mock objects for Talley's interface
        self.time_plan = self._create_time_plan() if params.get('dt(s)') else None
        self.z_plan = self._create_z_plan() if params.get('dz(um)') else None
    
    def _create_time_plan(self):
        """Create time plan object matching Talley's interface."""
        class TimePlan:
            def __init__(self, interval):
                self.interval = interval
        return TimePlan(self.params['dt(s)'])
    
    def _create_z_plan(self):
        """Create Z plan object matching Talley's interface."""
        class ZPlan:
            def __init__(self, step):
                self.step = step
        return ZPlan(self.params['dz(um)'])


class OMETiffWriter(_5DWriterBase):
    """MDA handler that writes to a 5D OME-TIFF file.
    
    Exact replication of Talley's OMETiffWriter class architecture.
    """

    def __init__(self, filename: Path | str) -> None:
        try:
            import tifffile  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tifffile is required to use this handler. "
                "Please `pip install tifffile`."
            ) from e

        self._filename = str(filename)
        if not self._filename.endswith((".tiff", ".tif")):
            raise ValueError("filename must end with '.tiff' or '.tif'")
        self._is_ome = ".ome.tif" in self._filename

        super().__init__()

    def sequenceStarted(self, seq: 'MDASequence', meta: Any = _NULL) -> None:
        """Initialize sequence (Talley's exact implementation pattern)."""
        super().sequenceStarted(seq, meta)
        # Non-OME (ImageJ) hyperstack axes MUST be in TZCYXS order
        # so we reorder the ordered position_sizes dicts.
        if not self._is_ome:
            self.position_sizes = [
                {k: x[k] for k in IMAGEJ_AXIS_ORDER if k.lower() in x}
                for x in self.position_sizes
            ]

    def write_frame(self, ary: np.memmap, index: tuple, frame: np.ndarray) -> None:
        """Write a frame to the file (Talley's exact method)."""
        super().write_frame(ary, index, frame)
        ary.flush()

    def new_array(self, position_key: str, dtype: np.dtype, sizes: Dict[str, int]) -> np.memmap:
        """Create a new tifffile file and memmap for this position (Talley's exact method)."""
        dims, shape = zip(*sizes.items())

        metadata: Dict[str, Any] = self._sequence_metadata()
        metadata["axes"] = "".join(dims).upper()

        # append the position key to the filename if there are multiple positions
        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            ext = ".ome.tif" if self._is_ome else ".tif"
            fname = self._filename.replace(ext, f"_{position_key}{ext}")
        else:
            fname = self._filename

        # write empty file to disk (Talley's exact approach)
        tifffile.imwrite(
            fname,
            shape=shape,
            dtype=dtype,
            metadata=metadata,
            imagej=not self._is_ome,
            ome=self._is_ome,
        )

        # memory-mapped NumPy array of image data stored in TIFF file.
        mmap = tifffile.memmap(fname, dtype=dtype)
        # This line is important, as tifffile.memmap appears to lose singleton dims
        mmap.shape = shape

        return mmap

    def _sequence_metadata(self) -> Dict[str, Any]:
        """Create metadata for the sequence (Talley's exact method)."""
        if not self._is_ome:
            return {}

        metadata: Dict[str, Any] = {}
        # see tifffile.tifffile for more metadata options
        if seq := self.current_sequence:
            if seq.time_plan and hasattr(seq.time_plan, "interval"):
                interval = seq.time_plan.interval
                if isinstance(interval, timedelta):
                    interval = interval.total_seconds()
                metadata["TimeIncrement"] = interval
                metadata["TimeIncrementUnit"] = "s"
            if seq.z_plan and hasattr(seq.z_plan, "step"):
                metadata["PhysicalSizeZ"] = seq.z_plan.step
                metadata["PhysicalSizeZUnit"] = "µm"
            if seq.channels:
                metadata["Channel"] = {"Name": [f"Channel_{c}nm" for c in seq.channels]}
            
            # Add physical pixel sizes
            metadata["PhysicalSizeX"] = seq.pixel_size
            metadata["PhysicalSizeY"] = seq.pixel_size
            metadata["PhysicalSizeXUnit"] = "µm"
            metadata["PhysicalSizeYUnit"] = "µm"

        return metadata


class AcquisitionConverter:
    """Main converter class that orchestrates the conversion process."""
    
    def __init__(self, acquisition_folder: Path, output_folder: Optional[Path] = None):
        self.acquisition_folder = Path(acquisition_folder)
        self.output_folder = output_folder
        self.pixel_size: float = 0.0
        self.dz: float = 0.0
        self.params: Dict[str, Any] = {}
    
    def load_acquisition_parameters(self) -> None:
        """Load acquisition parameters from JSON file."""
        with open(self.acquisition_folder / "acquisition parameters.json", "r") as f:
            self.params = json.load(f)
        
        self.pixel_size = self.params["sensor_pixel_size_um"] / self.params["objective"]["magnification"]
        self.dz = self.params["dz(um)"]
    
    def get_unique_fovs(self) -> Dict[str, List[Tuple[str, Path]]]:
        """Find all unique FOVs across all timepoints."""
        fov_map = {}
        
        timepoint_dirs = sorted([
            d for d in self.acquisition_folder.iterdir() 
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
    
    def organize_fov_files(self, files: List[Tuple[str, Path]]) -> Dict[str, Dict[int, Dict[str, Path]]]:
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
    
    def get_channels(self, organized_files: Dict) -> List[str]:
        """Extract unique channels from organized files."""
        channels = set()
        for tp_data in organized_files.values():
            for z_data in tp_data.values():
                channels.update(z_data.keys())
        return sorted(list(channels))
    
    def get_dimension_sizes(self, organized_files: Dict, channels: List[str]) -> Dict[str, int]:
        """Calculate dimension sizes from organized files."""
        timepoints = sorted(organized_files.keys())
        n_t = len(timepoints)
        n_z = max(len(tp_data) for tp_data in organized_files.values()) if organized_files else 1
        n_c = len(channels)
        
        # Get Y, X from first image
        if timepoints and organized_files:
            first_tp = timepoints[0]
            first_z = sorted(organized_files[first_tp].keys())[0]
            first_ch = channels[0]
            first_file = organized_files[first_tp][first_z][first_ch]
            first_img = tifffile.imread(first_file)
            n_y, n_x = first_img.shape
        else:
            n_y, n_x = 512, 512
        
        return {
            't': n_t,
            'z': n_z, 
            'c': n_c,
            'y': n_y,
            'x': n_x
        }
    
    def convert_fov(self, fov_id: str, files: List[Tuple[str, Path]], 
                   output_dir: Path, is_ome: bool = True) -> None:
        """Convert one FOV using Talley's exact workflow."""
        print(f"\nProcessing FOV {fov_id} ({'OME' if is_ome else 'ImageJ'} mode)...")
        
        # Organize files and get metadata
        organized_files = self.organize_fov_files(files)
        channels = self.get_channels(organized_files)
        sizes = self.get_dimension_sizes(organized_files, channels)
        
        # Create MDA sequence (matching Talley's interface)
        sequence = MDASequence(sizes, self.pixel_size, self.params, channels)
        
        # Create writer
        ext = ".ome.tif" if is_ome else ".tif"
        output_file = output_dir / f"{fov_id}{ext}"
        writer = OMETiffWriter(output_file)
        
        # Start sequence (Talley's workflow)
        writer.sequenceStarted(sequence)
        
        # Get position sizes (after potential reordering)
        position_sizes = writer.position_sizes[0]
        
        # Get dtype from first image
        timepoints = sorted(organized_files.keys())
        first_tp = timepoints[0]
        first_z = sorted(organized_files[first_tp].keys())[0]
        first_ch = channels[0]
        first_file = organized_files[first_tp][first_z][first_ch]
        first_img = tifffile.imread(first_file)
        dtype = first_img.dtype
        
        print(f"  Dimension order: {list(position_sizes.keys())}")
        print(f"  Shape: {list(position_sizes.values())}")
        
        # Create memory-mapped array (Talley's approach)
        mmap = writer.new_array("0", dtype, position_sizes)
        
        # Build index mapping based on position_sizes order
        dim_indices = {dim.lower(): i for i, dim in enumerate(position_sizes.keys())}
        
        def get_index_tuple(t_idx: int, z_idx: int, c_idx: int) -> tuple:
            """Build index tuple in the order of position_sizes."""
            index_values = {'t': t_idx, 'z': z_idx, 'c': c_idx}
            idx = []
            for dim in position_sizes.keys():
                if dim.lower() in index_values:
                    idx.append(index_values[dim.lower()])
            return tuple(idx)
        
        # Write frames (Talley's write_frame approach)
        for t_idx, tp in enumerate(timepoints):
            for z_idx in range(sizes['z']):
                for c_idx, ch in enumerate(channels):
                    if z_idx in organized_files[tp] and ch in organized_files[tp][z_idx]:
                        img = tifffile.imread(organized_files[tp][z_idx][ch])
                        index = get_index_tuple(t_idx, z_idx, c_idx)
                        writer.write_frame(mmap, index, img)
        
        # Cleanup
        del mmap
        print(f"  Saved {output_file.name}")
    
    def convert_all(self, mode: str = "ome") -> None:
        """Convert all FOVs using Talley's architecture."""
        # Setup output directory
        if self.output_folder is None:
            suffix = "ome_output" if mode == "ome" else "imagej_output"
            output_path = self.acquisition_folder / suffix
        else:
            output_path = Path(self.output_folder)
        
        output_path.mkdir(exist_ok=True)
        is_ome = mode.lower() == "ome"
        
        # Load parameters
        print(f"Loading acquisition parameters for {mode.upper()} mode...")
        self.load_acquisition_parameters()
        
        print(f"Pixel size: {self.pixel_size:.3f} µm")
        print(f"Z step: {self.dz} µm")
        print(f"Time interval: {self.params.get('dt(s)', 0)} s")
        print(f"Objective: {self.params['objective']['name']} ({self.params['objective']['magnification']}x)")
        print(f"Output format: {'OME-TIFF' if is_ome else 'ImageJ TIFF'}")
        
        # Find and convert all FOVs
        fov_map = self.get_unique_fovs()
        print(f"\nFound {len(fov_map)} FOVs across all timepoints")
        
        for fov_id, files in sorted(fov_map.items()):
            self.convert_fov(fov_id, files, output_path, is_ome)
        
        print(f"\nConversion complete! Output saved to: {output_path}")
        mode_desc = "OME-TIFF with full scientific metadata" if is_ome else "ImageJ-compatible TIFF"
        print(f"Files created in {mode_desc} format")


def main(acquisition_folder: str, output_folder: str = None, mode: str = "ome"):
    """Main entry point matching Talley's style."""
    converter = AcquisitionConverter(acquisition_folder, output_folder)
    converter.convert_all(mode)


if __name__ == "__main__":
    import sys
    try:
        from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
        import threading
    except ImportError:
        print("PyQt6 is required for the GUI. Please install it with 'pip install PyQt6'.")
        sys.exit(1)

    class DropBox(QWidget):
        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setWindowTitle("OME-TIFF Converter (Talley's Architecture)")
            self.setFixedSize(450, 250)
            
            layout = QVBoxLayout()
            
            # Mode selection
            mode_layout = QHBoxLayout()
            self.mode_group = QButtonGroup()
            
            self.ome_radio = QRadioButton("OME-TIFF (Scientific)")
            self.imagej_radio = QRadioButton("ImageJ TIFF (Compatible)")
            self.ome_radio.setChecked(True)
            
            self.mode_group.addButton(self.ome_radio, 0)
            self.mode_group.addButton(self.imagej_radio, 1)
            
            mode_layout.addWidget(self.ome_radio)
            mode_layout.addWidget(self.imagej_radio)
            layout.addLayout(mode_layout)
            
            # Drop area
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
                if os.path.isdir(folder):
                    mode = "ome" if self.ome_radio.isChecked() else "imagej"
                    self.label.setText(f"Converting in {mode.upper()} mode...")
                    self.setEnabled(False)
                    def run():
                        try:
                            main(folder, mode=mode)
                            self.label.setText(f"Done! Output in {mode}_output folder.")
                        except Exception as e:
                            self.label.setText(f"Error: {str(e)}")
                        self.setEnabled(True)
                    threading.Thread(target=run).start()

    app = QApplication(sys.argv)
    win = DropBox()
    win.show()
    sys.exit(app.exec())
