#!/usr/bin/env python3
"""
Convert microscopy acquisition folder to OME-TIFF format.
Based on Talley's OMETiffWriter architecture.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import tifffile

IMAGEJ_AXIS_ORDER = "tzcyxs"


class MDASequence:
    """Acquisition sequence metadata."""

    def __init__(self, sizes: Dict[str, int], pixel_size: float, params: dict, channels: List[str]):
        self.sizes = sizes
        self.pixel_size = pixel_size
        self.channels = channels
        self.time_interval = params.get('dt(s)')
        self.z_step = params.get('dz(um)')


class OMETiffWriter:
    """Writes 5D OME-TIFF or ImageJ TIFF files using memory-mapped arrays."""

    def __init__(self, filename: Path | str) -> None:
        self._filename = str(filename)
        if not self._filename.endswith((".tiff", ".tif")):
            raise ValueError("filename must end with '.tiff' or '.tif'")
        self._is_ome = ".ome.tif" in self._filename
        self.current_sequence: Optional[MDASequence] = None
        self.position_sizes: List[Dict[str, int]] = []

    def sequenceStarted(self, seq: MDASequence) -> None:
        self.current_sequence = seq
        self.position_sizes = [seq.sizes.copy()]
        # ImageJ hyperstack axes must be in TZCYXS order
        if not self._is_ome:
            self.position_sizes = [
                {k: x[k] for k in IMAGEJ_AXIS_ORDER if k.lower() in x}
                for x in self.position_sizes
            ]

    def write_frame(self, ary: np.memmap, index: tuple, frame: np.ndarray) -> None:
        ary[index] = frame
        ary.flush()

    def new_array(self, position_key: str, dtype: np.dtype, sizes: Dict[str, int]) -> np.memmap:
        dims, shape = zip(*sizes.items())

        metadata: Dict[str, Any] = self._sequence_metadata()
        metadata["axes"] = "".join(dims).upper()

        if (seq := self.current_sequence) and seq.sizes.get("p", 1) > 1:
            ext = ".ome.tif" if self._is_ome else ".tif"
            fname = self._filename.replace(ext, f"_{position_key}{ext}")
        else:
            fname = self._filename

        tifffile.imwrite(
            fname,
            shape=shape,
            dtype=dtype,
            metadata=metadata,
            imagej=not self._is_ome,
            ome=self._is_ome,
        )

        mmap = tifffile.memmap(fname, dtype=dtype)
        # tifffile.memmap loses singleton dims
        mmap.shape = shape
        return mmap

    def _sequence_metadata(self) -> Dict[str, Any]:
        if not self._is_ome:
            return {}

        metadata: Dict[str, Any] = {}
        if seq := self.current_sequence:
            if seq.time_interval:
                metadata["TimeIncrement"] = seq.time_interval
                metadata["TimeIncrementUnit"] = "s"
            if seq.z_step:
                metadata["PhysicalSizeZ"] = seq.z_step
                metadata["PhysicalSizeZUnit"] = "µm"
            if seq.channels:
                metadata["Channel"] = {"Name": [f"Channel_{c}nm" for c in seq.channels]}
            metadata["PhysicalSizeX"] = seq.pixel_size
            metadata["PhysicalSizeY"] = seq.pixel_size
            metadata["PhysicalSizeXUnit"] = "µm"
            metadata["PhysicalSizeYUnit"] = "µm"

        return metadata


class AcquisitionConverter:
    """Orchestrates conversion of acquisition folders to OME-TIFF."""

    def __init__(self, acquisition_folder: Path, output_folder: Optional[Path] = None):
        self.acquisition_folder = Path(acquisition_folder)
        self.output_folder = output_folder
        self.pixel_size: float = 0.0
        self.dz: float = 0.0
        self.params: Dict[str, Any] = {}

    def load_acquisition_parameters(self) -> None:
        with open(self.acquisition_folder / "acquisition parameters.json", "r") as f:
            self.params = json.load(f)
        self.pixel_size = self.params["sensor_pixel_size_um"] / self.params["objective"]["magnification"]
        self.dz = self.params["dz(um)"]

    def get_unique_fovs(self) -> Dict[str, List[Tuple[str, Path]]]:
        fov_map = {}
        timepoint_dirs = sorted([
            d for d in self.acquisition_folder.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        for tp_dir in timepoint_dirs:
            for tiff_file in tp_dir.glob("*.tiff"):
                parts = tiff_file.stem.split("_")
                if len(parts) >= 6:
                    fov_id = f"{parts[0]}_{parts[1]}"
                    if fov_id not in fov_map:
                        fov_map[fov_id] = []
                    fov_map[fov_id].append((tp_dir.name, tiff_file))
        return fov_map

    def get_all_channels(self) -> List[str]:
        channels = set()
        timepoint_dirs = sorted([
            d for d in self.acquisition_folder.iterdir()
            if d.is_dir() and d.name.isdigit()
        ])
        if timepoint_dirs:
            for tiff_file in timepoint_dirs[0].glob("*.tiff"):
                parts = tiff_file.stem.split("_")
                if len(parts) >= 6:
                    channels.add(parts[4])
        return sorted(channels)

    def organize_fov_files(self, files: List[Tuple[str, Path]]) -> Dict[str, Dict[int, Dict[str, Path]]]:
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
        channels = set()
        for tp_data in organized_files.values():
            for z_data in tp_data.values():
                channels.update(z_data.keys())
        return sorted(channels)

    def get_dimension_sizes(self, organized_files: Dict, channels: List[str]) -> Dict[str, int]:
        timepoints = sorted(organized_files.keys())
        n_t = len(timepoints)
        n_z = max(len(tp_data) for tp_data in organized_files.values()) if organized_files else 1
        n_c = len(channels)

        if timepoints and organized_files:
            first_tp = timepoints[0]
            first_z = sorted(organized_files[first_tp].keys())[0]
            first_file = organized_files[first_tp][first_z][channels[0]]
            first_img = tifffile.imread(first_file)
            n_y, n_x = first_img.shape
        else:
            n_y, n_x = 512, 512

        return {'t': n_t, 'z': n_z, 'c': n_c, 'y': n_y, 'x': n_x}

    def convert_fov(self, fov_id: str, files: List[Tuple[str, Path]],
                    output_dir: Path, is_ome: bool = True,
                    channel_filter: Optional[List[str]] = None) -> None:
        print(f"\nProcessing FOV {fov_id} ({'OME' if is_ome else 'ImageJ'} mode)...")

        if channel_filter is not None:
            channel_set = set(channel_filter)
            files = [(tp, f) for tp, f in files if f.stem.split("_")[4] in channel_set]

        organized_files = self.organize_fov_files(files)
        channels = self.get_channels(organized_files)
        sizes = self.get_dimension_sizes(organized_files, channels)

        sequence = MDASequence(sizes, self.pixel_size, self.params, channels)

        ext = ".ome.tif" if is_ome else ".tif"
        output_file = output_dir / f"{fov_id}{ext}"
        writer = OMETiffWriter(output_file)
        writer.sequenceStarted(sequence)

        position_sizes = writer.position_sizes[0]

        timepoints = sorted(organized_files.keys())
        first_tp = timepoints[0]
        first_z = sorted(organized_files[first_tp].keys())[0]
        first_file = organized_files[first_tp][first_z][channels[0]]
        dtype = tifffile.imread(first_file).dtype

        print(f"  Dimension order: {list(position_sizes.keys())}")
        print(f"  Shape: {list(position_sizes.values())}")

        mmap = writer.new_array("0", dtype, position_sizes)

        def get_index_tuple(t_idx: int, z_idx: int, c_idx: int) -> tuple:
            index_values = {'t': t_idx, 'z': z_idx, 'c': c_idx}
            return tuple(index_values[dim.lower()] for dim in position_sizes if dim.lower() in index_values)

        for t_idx, tp in enumerate(timepoints):
            for z_idx in range(sizes['z']):
                for c_idx, ch in enumerate(channels):
                    if z_idx in organized_files[tp] and ch in organized_files[tp][z_idx]:
                        img = tifffile.imread(organized_files[tp][z_idx][ch])
                        writer.write_frame(mmap, get_index_tuple(t_idx, z_idx, c_idx), img)

        del mmap
        print(f"  Saved {output_file.name}")

    def convert_all(self, mode: str = "ome", channels: Optional[List[str]] = None,
                    fovs: Optional[List[str]] = None) -> None:
        if self.output_folder is None:
            suffix = "ome_output" if mode == "ome" else "imagej_output"
            output_path = self.acquisition_folder / suffix
        else:
            output_path = Path(self.output_folder)

        output_path.mkdir(exist_ok=True)
        is_ome = mode.lower() == "ome"

        print(f"Loading acquisition parameters for {mode.upper()} mode...")
        self.load_acquisition_parameters()

        print(f"Pixel size: {self.pixel_size:.3f} µm")
        print(f"Z step: {self.dz} µm")
        print(f"Time interval: {self.params.get('dt(s)', 0)} s")
        print(f"Objective: {self.params['objective']['name']} ({self.params['objective']['magnification']}x)")
        print(f"Output format: {'OME-TIFF' if is_ome else 'ImageJ TIFF'}")

        fov_map = self.get_unique_fovs()

        if fovs is not None:
            fov_set = set(fovs)
            fov_map = {k: v for k, v in fov_map.items() if k in fov_set}

        print(f"\nFound {len(fov_map)} FOVs to convert")

        for fov_id, files in sorted(fov_map.items()):
            self.convert_fov(fov_id, files, output_path, is_ome, channel_filter=channels)

        print(f"\nConversion complete! Output saved to: {output_path}")
        mode_desc = "OME-TIFF with full scientific metadata" if is_ome else "ImageJ-compatible TIFF"
        print(f"Files created in {mode_desc} format")


def main(acquisition_folder: str, output_folder: str = None, mode: str = "ome",
         channels: Optional[List[str]] = None, fovs: Optional[List[str]] = None):
    converter = AcquisitionConverter(acquisition_folder, output_folder)
    converter.convert_all(mode, channels=channels, fovs=fovs)


if __name__ == "__main__":
    import sys
    import threading
    try:
        from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                                      QRadioButton, QButtonGroup, QDialog, QCheckBox,
                                      QPushButton, QListWidget, QListWidgetItem)
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont
    except ImportError:
        print("PyQt6 is required for the GUI. Please install it with 'pip install PyQt6'.")
        sys.exit(1)

    class SelectionDialog(QDialog):
        def __init__(self, channels, fov_ids, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Select Channels & FOVs")
            self.setMinimumSize(400, 500)

            layout = QVBoxLayout()

            layout.addWidget(QLabel("Channels:"))
            self.ch_checks = {}
            for ch in channels:
                cb = QCheckBox(ch)
                cb.setChecked(True)
                self.ch_checks[ch] = cb
                layout.addWidget(cb)

            layout.addWidget(QLabel("FOVs:"))
            btn_row = QHBoxLayout()
            sel_all = QPushButton("Select All")
            desel_all = QPushButton("Deselect All")
            sel_all.clicked.connect(lambda: self._set_all_fovs(True))
            desel_all.clicked.connect(lambda: self._set_all_fovs(False))
            btn_row.addWidget(sel_all)
            btn_row.addWidget(desel_all)
            layout.addLayout(btn_row)

            self.fov_list = QListWidget()
            for fov_id in fov_ids:
                item = QListWidgetItem(fov_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.fov_list.addItem(item)
            layout.addWidget(self.fov_list)

            btn_row2 = QHBoxLayout()
            cancel_btn = QPushButton("Cancel")
            convert_btn = QPushButton("Convert")
            cancel_btn.clicked.connect(self.reject)
            convert_btn.clicked.connect(self.accept)
            btn_row2.addWidget(cancel_btn)
            btn_row2.addWidget(convert_btn)
            layout.addLayout(btn_row2)

            self.setLayout(layout)

        def _set_all_fovs(self, checked):
            state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            for i in range(self.fov_list.count()):
                self.fov_list.item(i).setCheckState(state)

        def get_selections(self):
            channels = [ch for ch, cb in self.ch_checks.items() if cb.isChecked()]
            fovs = []
            for i in range(self.fov_list.count()):
                item = self.fov_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    fovs.append(item.text())
            return channels, fovs

    class DropBox(QWidget):
        def __init__(self):
            super().__init__()
            self.setAcceptDrops(True)
            self.setWindowTitle("OME-TIFF Converter")
            self.setFixedSize(450, 250)

            layout = QVBoxLayout()

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

                    try:
                        converter = AcquisitionConverter(folder)
                        all_channels = converter.get_all_channels()
                        fov_map = converter.get_unique_fovs()
                        all_fovs = sorted(fov_map.keys())
                    except Exception as e:
                        self.label.setText(f"Error scanning folder: {e}")
                        return

                    dialog = SelectionDialog(all_channels, all_fovs, self)
                    if dialog.exec() != QDialog.DialogCode.Accepted:
                        return

                    selected_channels, selected_fovs = dialog.get_selections()
                    if not selected_channels or not selected_fovs:
                        self.label.setText("No channels or FOVs selected.")
                        return

                    self.label.setText(
                        f"Converting {len(selected_fovs)} FOVs, "
                        f"{len(selected_channels)} channels...")
                    self.setEnabled(False)

                    def run():
                        try:
                            main(folder, mode=mode,
                                 channels=selected_channels, fovs=selected_fovs)
                            self.label.setText(
                                f"Done! Output in {mode}_output folder.")
                        except Exception as e:
                            self.label.setText(f"Error: {str(e)}")
                        self.setEnabled(True)
                    threading.Thread(target=run).start()

    app = QApplication(sys.argv)
    win = DropBox()
    win.show()
    sys.exit(app.exec())
