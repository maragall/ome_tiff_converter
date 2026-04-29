#!/usr/bin/env python3
"""
Convert microscopy acquisition folder to OME-TIFF format.
Based on Talley's OMETiffWriter architecture.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
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
            photometric='minisblack',
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
                # Use the channel name as-is, restoring spaces from underscores
                # so OME-XML carries the original Squid config name
                # ("BF LED matrix full" rather than "BF_LED_matrix_full").
                metadata["Channel"] = {"Name": [c.replace("_", " ") for c in seq.channels]}
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
        self._known_channels_safe: Optional[Set[str]] = None

    def load_acquisition_parameters(self) -> None:
        with open(self.acquisition_folder / "acquisition parameters.json", "r") as f:
            self.params = json.load(f)
        self.pixel_size = self.params["sensor_pixel_size_um"] / self.params["objective"]["magnification"]
        self.dz = self.params["dz(um)"]

    def _get_known_channels_safe(self) -> Set[str]:
        """Channel name list from configurations.xml, with spaces replaced by
        underscores to match Squid's channel_name_safe in filenames. Returns
        an empty set if configurations.xml is missing or unreadable; in that
        case _parse_stem falls back to a positional split that assumes
        region_id has no internal underscores.
        """
        if self._known_channels_safe is not None:
            return self._known_channels_safe
        cfg = self.acquisition_folder / "configurations.xml"
        if not cfg.exists():
            self._known_channels_safe = set()
            return self._known_channels_safe
        try:
            root = ET.parse(cfg).getroot()
            self._known_channels_safe = {
                el.get("Name").replace(" ", "_")
                for el in root.iter("mode")
                if el.get("Name")
            }
        except ET.ParseError:
            self._known_channels_safe = set()
        return self._known_channels_safe

    def _parse_stem(self, stem: str) -> Optional[Tuple[str, str, int, str]]:
        """Parse a Squid TIFF stem `{region_id}_{fov}_{z}_{channel_name_safe}`
        into (region_id, fov, z_level, channel_name_safe).

        With configurations.xml available, strip the longest matching channel
        suffix from the end and split the remaining file_id; this handles
        region_ids and channel names that contain underscores.

        Without configurations.xml, fall back to positional split (assumes
        region_id is exactly the first underscore-separated token).
        """
        for ch in sorted(self._get_known_channels_safe(), key=len, reverse=True):
            suffix = "_" + ch
            if stem.endswith(suffix):
                file_id = stem[: -len(suffix)]
                id_parts = file_id.split("_")
                if len(id_parts) < 3:
                    continue
                try:
                    z = int(id_parts[-1])
                except ValueError:
                    continue
                region = "_".join(id_parts[:-2])
                return (region, id_parts[-2], z, ch)
        parts = stem.split("_")
        if len(parts) < 4:
            return None
        try:
            z = int(parts[2])
        except ValueError:
            return None
        return (parts[0], parts[1], z, "_".join(parts[3:]))

    def _channel_name(self, stem: str) -> Optional[str]:
        parsed = self._parse_stem(stem)
        return parsed[3] if parsed is not None else None

    def _timepoint_dirs(self) -> List[Path]:
        return sorted(
            [d for d in self.acquisition_folder.iterdir()
             if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name),
        )

    def detect_format(self) -> str:
        """Classify the dropped folder as one of Squid's save formats.

        Returns one of:
          'native_ome'        - SaveOMETiffJob output (since 2025-09-29);
                                per-FOV {region}_{fov}.ome.tiff in ome_tiff/
                                subfolder; per-timepoint dirs only have
                                coordinates.csv.
          'multi_page_stack'  - SaveImageJob with FILE_SAVING_OPTION =
                                MULTI_PAGE_TIFF; per-FOV {region}_{fov}_stack.tiff
                                inside each timepoint dir, all (z, c) appended
                                as pages with JSON metadata in ImageDescription.
          'single_tiff'       - SaveImageJob default (since 2024); one
                                {region}_{fov}_{z}_{channel_name_safe}.tiff
                                per capture inside each timepoint dir.
          'unknown'           - none of the above; probably zarr or empty.
        """
        ome_dir = self.acquisition_folder / "ome_tiff"
        if ome_dir.is_dir() and any(ome_dir.glob("*.ome.tiff")):
            return "native_ome"
        tp_dirs = self._timepoint_dirs()
        for tp in tp_dirs:
            tiffs = list(tp.glob("*.tiff"))
            if not tiffs:
                continue
            stack_count = sum(1 for t in tiffs if t.stem.endswith("_stack"))
            if stack_count == len(tiffs):
                return "multi_page_stack"
            if stack_count < len(tiffs):
                return "single_tiff"
        return "unknown"

    def get_unique_fovs(self) -> Dict[str, List[Tuple[str, Path]]]:
        fmt = self.detect_format()
        if fmt == "native_ome":
            return self._unique_fovs_native_ome()
        if fmt == "multi_page_stack":
            return self._unique_fovs_stack()
        return self._unique_fovs_single_tiff()

    def _unique_fovs_single_tiff(self) -> Dict[str, List[Tuple[str, Path]]]:
        fov_map: Dict[str, List[Tuple[str, Path]]] = {}
        for tp_dir in self._timepoint_dirs():
            for tiff_file in tp_dir.glob("*.tiff"):
                if tiff_file.stem.endswith("_stack"):
                    continue
                parsed = self._parse_stem(tiff_file.stem)
                if parsed is None:
                    continue
                region, fov, _z, _ch = parsed
                fov_id = f"{region}_{fov}"
                fov_map.setdefault(fov_id, []).append((tp_dir.name, tiff_file))
        return fov_map

    @staticmethod
    def _native_ome_fov_id(p: Path) -> str:
        stem = p.stem
        return stem[:-4] if stem.endswith(".ome") else stem

    def _unique_fovs_native_ome(self) -> Dict[str, List[Tuple[str, Path]]]:
        fov_map: Dict[str, List[Tuple[str, Path]]] = {}
        ome_dir = self.acquisition_folder / "ome_tiff"
        if not ome_dir.is_dir():
            return fov_map
        for f in sorted(ome_dir.glob("*.ome.tiff")):
            # Native OME files are already multi-timepoint internally;
            # we use a single placeholder timepoint slot.
            fov_map[self._native_ome_fov_id(f)] = [("0", f)]
        return fov_map

    def _unique_fovs_stack(self) -> Dict[str, List[Tuple[str, Path]]]:
        fov_map: Dict[str, List[Tuple[str, Path]]] = {}
        for tp_dir in self._timepoint_dirs():
            for stack_file in tp_dir.glob("*_stack.tiff"):
                stem = stack_file.stem
                if stem.endswith("_stack"):
                    fov_id = stem[: -len("_stack")]
                    fov_map.setdefault(fov_id, []).append((tp_dir.name, stack_file))
        return fov_map

    def get_all_channels(self) -> List[str]:
        fmt = self.detect_format()
        if fmt == "native_ome":
            return self._channels_native_ome()
        if fmt == "multi_page_stack":
            return self._channels_stack()
        return self._channels_single_tiff()

    def _channels_single_tiff(self) -> List[str]:
        channels: Set[str] = set()
        tp_dirs = self._timepoint_dirs()
        if tp_dirs:
            for tiff_file in tp_dirs[0].glob("*.tiff"):
                if tiff_file.stem.endswith("_stack"):
                    continue
                ch = self._channel_name(tiff_file.stem)
                if ch is not None:
                    channels.add(ch)
        return sorted(channels)

    def _channels_native_ome(self) -> List[str]:
        """Channel names from the OME-XML of the first .ome.tiff. Names
        are returned in the same '_'-separated form as filename channels
        so the GUI can present a single uniform list across formats."""
        ome_dir = self.acquisition_folder / "ome_tiff"
        if not ome_dir.is_dir():
            return []
        files = sorted(ome_dir.glob("*.ome.tiff"))
        if not files:
            return []
        try:
            with tifffile.TiffFile(files[0]) as tf:
                xml = tf.ome_metadata or ""
        except Exception:
            return []
        import re
        names = re.findall(r'<Channel[^/]*?Name="([^"]+)"', xml)
        return sorted({n.replace(" ", "_") for n in names if n})

    @staticmethod
    def _stack_page_metadata(page) -> Optional[Dict[str, Any]]:
        """Pull the per-page metadata SaveImageJob writes into the
        ImageDescription tag (a JSON dict with z_level, channel,
        channel_index, region_id, fov, ...). Returns None if missing."""
        desc = page.description
        if not desc:
            return None
        try:
            meta = json.loads(desc)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        if isinstance(meta, dict) and "z_level" in meta and "channel" in meta:
            return meta
        return None

    def _channels_stack(self) -> List[str]:
        channels: Set[str] = set()
        for tp_dir in self._timepoint_dirs():
            for stack_file in tp_dir.glob("*_stack.tiff"):
                try:
                    with tifffile.TiffFile(stack_file) as tf:
                        for page in tf.pages:
                            meta = self._stack_page_metadata(page)
                            if meta:
                                channels.add(str(meta["channel"]).replace(" ", "_"))
                except Exception:
                    continue
                if channels:
                    return sorted(channels)
        return sorted(channels)

    def organize_fov_files(self, files: List[Tuple[str, Path]]) -> Dict[str, Dict[int, Dict[str, Path]]]:
        organized = {}
        for timepoint, filepath in files:
            parsed = self._parse_stem(filepath.stem)
            if parsed is None:
                continue
            _region, _fov, z_level, channel = parsed
            organized.setdefault(timepoint, {}).setdefault(z_level, {})[channel] = filepath
        return organized

    def get_channels(self, organized_files: Dict) -> List[str]:
        channels = set()
        for tp_data in organized_files.values():
            for z_data in tp_data.values():
                channels.update(z_data.keys())
        return sorted(channels)

    def get_dimension_sizes(self, organized_files: Dict, channels: List[str]) -> Dict[str, int]:
        timepoints = sorted(organized_files.keys(), key=int)
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
            files = [(tp, f) for tp, f in files if self._channel_name(f.stem) in channel_set]

        organized_files = self.organize_fov_files(files)
        channels = self.get_channels(organized_files)
        if not organized_files or not channels:
            print(f"  Skipping FOV {fov_id}: no files match the current channel/FOV selection.")
            return
        sizes = self.get_dimension_sizes(organized_files, channels)

        sequence = MDASequence(sizes, self.pixel_size, self.params, channels)

        ext = ".ome.tif" if is_ome else ".tif"
        output_file = output_dir / f"{fov_id}{ext}"
        writer = OMETiffWriter(output_file)
        writer.sequenceStarted(sequence)

        position_sizes = writer.position_sizes[0]

        timepoints = sorted(organized_files.keys(), key=int)
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

        # acquisition parameters.json may be missing on some legacy / stitched
        # folders. Best-effort load; fall back to defaults so the conversion
        # path still functions for native-OME passthrough where we don't need
        # pixel size / dz to copy bytes through.
        try:
            self.load_acquisition_parameters()
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Warning: could not load acquisition parameters ({e}); using defaults.")
            self.params = self.params or {}
            self.pixel_size = self.pixel_size or 0.0
            self.dz = self.dz or 0.0

        fmt = self.detect_format()
        print(f"Detected save format: {fmt}")
        if self.pixel_size:
            print(f"Pixel size: {self.pixel_size:.3f} µm")
        if self.dz:
            print(f"Z step: {self.dz} µm")
        if self.params.get("dt(s)") is not None:
            print(f"Time interval: {self.params['dt(s)']} s")
        if self.params.get("objective"):
            obj = self.params["objective"]
            print(f"Objective: {obj.get('name', '?')} ({obj.get('magnification', '?')}x)")
        print(f"Output format: {'OME-TIFF' if is_ome else 'ImageJ TIFF'}")

        if fmt == "native_ome":
            self._convert_all_native_ome(output_path, is_ome, channels, fovs)
        elif fmt == "multi_page_stack":
            self._convert_all_stack(output_path, is_ome, channels, fovs)
        elif fmt == "single_tiff":
            self._convert_all_single_tiff(output_path, is_ome, channels, fovs)
        else:
            raise ValueError(
                f"Could not detect a Squid save format in {self.acquisition_folder}. "
                f"Expected one of: per-tile TIFFs in numbered timepoint dirs, "
                f"_stack.tiff per FOV in timepoint dirs, or .ome.tiff in an "
                f"ome_tiff/ subfolder."
            )

        print(f"\nConversion complete! Output saved to: {output_path}")
        mode_desc = "OME-TIFF with full scientific metadata" if is_ome else "ImageJ-compatible TIFF"
        print(f"Files created in {mode_desc} format")

    def _convert_all_single_tiff(self, output_path: Path, is_ome: bool,
                                 channels: Optional[List[str]],
                                 fovs: Optional[List[str]]) -> None:
        fov_map = self._unique_fovs_single_tiff()
        if fovs is not None:
            fov_map = {k: v for k, v in fov_map.items() if k in set(fovs)}
        print(f"\nFound {len(fov_map)} FOVs to convert")
        for fov_id, files in sorted(fov_map.items()):
            self.convert_fov(fov_id, files, output_path, is_ome, channel_filter=channels)

    def _convert_all_native_ome(self, output_path: Path, is_ome: bool,
                                channels: Optional[List[str]],
                                fovs: Optional[List[str]]) -> None:
        import shutil
        ome_dir = self.acquisition_folder / "ome_tiff"
        files = sorted(ome_dir.glob("*.ome.tiff"))
        if fovs is not None:
            fov_set = set(fovs)
            files = [f for f in files if self._native_ome_fov_id(f) in fov_set]
        print(f"\nFound {len(files)} FOVs to convert")
        for src in files:
            fov_id = self._native_ome_fov_id(src)
            print(f"\nProcessing FOV {fov_id} (native OME -> {'OME' if is_ome else 'ImageJ'} mode)...")
            ext = ".ome.tif" if is_ome else ".tif"
            dest = output_path / f"{fov_id}{ext}"
            if is_ome and channels is None:
                # Already in the target format and no channel filtering needed.
                # Byte-copy is fastest and preserves the OME-XML exactly as
                # Squid wrote it.
                shutil.copy2(src, dest)
                print(f"  Copied {dest.name} (passthrough)")
            else:
                self._transcode_native_ome(src, dest, is_ome, channels)

    def _transcode_native_ome(self, src: Path, dest: Path, is_ome: bool,
                              channel_filter: Optional[List[str]]) -> None:
        """Read a native OME-TIFF, optionally slice the channel axis to a
        requested subset, write back as OME or ImageJ format."""
        with tifffile.TiffFile(src) as tf:
            series = tf.series[0]
            arr = series.asarray()
            axes = series.axes  # e.g., 'TZCYX' or 'ZCYX'
            xml = tf.ome_metadata or ""
        import re
        all_names = re.findall(r'<Channel[^/]*?Name="([^"]+)"', xml)
        all_names_safe = [n.replace(" ", "_") for n in all_names]
        kept_names_safe = all_names_safe
        if channel_filter is not None:
            keep = set(channel_filter)
            indices = [i for i, n in enumerate(all_names_safe) if n in keep]
            if not indices:
                print(f"  Skipping {dest.name}: no channels match the filter "
                      f"(file has {all_names_safe!r}, filter wants {channel_filter!r}).")
                return
            if "C" in axes:
                arr = np.take(arr, indices, axis=axes.index("C"))
            kept_names_safe = [all_names_safe[i] for i in indices]

        metadata: Dict[str, Any] = {"axes": axes}
        if is_ome and kept_names_safe:
            metadata["Channel"] = {"Name": [n.replace("_", " ") for n in kept_names_safe]}
        tifffile.imwrite(
            str(dest),
            arr,
            imagej=not is_ome,
            ome=is_ome,
            photometric="minisblack",
            metadata=metadata,
        )
        print(f"  Transcoded {dest.name} (axes={axes}, shape={arr.shape}, "
              f"channels={kept_names_safe})")

    def _convert_all_stack(self, output_path: Path, is_ome: bool,
                           channels: Optional[List[str]],
                           fovs: Optional[List[str]]) -> None:
        fov_map = self._unique_fovs_stack()
        if fovs is not None:
            fov_map = {k: v for k, v in fov_map.items() if k in set(fovs)}
        print(f"\nFound {len(fov_map)} FOVs to convert")
        for fov_id in sorted(fov_map):
            self._convert_stack_fov(fov_id, fov_map[fov_id], output_path, is_ome, channels)

    def _convert_stack_fov(self, fov_id: str,
                           tp_files: List[Tuple[str, Path]],
                           output_path: Path, is_ome: bool,
                           channel_filter: Optional[List[str]]) -> None:
        print(f"\nProcessing FOV {fov_id} (multi-page stack -> {'OME' if is_ome else 'ImageJ'} mode)...")
        tp_files = sorted(tp_files, key=lambda x: int(x[0]))

        # First pass: discover dimensions from page metadata.
        z_levels: Set[int] = set()
        channels_set: Set[str] = set()
        sample_shape = None
        sample_dtype = None
        for _tp, stack_path in tp_files:
            try:
                with tifffile.TiffFile(stack_path) as tf:
                    for page in tf.pages:
                        meta = self._stack_page_metadata(page)
                        if meta is None:
                            continue
                        if sample_shape is None:
                            sample_shape = page.shape
                            sample_dtype = page.dtype
                        z_levels.add(int(meta["z_level"]))
                        channels_set.add(str(meta["channel"]).replace(" ", "_"))
            except Exception as e:
                print(f"  Warning: could not read {stack_path.name}: {e}")
                continue
        if not channels_set or sample_shape is None:
            print(f"  Skipping FOV {fov_id}: no readable pages with metadata.")
            return
        if channel_filter is not None:
            channels_set &= set(channel_filter)
            if not channels_set:
                print(f"  Skipping FOV {fov_id}: no channels match the filter.")
                return

        channels = sorted(channels_set)
        n_t, n_z, n_c = len(tp_files), len(z_levels), len(channels)
        n_y, n_x = sample_shape
        sizes = {"t": n_t, "z": n_z, "c": n_c, "y": n_y, "x": n_x}

        sequence = MDASequence(sizes, self.pixel_size, self.params, channels)
        ext = ".ome.tif" if is_ome else ".tif"
        output_file = output_path / f"{fov_id}{ext}"
        writer = OMETiffWriter(output_file)
        writer.sequenceStarted(sequence)
        position_sizes = writer.position_sizes[0]
        print(f"  Dimension order: {list(position_sizes.keys())}")
        print(f"  Shape: {list(position_sizes.values())}")

        mmap = writer.new_array("0", sample_dtype, position_sizes)

        def get_index_tuple(t_idx: int, z_idx: int, c_idx: int) -> tuple:
            v = {"t": t_idx, "z": z_idx, "c": c_idx}
            return tuple(v[d.lower()] for d in position_sizes if d.lower() in v)

        z_idx_map = {z: i for i, z in enumerate(sorted(z_levels))}
        c_idx_map = {ch: i for i, ch in enumerate(channels)}

        # Second pass: write each page to its (t, z, c) slot.
        for t_idx, (_tp, stack_path) in enumerate(tp_files):
            try:
                with tifffile.TiffFile(stack_path) as tf:
                    for page in tf.pages:
                        meta = self._stack_page_metadata(page)
                        if meta is None:
                            continue
                        ch = str(meta["channel"]).replace(" ", "_")
                        z = int(meta["z_level"])
                        if ch not in c_idx_map or z not in z_idx_map:
                            continue
                        writer.write_frame(
                            mmap,
                            get_index_tuple(t_idx, z_idx_map[z], c_idx_map[ch]),
                            page.asarray(),
                        )
            except Exception as e:
                print(f"  Warning: failed to write some pages from {stack_path.name}: {e}")
                continue

        del mmap
        print(f"  Saved {output_file.name}")


def main(acquisition_folder: str, output_folder: str = None, mode: str = "ome",
         channels: Optional[List[str]] = None, fovs: Optional[List[str]] = None):
    converter = AcquisitionConverter(acquisition_folder, output_folder)
    converter.convert_all(mode, channels=channels, fovs=fovs)


def gui_main():
    """Launch the drag-and-drop GUI. Imports PyQt6 lazily so the rest of
    the module can be imported as a library without a Qt install."""
    import sys
    import threading
    try:
        from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                                      QRadioButton, QButtonGroup, QDialog, QCheckBox,
                                      QPushButton, QListWidget, QListWidgetItem, QFileDialog)
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
            self.setFixedSize(500, 320)

            self.output_folder: Optional[str] = None

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

            out_row = QHBoxLayout()
            self.out_label = QLabel("Output: default (next to acquisition)")
            self.out_label.setStyleSheet("color: #444;")
            browse_btn = QPushButton("Choose output…")
            clear_btn = QPushButton("Reset")
            browse_btn.clicked.connect(self._pick_output_folder)
            clear_btn.clicked.connect(self._clear_output_folder)
            out_row.addWidget(self.out_label, 1)
            out_row.addWidget(browse_btn)
            out_row.addWidget(clear_btn)
            layout.addLayout(out_row)

            self.label = QLabel("Drop your acquisition folder here")
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setFont(QFont("Arial", 14))
            self.label.setStyleSheet("border: 2px dashed #888; padding: 40px; color: #444;")
            layout.addWidget(self.label)

            self.setLayout(layout)

        def _pick_output_folder(self):
            chosen = QFileDialog.getExistingDirectory(self, "Select output folder")
            if chosen:
                self.output_folder = chosen
                self.out_label.setText(f"Output: {chosen}")

        def _clear_output_folder(self):
            self.output_folder = None
            self.out_label.setText("Output: default (next to acquisition)")

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

                    out_dir = self.output_folder
                    def run():
                        try:
                            main(folder, output_folder=out_dir, mode=mode,
                                 channels=selected_channels, fovs=selected_fovs)
                            location = out_dir if out_dir else f"{mode}_output folder"
                            self.label.setText(f"Done! Output in {location}.")
                        except Exception as e:
                            self.label.setText(f"Error: {str(e)}")
                        self.setEnabled(True)
                    threading.Thread(target=run).start()

    app = QApplication(sys.argv)
    win = DropBox()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    gui_main()
