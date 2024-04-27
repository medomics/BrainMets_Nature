import numpy as np
import pandas as pd
import pydicom
import utils
import dbutils as dbu
import re
import cv2
import dcmio
import corrections as corr
import contextlib
import os
from dataclasses import dataclass
from tqdm import tqdm
from scipy import linalg
from multiprocess import Pool
from abc import ABC


os.system('export OMP_NUM_THREADS=4')
cv2.setNumThreads(4)


@dataclass
class ImageDataGroup:
    ds: pydicom.dataset.Dataset
    volume: np.ndarray
    missing: list = None
    units: str = 'Unitless'


class VolumeUtils(ABC):
    def coordinate_to_ijk(self, coordinate):
        M_inv = linalg.inv(self.coordinate_system)
        if len(coordinate.shape) < 2:
            coordinate = np.expand_dims(coordinate, axis=-1).T
        pad = np.ones((coordinate.shape[0], 1))
        coord_fmt = np.hstack((coordinate, pad)).astype('float32')
        ijk = linalg.blas.sgemm(1, M_inv, coord_fmt, trans_b=True).T
        return np.round(ijk)[:, :3].astype('int')

    def ijk_to_coordinate(self, ijk):
        if len(ijk.shape) < 2:
            ijk = np.expand_dims(ijk, axis=-1).T
        pad = np.ones((ijk.shape[0], 1))
        ijk_fmt = np.hstack((ijk, pad)).astype('float32')
        return linalg.blas.sgemm(1, self.coordinate_system, ijk_fmt, trans_b=True).T[:, :3]

    def ipp_to_k_index(self, ipp):
        ijk = self.coordinate_to_ijk(ipp)
        return ijk[0, -1]

    def k_to_coordinate(self, k):
        return self.ijk_to_coordinate(np.array([0, 0, k]))

    def axes_ordered(self):
        axes = np.argmax(np.abs(self.coordinate_system[:3, :3]), axis=0)
        return [['x', 'y', 'z'][i] for i in axes]

    def coordinate_axes(self):
        out = []
        for i in range(3):
            temp = np.zeros((4, self.shape[i]))
            temp[i] = np.arange(self.shape[i])
            temp[3] = 1
            coords = np.tensordot(
                self.coordinate_system[:3], temp, axes=([1, 0]))
            out.append(coords[i].astype('float32'))
        return out


class VolumeProperties(VolumeUtils):
    def __init__(self, series_df):
        self.series_uid = series_df.iloc[0]['SeriesInstanceUID']
        self._compute_from_series_df(series_df)

    def __str__(self):
        return utils.generic_repr(self)

    def _get_uniques(self, series_df, name):
        no_null = [x for x in series_df[name] if not np.array_equal(x, 'NULL')]
        if len(no_null) > 1:
            return np.unique(np.stack(no_null), axis=0)
        return np.array(no_null)

    def _compute_from_series_df(self, series_df):
        # Pulls rows, checks for conflicts
        unique_rows = [x for x in series_df['Rows'].unique() if x != 'NULL']
        assert len(unique_rows) == 1, f'Disagreement of Row Count for Series: {self.series_uid}'
        self.rows = unique_rows[0]

        # Pulls columns, checks for conflicts
        unique_cols = [x for x in series_df['Columns'].unique() if x != 'NULL']
        assert len(unique_cols) == 1, f'Disagreement of Column Count for Series: {self.series_uid}'
        self.columns = unique_cols[0]

        # Stop early if CR 2D images
        if series_df['Modality'].iloc[0] == 'CR' or series_df['Modality'].iloc[0] == 'RTIMAGE':
            self.Di, self.Dj, self.Dk = (1, 1, 1)
            self.slices = 1
            return None

        # Pulls pixel spacing, checks for conflicts
        pixel_spacing = self._get_uniques(series_df, 'PixelSpacing')
        assert len(pixel_spacing) == 1, f'Disagreement of Pixel Spacing for Series: {self.series_uid}'
        self.Dj, self.Di = pixel_spacing[0]  # In rows, columns, but i=cols, j=rows

        # Stop early if MG 2D images
        if series_df['Modality'].iloc[0] == 'MG':
            self.Dk = 1
            self.slices = 1
            return None

        # Pulls Image Orientation Patient, checks for conflicts
        iop = self._get_uniques(series_df, 'ImageOrientationPatient')
        if len(iop) != 1:  # Occassionally 1e-7 discrepancies occur, nothing to worry about
            iop_diff = np.sum(np.diff(iop, axis=0))
            assert iop_diff < 1e-5, f'Significant disagreement of Image Orientation Patient for Series: {self.series_uid} \n {iop}'
        self.iop = np.array(iop[0]).reshape(2, 3)  # Flips so we have (rows, cols)

        # 3 Basis vectors for volume coordinates
        basis0 = self.iop[0]  # cols
        basis1 = self.iop[1]  # rows
        basis2 = np.cross(basis0, basis1)

        # Sort IPP and compute magnitude of Z-axis (slice thickness)
        ipp = self._get_uniques(series_df, 'ImagePositionPatient')
        if len(ipp) > 1:
            reindex = np.argsort([np.matmul(basis2, x) for x in ipp])
            ipp_sorted = np.stack(ipp[reindex])
            self.origin = ipp_sorted[0].astype('float32')
        else:
            ipp_sorted = ipp
            self.origin = np.squeeze(ipp)

        # RTDOSE or NM
        if series_df['Modality'].iloc[0] in ['NM', 'RTDOSE']:
            ds = pydicom.dcmread(series_df['FilePath'].iloc[0])
            if hasattr(ds, 'GridFrameOffsetVector'):
                if ds.GridFrameOffsetVector[0]:
                    # First element is the actual z origin
                    self.origin[-1] = ds.GridFrameOffsetVector[0]
                delta_Z = np.diff(ds.GridFrameOffsetVector, axis=0)
                self.Dk = np.min(delta_Z, axis=0)
                self.slices = len(ds.GridFrameOffsetVector)

            if hasattr(ds, 'SliceThickness'):
                self.Dk = ds.SliceThickness
            else:
                self.Dk = 1

            if hasattr(ds, 'NumberOfFrames'):
                self.slices = ds.NumberOfFrames
            else:
                self.slices = 1
            return None

        # Images
        if len(ipp_sorted) > 1:
            delta_Z = np.diff(ipp_sorted, axis=0)
            self.Dk = np.min(np.round(np.linalg.norm(delta_Z, axis=1), 3))
            # Set origin and number of slices from ipp_sorted
            self.slices = 1 + self.ipp_to_k_index(ipp_sorted[-1])
        else:
            self.Dk = 1
            self.slices = 1

    @property
    def voxel_size(self):
        # columns, rows, slices
        return np.abs(np.array([self.Di, self.Dj, self.Dk]))

    @property
    def shape(self):
        return np.array([self.columns, self.rows, self.slices])

    @property
    def coordinate_system(self):
        # Formatting for DICOM Equation C.7.6.2.1-1
        # Split basis into components
        if not hasattr(self, 'iop'):
            return np.identity(4)

        Xx, Xy, Xz = self.iop[0]
        Yx, Yy, Yz = self.iop[1]
        # Coordinates become left-handed to allow for rows, columns indexing
        Zx, Zy, Zz = np.cross(self.iop[0], self.iop[1])

        # Image Patient Position Components
        Sx, Sy, Sz = self.origin

        M = np.array([[Xx*self.Di, Yx*self.Dj, Zx*self.Dk, Sx],
                      [Xy*self.Di, Yy*self.Dj, Zy*self.Dk, Sy],
                      [Xz*self.Di, Yz*self.Dj, Zz*self.Dk, Sz],
                      [0, 0, 0, 1]])
        return M.astype('float32')

    @property
    def patient_coordinates(self):
        # Range of i, j, k
        Ri = np.arange(self.columns, dtype='float32')
        Rj = np.arange(self.rows, dtype='float32')
        Rk = np.arange(self.slices, dtype='float32')
        ijk_grid = np.array(np.meshgrid(Ri, Rj, Rk, 1), dtype='float32')

        # Comes out as (4,X,Y,Z), needs cropping and rolling
        coords_raw = np.tensordot(self.coordinate_system[:3], ijk_grid, axes=([1, 0]))
        coords = np.rollaxis(coords_raw[..., 0], 0, 4)
        return coords.astype('float32')

    @property
    def coordinate_extent(self):
        # (2, 3) as [[min x, y, z], [max x, y, z]]
        xyz_min = self.ijk_to_coordinate([0, 0, 0])
        xyz_max = self.ijk_to_coordinate(self.shape - 1)
        return np.stack([xyz_min, xyz_max])

    @property
    def direction(self):
        direction = self.coordinate_system[:3, :3] / self.voxel_size
        return direction.flatten()

    def xr_format(self):
        xyz = self.coordinate_axes()
        axes = self.axes_ordered()
        return dict(zip(axes, xyz))


class Assembler:
    def __init__(self, save_dir, db_name, corrections, write_nifti,
                 write_xarray, only_new_volumes, log,
                 single_frame_optimized, njobs, progress_bar, **kwargs):
        self.db_name = db_name
        self.corrections = corrections
        self.write_nifti = write_nifti
        self.write_xarray = write_xarray
        self.only_new_volumes = only_new_volumes
        self.log = log
        self.single_frame_optimized = single_frame_optimized
        self.njobs = njobs
        self.progress_bar = progress_bar
        self._filesaver = dcmio.FileSaver(save_dir, log, **kwargs)
        self._supported_image_types = ['CT', 'MR', 'NM', 'PT', 'CR', 'MG', 'RTIMAGE']
        with dbu.make_connection(self.db_name) as conn:
            with conn:
                columns = dbu.list_columns(conn, 'volumes')
                self.ph = dcmio.PullHeader(columns)

    def _coordinate_list_to_ijk(self, coordinate_system, coordinates):
        M_inv = linalg.inv(coordinate_system)
        pad = np.ones((coordinates.shape[0], 1))
        coords_fmt = np.hstack((coordinates, pad))
        ijk = np.tensordot(M_inv, coords_fmt, axes=([1, 1]))

        # Needs to be shape (1, N, 3)
        ijk_fmt = np.round(ijk).astype('int32')[:3].T
        axes = np.argmax(np.abs(coordinate_system[2, :3]))
        return np.array([ijk_fmt]), axes

    def _build_coordinates(self, coordinate_system, volume_dimensions):
        Ri = np.arange(volume_dimensions[0], dtype='float32')
        Rj = np.arange(volume_dimensions[1], dtype='float32')
        Rk = np.arange(volume_dimensions[2], dtype='float32')
        ijk_grid = np.array(np.meshgrid(Ri, Rj, Rk, 1), dtype='float32')

        # Comes out as (3,X,Y,Z), needs rolling
        coords_raw = np.tensordot(coordinate_system[:3], ijk_grid, axes=([1, 0]))
        coords = np.rollaxis(coords_raw[..., 0], 0, 4)
        return coords.astype('float32')

    def _find_referenced_img(self, reffed_instance_uid, current_row):
        # Dose -> Plan -> Struct -> Image
        with contextlib.closing(dbu.make_connection(self.db_name, dicts=True)) as conn:
            with conn:
                subquery = f'SELECT * FROM dicoms WHERE SOPInstanceUID = "{reffed_instance_uid}"'
                # Referenced DICOMs from Series Instance UID
                reffed_dicom = conn.execute(subquery).fetchone()

                # Referenced Volumes
                if reffed_dicom is None:
                    frame = current_row['FrameOfReferenceUID']
                    for mod in ['CT', 'MR', 'PT', 'PET', 'NM']:
                        query = f'SELECT * FROM volumes WHERE FrameOfReferenceUID = "{frame}" AND Modality = "{mod}"'
                        reffed_vol = conn.execute(query).fetchone()
                        if reffed_vol is not None:
                            return reffed_vol
                    self.log.error(f'No images found to register STRUCT/DOSE to in Frame Of Reference: {frame}')
                elif reffed_dicom['Modality'] in ['CT', 'MR']:
                    series_uid = reffed_dicom['SeriesInstanceUID']
                    query = f'SELECT * FROM volumes WHERE SeriesInstanceUID = "{series_uid}"'
                    reffed_vol = conn.execute(query).fetchone()
                    return reffed_vol
                else:
                    return self._find_referenced_img(reffed_dicom['ReferencedSOPInstanceUID'], current_row)

    def _image2d(self, series_df, dims):
        volume = np.zeros(dims.shape, dtype='int16')
        found = []
        sop_ds_pairs = {}

        for _, row in series_df.iterrows():
            index_k = dims.ipp_to_k_index(row['ImagePositionPatient'])
            try:
                ds = pydicom.dcmread(row['FilePath'])
                volume[..., index_k] = ds.pixel_array.T
            except FileNotFoundError:
                self.log.error(f'Cannot find {row["FilePath"]} dicom file')
            except Exception:
                print('exception')
                continue
            else:
                found.append(index_k)
                sop_ds_pairs[ds.SOPInstanceUID] = ds

        sops = list(sop_ds_pairs.keys())
        missing = sorted(set(range(dims.shape[-1])) - set(found))
        missing = None
        image_group = ImageDataGroup(sop_ds_pairs[sops[0]], volume, missing)
        return corr.padding(image_group)

    def _image3d(self, series_df, dims):
        row = series_df.iloc[0]
        ds = pydicom.dcmread(row['FilePath'])
        volume = ds.pixel_array.T
        image_group = ImageDataGroup(ds, volume)
        return corr.padding(image_group)

    def dose(self, series_df):
        dose_dims = VolumeProperties(series_df)
        rows = []

        for _, row in series_df.iterrows():
            try:
                ds = pydicom.dcmread(row['FilePath'])
            except FileNotFoundError:
                self.log.error(f'Cannot find {row["FilePath"]} dicom file')
                return None

            dose_vol = ds.pixel_array * ds.DoseGridScaling  # Z, Y, X

            reffed_img = self._find_referenced_img(row['ReferencedSOPInstanceUID'], row)
            if reffed_img and self.corrections and self.corrections.get('InterpolatedDose', None):
                dose_vol = corr.interpolate(dose_vol, dose_dims, reffed_img)
            else:
                dose_vol = dose_vol.T
                reffed_img = {}
                reffed_img['CoordinateSystem'] = dose_dims.coordinate_system
                reffed_img['VolumeDimensions'] = dose_dims.shape
                reffed_img['VoxelSize'] = dose_dims.voxel_size

            # Save and return row to write to db
            row = self._save_dependent(ds, reffed_img, dose_vol, series_df)
            rows.append(row)
        return rows

    def struct(self, series_df):
        rows = []  # May potentially have multiple RTSTRUCT per series
        for _, row in series_df.iterrows():
            reffed_img = self._find_referenced_img(row['ReferencedSOPInstanceUID'], row)
            if reffed_img is None:
                # No corresponding images
                continue

            coordinate_system = reffed_img['CoordinateSystem']
            volume_dimensions = reffed_img['VolumeDimensions'].astype('int32')
            fpath = row['FilePath']

            try:
                ds = pydicom.dcmread(fpath)
            except FileNotFoundError:
                self.log.error(f'Cannot find {fpath} dicom file')
                continue

            structures = {}
            for contour in ds.StructureSetROISequence:
                # NetCDF doesn't like special chars (*^~)
                cleaned_name = re.sub(r'[^a-zA-Z0-9 ()_]', '', contour.ROIName)

                contour_index = {}
                for i, cs in enumerate(ds.ROIContourSequence):
                    contour_index[cs.ReferencedROINumber] = i

                # Check if this contour is empty
                try:
                    index = contour_index[contour.ROINumber]
                    roi = ds.ROIContourSequence[index]
                except Exception:
                    self.log.error(f'Missing ROIContourSequence preventing assembly within: {fpath}')
                    structures[cleaned_name] = None
                    continue

                if not hasattr(roi, 'ContourSequence') or not len(roi.ContourSequence):
                    self.log.warning(f'Empty structure for {cleaned_name} within: {fpath}')
                    structures[cleaned_name] = None
                    continue

                # Create array and fill through proper axis
                fill_array = np.zeros(volume_dimensions, dtype='uint8')
                for one_slice in roi.ContourSequence:
                    try:
                        contour_pts = np.array(one_slice.ContourData).reshape(-1, 3)
                    except Exception:
                        # Incase there isn't 3N points due to corruption, or DICOM just being DICOM
                        n_pts = 3 * len(one_slice.ContourData) // 3
                        contour_pts = np.array(one_slice.ContourData[:n_pts]).reshape(-1, 3)
                    #print(coordinate_system)
                    ijk_pts, axes = self._coordinate_list_to_ijk(coordinate_system, contour_pts)
                    ortho_pt = np.round(np.mean(ijk_pts[..., axes])).astype('int32')

                    # Handles segmentation in any plane
                    if axes == 0:
                        axes_fmt = slice(1, None, None)
                    elif axes == 1:
                        axes_fmt = slice(None, None, 2)
                    elif axes == 2:
                        axes_fmt = slice(None, 2, None)

                    # Constructs plane into filled polygon
                    in_plane_pts = np.array(ijk_pts[..., axes_fmt], dtype='int32')
                    plane_dims = volume_dimensions[axes_fmt][::-1]  # NEED TO CHECK!!!
                    poly2D = np.zeros(plane_dims, dtype='uint8')

                    cv2.fillPoly(poly2D, in_plane_pts, 1)

                    # Place polygon into proper plane of the volume
                    axes_fill = [slice(None) for _ in range(axes)] + [ortho_pt]
                    try:
                        fill_array[tuple(axes_fill)] += np.array(poly2D.T, dtype='uint8')
                    except Exception as e:
                        print(e, fpath, contour, fill_array.shape)
                        # TODO: Need to figure out why some are transposed and other not...
                        fill_array[tuple(axes_fill)] += np.array(poly2D, dtype='uint8')

                # Ensures no overlaps exist and handles improper inner struct encoding
                fill_array = fill_array % 2
                structures[cleaned_name] = fill_array

            row = self._save_dependent(ds, reffed_img, structures, series_df)
            rows.append(row)
        return rows

    def mg(self, series_df, dims):
        ds = pydicom.dcmread(series_df['FilePath'].iloc[0])
        volume = ds.pixel_array.T
        return ImageDataGroup(ds, volume)

    def cr(self, series_df, dims):
        ds = pydicom.dcmread(series_df['FilePath'].iloc[0])
        volume = ds.pixel_array.T
        return ImageDataGroup(ds, volume)

    def ct(self, series_df, dims):
        image_group = self._image2d(series_df, dims)
        image_group.units = 'CTNumber'
        if self.corrections and self.corrections.get('Hounsfield', None):
            image_group = corr.hounsfield(image_group)
        return image_group

    def mr(self, series_df, dims):
        image_group = self._image2d(series_df, dims)
        if self.corrections and self.corrections.get('N4BiasField', None):
            image_group = corr.n4_bias(image_group)
        return image_group

    def pt(self, series_df, dims):
        image_group = self._image2d(series_df, dims)
        if self.corrections and self.corrections.get('SUVBodyWeight', None):
            image_group = corr.suv_bw(image_group)
        return image_group

    def nm(self, series_df, dims):
        image_group = self._image3d(series_df, dims)
        if self.corrections and self.corrections.get('NMCounts', None):
            image_group = corr.counts(image_group)
        return image_group

    def rtimage(self, series_df, dims):
        ds = pydicom.dcmread(series_df['FilePath'].iloc[0])
        volume = ds.pixel_array.T
        return ImageDataGroup(ds, volume)

    def assemble(self, frame):
        with dbu.make_connection(self.db_name) as conn:
            row = frame[0]
            frame_uid = row['FrameOfReferenceUID']
            query = f'SELECT SeriesInstanceUID FROM volumes WHERE FrameOfReferenceUID == "{frame_uid}"'
            volume_series_uids = utils.unnest([x for x in dbu.query_db(self.db_name, query)])

            # Images like CT, MR, NM, PET
            rows = self._assemble_images(frame, volume_series_uids)
            dbu.add_rows(conn, table='volumes', rows=rows)

            frame_df = pd.DataFrame(frame)
            series_iter = utils.series_df_generator(frame_df, volume_series_uids=volume_series_uids)
            temp = utils.series_df_generator(frame_df, volume_series_uids=volume_series_uids)
            n = sum([1 for _ in temp])

            params = dict(total=n, desc='Assembly')

            if self.single_frame_optimized and n > 10:
                with Pool(processes=self.njobs) as P:
                    if self.progress_bar:
                        rows_nested = list(tqdm(P.imap_unordered(self._dependent_mapper, series_iter), **params))
                    else:
                        rows_nested = list(P.imap_unordered(self._dependent_mapper, series_iter))
            elif self.single_frame_optimized and self.progress_bar:
                rows_nested = list(tqdm(map(self._dependent_mapper, series_iter), **params))
            else:
                rows_nested = list(map(self._dependent_mapper, series_iter))
            rows = utils.unnest(rows_nested)
            dbu.add_rows(conn, table='volumes', rows=rows)
        return rows

    def _assemble_images(self, frame: pd.DataFrame, volume_series_uids) -> None:
        frame_df = pd.DataFrame(frame)
        rows = []

        for series_df in utils.series_df_generator(frame_df, self._supported_image_types, volume_series_uids):
            modality = series_df.iloc[0]['Modality']
            dims = VolumeProperties(series_df)
            mod_fn = getattr(self, modality.lower(), None)
            if mod_fn:
                image_group = mod_fn(series_df, dims)
            else:
                self.log.error(f'No function for: {modality}')
                continue

            # Saves to disk
            row = self.ph.from_ds(image_group.ds)
            row['Units'] = image_group.units
            if image_group.missing:
                row['MissingSlices'] = np.array(image_group.missing)
                self.log.warning(f'Series {row["SeriesInstanceUID"]} is missing slices: {str(row["MissingSlices"].flatten())}')
            else:
                row['MissingSlices'] = 'NULL'
            row['CoordinateSystem'] = dims.coordinate_system
            row['VoxelSize'] = dims.voxel_size
            row['VolumeDimensions'] = dims.shape
            row['Institution'] = series_df.iloc[0]['Institution']
            row['NiftiROIs'] = None
            row['NiftiFilePaths'] = None
            row['FilePath'] = None
            if self.write_nifti:
                _, nifti_file = self._filesaver.save_nifti(row, image_group.volume)
                row['NiftiFilePaths'] = nifti_file
            if self.write_xarray:
                row['FilePath'] = self._filesaver.save_xarray(row, dims.xr_format(), image_group.volume)
            if not self.write_nifti and not self.write_xarray:
                self.log.error(f'No files written for {row["SOPInstanceUID"]}')
            rows.append(row)
        return rows

    def _dependent_mapper(self, series_df):
        modality = series_df.iloc[0]['Modality']

        if modality == 'RTSTRUCT':
            return self.struct(series_df)
        elif modality == 'RTDOSE':
            return self.dose(series_df)
        elif modality == 'RTPLAN':
            # Need to figure out what to do with plans
            pass
        elif modality == 'REG':
            # Will likely only be useful for applying registrations
            pass
        elif modality not in self._supported_image_types:
            print(f'{modality} filetype is currently not supported')
        return []

    def _assemble_dependents(self, frame, volume_series_uids):
        frame_df = pd.DataFrame(frame)
        rows = []

        for series_df in utils.series_df_generator(frame_df, volume_series_uids=volume_series_uids):
            modality = series_df.iloc[0]['Modality']

            if modality == 'RTSTRUCT':
                rows.extend(self.struct(series_df))
            elif modality == 'RTDOSE':
                rows.extend(self.dose(series_df))
            elif modality == 'RTPLAN':
                # Need to figure out what to do with plans
                pass
            elif modality == 'REG':
                # Will likely only be useful for applying registrations
                pass
            elif modality not in self._supported_image_types:
                print(f'{modality} filetype is currently not supported')
        return rows

    def _save_dependent(self, ds, reffed_img, assembled, series_df):
        volume_dimensions = reffed_img['VolumeDimensions'].astype('int32')
        coordinate_system = reffed_img['CoordinateSystem']

        # Build the coordinates needed, make a function to handle this normally
        coords = {}
        coord_order = np.argmax(np.abs(coordinate_system[:3, :3]), axis=0).astype('int')
        axes = [['x', 'y', 'z'][i] for i in coord_order]

        for i, n in enumerate(axes):
            axes_ijk = np.zeros((4, volume_dimensions[i]))
            axes_ijk[i] = np.arange(volume_dimensions[i])
            axes_ijk[3] = 1
            axes_coords = np.tensordot(coordinate_system, axes_ijk, axes=([1, 0]))
            coords[n] = axes_coords[i].astype('float32')

        # Organizes the row to save to the database
        row = self.ph.from_ds(ds)
        if isinstance(assembled, dict):
            row['StructureNames'] = list(assembled.keys())
        if hasattr(ds, 'DoseUnits'):
            row['Units'] = ds.DoseUnits
        row['CoordinateSystem'] = reffed_img['CoordinateSystem']
        row['VoxelSize'] = reffed_img['VoxelSize']
        row['VolumeDimensions'] = reffed_img['VolumeDimensions']
        row['Institution'] = series_df.iloc[0]['Institution']
        row['NiftiROIs'] = None
        row['FilePath'] = None

        if self.write_nifti:
            nifti_roi_files = self._filesaver.save_nifti(row, assembled)
            row['NiftiROIs'] = list(nifti_roi_files.keys())
            row['NiftiFilePaths'] = list(nifti_roi_files.values())
        if self.write_xarray:
            row['FilePath'] = self._filesaver.save_xarray(row, coords, assembled)
        if not self.write_nifti and not self.write_xarray:
            self.log.error(f'No files written for {row["SOPInstanceUID"]}')
        return row
