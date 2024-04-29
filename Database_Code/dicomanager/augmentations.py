import SimpleITK as sitk
import bisect
import numpy as np
import dbutils as dbu
import xarray as xr
import contextlib
from abc import ABC
from scipy.interpolate import interpn
from scipy.ndimage import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.transform import Rotation


class ToolHandler:
    def __init__(self, tools, db_name, log):
        self.tools = tools
        self.db_name = db_name
        self.log = log

    def empty_row(self, ds):
        row = {}
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                columns = dbu.list_columns(conn, 'augmentations')

        for col in columns:
            simple = col.replace('Original', '')
            fill = ds.attrs.get(simple, None)
            row[col] = fill

        row['CurrentCornerInVoxels'] = np.array([0, 0, 0])
        if hasattr(ds, 'volume'):
            row['OriginalDynamicRange'] = np.array([ds.volume.min(), ds.volume.max()])
        return row

    def load_volume(self, series):
        with xr.open_dataset(series['FilePath']) as ncfile:
            ds = ncfile.load()
        return ds

    def save_modified(self, ds, new_row, series):
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                dbu.add_rows(conn, 'augmentations', new_row)
        try:
            comp = dict(compression='lzf')
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(series['FilePath'], encoding=encoding, engine='h5netcdf')
        except Exception:
            ds.to_netcdf(series['FilePath'])

    def apply(self, series):
        series = series[0]
        ds = self.load_volume(series)
        new_row = self.empty_row(ds)

        # Apply tools
        for tool in self.tools:
            ds, new_row = tool(ds, new_row, self.log, self.db_name)

        # Overwrite previous volume
        self.save_modified(ds, new_row, series)


def passes_filter(ds, row, filter_by):
    merged = {**ds.attrs, **row}
    if filter_by is None:
        return True
    for key, values in filter_by.items():
        if merged.get(key, None) in values:
            return True
    return False


def calc_coords(shape, coordinate_system, dims):
    coordinate_system = coordinate_system.reshape(4, 4)
    out = {}
    for i in range(3):
        temp = np.zeros((4, shape[i]))
        temp[i] = np.arange(shape[i])
        temp[3] = 1
        coords = np.tensordot(coordinate_system[:3], temp, axes=([1, 0]))
        out[list(dims.keys())[i]] = coords[i].astype('float32')
    return out


class WindowLevel:
    def __init__(self, window, level, filter_by=None):
        self.window = window
        self.level = level
        self.filter_by = filter_by

    def __call__(self, ds, row, log, db_name):
        if not passes_filter(ds, row, self.filter_by):
            return (ds, row)

        if ds.attrs['Modality'] != 'CT':
            return (ds, row)

        if hasattr(ds.attrs, 'Window') or hasattr(ds.attrs, 'Level'):
            return (ds, row)

        lohi = {}
        for name, vol in ds.data_vars.items():
            lo_val = self.level - self.window / 2
            hi_val = lo_val + self.window

            vol = vol.to_numpy()
            lohi[name] = np.array([vol.min(), vol.max()])

            vol[vol < lo_val] = lo_val
            vol[vol > hi_val] = hi_val
            ds[name] = (ds[name].dims, vol)

        row['Window'] = self.window
        row['Level'] = self.level
        row['OriginalDynamicRange'] = np.array(list(lohi.values()))[0]
        ds.attrs['Window'] = self.window
        ds.attrs['Level'] = self.level
        formatted = []
        for name, values in lohi.items():
            formatted.append([values[0], values[1]])
        formatted = np.array(formatted).flatten()
        ds.attrs['OriginalDynamicRange'] = formatted
        return ds, row


class Normalize:
    def __init__(self, filter_by=None):
        self.filter_by = filter_by

    def __call__(self, ds, row, log, db_name):
        if not passes_filter(ds, row, self.filter_by):
            return (ds, row)

        for name, vol in ds.data_vars.items():
            vmin, vmax = (vol.min(), vol.max())
            if vmin != vmax:
                ds[name] = (vol - vmin) / (vmax - vmin)
            elif vmin:
                ds[name] = vol / vmin

        row['Normalized'] = True
        return ds, row


class Standardize:
    def __init__(self, filter_by=None):
        self.filter_by = filter_by

    def __call__(self, ds, row, log, db_name):
        if not passes_filter(ds, row, self.filter_by):
            return (ds, row)

        for name, vol in ds.data_vars.items():
            std = vol.std()
            if std:
                ds[name] = (vol - vol.mean()) / std

        row['Standardized'] = True
        return ds, row


class Crop:
    def __init__(self, crop_size, centroid, filter_by=None):
        self.crop_size = crop_size
        self.centroid = centroid
        self.filter_by = filter_by

    def __call__(self, ds, row, log, db_name):
        if not passes_filter(ds, row, self.filter_by):
            return (ds, row)

        current_dimensions = np.array(list(ds.sizes.values()))
        corner = self.centroid - self.crop_size // 2
        corner[corner < 0] = 0
        new_system = ds.attrs['CoordinateSystem'].copy()
        new_system[3] = new_system[3] + np.dot(new_system[:3], corner)
        new_coords = calc_coords(self.crop_size, new_system, ds.dims)

        indices = ()
        for i, (point, size) in enumerate(zip(self.centroid, self.crop_size)):
            if size > current_dimensions[i]:
                # Need to log this as an error
                uid = row['SeriesInstanceUID']
                log.critical(f'Crop size: {self.crop_size} larger than volume for series: {uid}')
                size = current_dimensions[i]
            offset_hi = min(0, (current_dimensions[i] - 1) - (point + size//2))
            lo = max(0, point - size//2 - offset_hi)
            hi = lo + size
            indices += slice(lo, hi)

        data_vars = {}
        for name, vol in ds.data_vars.items():
            arr = vol.to_numpy()
            data_vars[name] = (ds[name].dims, arr[indices])

        # Preserve original information in row data
        row['OriginalCoordinateSystem'] = ds.attrs['CoordinateSystem']
        row['OriginalVolumeDimensions'] = current_dimensions
        row['CurrentCornerInVoxels'] = corner

        # Create new dataset containing new data
        new_ds = xr.Dataset(data_vars=data_vars, coords=new_coords)
        new_ds.attrs = ds.attrs
        new_ds.attrs['CoordinateSystem'] = new_system.flatten()
        new_ds.attrs['OriginalVolumeDimensions'] = current_dimensions
        new_ds.attrs['OriginalCoordinateSystem'] = ds.attrs['CoordinateSystem']
        new_ds.attrs['CurrentCornerInVoxels'] = corner
        return (new_ds, row)


class Resample:
    def __init__(self, voxel_size=None, volume_dimensions=None, filter_by=None):
        assert voxel_size is not None or volume_dimensions is not None, 'Must specify Voxel Size or Dimensions'
        self.voxel_size = voxel_size
        self.volume_dimensions = volume_dimensions
        self.filter_by = filter_by

    def __call__(self, ds, row, log, db_name):
        if not passes_filter(ds, row, self.filter_by):
            return (ds, row)

        # Return if already correct, else compute new dimensions / voxel size
        current_dimensions = np.array(list(ds.sizes.values()))
        if self.volume_dimensions:
            if np.array_equal(self.volume_dimensions, current_dimensions):
                return (ds, row)
            new_shape = np.array(self.volume_dimensions, dtype='int32')
            new_voxel = ds.attrs['VoxelSize'] * (current_dimensions / self.volume_dimensions)
        elif self.voxel_size:
            if np.array_equal(self.voxel_size, ds.attrs['VoxelSize']):
                return (ds, row)
            new_shape = np.round(current_dimensions * (ds.attrs['VoxelSize'] / self.voxel_size)).astype('int32')
            new_voxel = np.array(self.voxel_size, dtype='float64')

        # Set interpolator
        interp = sitk.ResampleImageFilter()
        interp.SetOutputSpacing(new_voxel)
        interp.SetSize(new_shape.tolist())
        interp.SetOutputPixelType(sitk.sitkFloat32)
        interp.SetInterpolator(sitk.sitkLinear)
        if ds.attrs['Modality'] == 'RTSTRUCT':
            interp.SetInterpolator(sitk.sitkNearestNeighbor)

        # Interpolate for each DataArray in the Dataset
        data_vars = {}
        for name, vol in ds.data_vars.items():
            if vol.shape:
                raw_sitk = sitk.GetImageFromArray(vol.T)
                raw_sitk.SetSpacing(ds.attrs['VoxelSize'].astype('float64'))
                new_sitk = interp.Execute(raw_sitk)
                new_arr = sitk.GetArrayFromImage(new_sitk).T
                data_vars[name] = (ds[name].dims, new_arr.astype(ds[name].dtype))
            else:
                data_vars[name] = (ds[name].dims, vol.to_numpy())

        # Preserve original information in row data
        row['OriginalCoordinateSystem'] = ds.attrs['CoordinateSystem']
        row['OriginalVolumeDimensions'] = current_dimensions
        row['OriginalVoxelSize'] = ds.attrs['VoxelSize']

        # Compute new attrs information
        new_system = ds.attrs['CoordinateSystem'].copy()
        new_system[:3] *= new_voxel / ds.attrs['VoxelSize']
        new_coords = calc_coords(new_shape, new_system, ds.dims)

        # Create new dataset containing new data
        new_ds = xr.Dataset(data_vars=data_vars, coords=new_coords)
        new_ds.attrs = ds.attrs
        new_ds.attrs['CoordinateSystem'] = new_system.flatten()
        new_ds.attrs['VoxelSize'] = new_voxel
        new_ds.attrs['OriginalCoordinateSystem'] = ds.attrs['CoordinateSystem']
        new_ds.attrs['OriginalVolumeDimensions'] = current_dimensions
        new_ds.attrs['OriginalVoxelSize'] = ds.attrs['VoxelSize']
        return (new_ds, row)


class BasicInterpolation(ABC):
    def __call__(self, ds, row, log, db_name):
        if not passes_filter(ds, row, self.filter_by):
            return (ds, row)

        if not ds.attrs['Modality'] in ['RTSTRUCT', 'CT', 'MR', 'PT', 'NM']:
            return (ds, row)

        if not isinstance(ds.attrs['MissingSlices'], str) and ds.attrs['MissingSlices'].any():
            for name, vol in ds.data_vars.items():
                new_arr = self._function(ds, vol)
                if new_arr is not None:
                    ds[name] = (ds[name].dims, new_arr)

            row['InterpolatedSlices'] = ds.attrs['MissingSlices']
            ds.attrs['MissingSlices'] = 'NULL'
        return (ds, row)

    def _nearest(self, filled, index):
        # Nearest index below interpolation slice
        below = np.array(filled.copy(), dtype='int')
        below = below[below < index]
        lo = min(below, key=lambda x: abs(x-index))
        # Nearest index above interpolation slice
        above = np.array(filled.copy(), dtype='int')
        above = above[above > index]
        hi = min(above, key=lambda x: abs(x-index))
        # Ratio between above and below indices
        ratio = (index - lo) / (hi - lo)
        return (lo, hi, ratio)

    def _interpolate(self, array, empties, filled, interp_fn):
        output = np.copy(array)
        for z in empties:
            lo, hi, ratio = self._nearest(filled, z)
            # TODO: Interpolate along arbitrary axes, need to roll to original before interpolating
            output[..., z] = interp_fn(array[..., lo], array[..., hi], ratio)
            bisect.insort(filled, z)
        return output


class InterpolateImages(BasicInterpolation):
    def __init__(self, filter_by=None):
        self.filter_by = filter_by

    def _function(self, ds, vol):
        if ds.attrs['Modality'] != 'RTSTRUCT':
            filled = sorted(set(range(vol.shape[-1])) - set(ds.attrs['MissingSlices']))
            return self._interpolate(vol, ds.attrs['MissingSlices'], filled, self._linear_interp)

    def _linear_interp(self, bottom, top, ratio):
        return bottom*ratio + top*(1-ratio)


class InterpolateContours(BasicInterpolation):
    def __init__(self, filter_by=None):
        self.filter_by = filter_by

    def _function(self, ds, vol):
        if ds.attrs['Modality'] == 'RTSTRUCT':
            good_slices = set([])
            # TODO: Interpolate along arbitrary axes, find good slices along said axes
            for i in range(vol.shape[-1]):
                if np.any(vol[..., i]):
                    good_slices.add(i)
            good_slices = sorted(good_slices)
            empties = set(range(vol.shape[-1])) - good_slices

            # only take empties within the range of the structure
            empties = [x for x in empties if good_slices.min() < x < good_slices.max()]
            return self._interpolate(vol, ds.attrs['MissingSlices'], good_slices, self._interp_contour)

    def _signed_bwdist(self, im):
        if im.dtype != np.dtype('bool'):
            im = np.array(im, dtype='bool')
        perim = im ^ binary_erosion(im)
        bwdist = distance_transform_edt(1-perim)
        im_ma = -1 * bwdist*np.logical_not(im) + bwdist*im
        return im_ma

    def _interp_contour(self, bottom, top, precision):
        bottom = self._signed_bwdist(bottom)
        top = self._signed_bwdist(top)

        # create ndgrids
        r, c = top.shape
        points = (np.r_[0, 2], np.arange(r), np.arange(c))
        xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
        xi = np.c_[np.full((r*c), precision+1), xi]

        # Interpolate for new plane, reshape to original dims and threshold
        out = interpn(points, np.stack((top, bottom)), xi).reshape((r, c))
        return np.round(out > 0)


class ApplyRegistrations:
    def __init__(self, filter_by=None):
        self.filter_by = filter_by

    def _align(ds, transform_matrix, reffed_ds):
        # Reffed coordinates
        new_system = reffed_ds.attrs['CoordinateSystem'].copy()
        new_voxel = reffed_ds['VoxelSize'].astype('float64')
        new_shape = np.array(list(reffed_ds.sizes.values()))

        # Need to compute a Euler3DTransform Matrix
        rotation_center = (0, 0, 0)
        translation = tuple(transform_matrix[:3, 3])
        rotation = Rotation.from_matrix(transform_matrix[:3, :3])
        angles = rotation.as_euler("xyz", degrees=False)
        euler_transform = sitk.Euler3DTransform(rotation_center, *angles, translation)

        # Configuration Interpolator
        interp = sitk.ResampleImageFilter()
        interp.SetSize(new_shape)
        interp.SetTransform(euler_transform)
        interp.SetOutputDirection(new_system[:3, 3] / new_voxel)
        interp.SetOutputSpacing(new_voxel)
        interp.SetOutputPixelType(sitk.sitkFloat32)
        interp.SetOutputOrigin(new_system.reshape(4, 4)[:3, 3])
        interp.SetInterpolator(sitk.sitkLinear)

        # Construct coordinate systems for resampling and transforming
        coordinate_system = ds.attrs['CoordinateSystem'].copy()
        origin = coordinate_system.reshape(4, 4)[:3, 3]
        voxel_size = ds.attrs['VoxelSize'].astype('float64')
        direction = coordinate_system[:3, :3] / voxel_size

        data_vars = {}
        for name, vol in ds.data_vars.items():
            raw_sitk = sitk.GetImageFromArray(vol.T)
            raw_sitk.SetDirection(direction)
            raw_sitk.SetSpacing(voxel_size)
            raw_sitk.SetOrigin(origin)

            new_sitk = interp.Execute(raw_sitk)
            new_arr = sitk.GetArrayFromImage(new_sitk).T
            data_vars[name] = (ds[name].dims, new_arr.astype(ds[name].dtype))

        new_coords = calc_coords(new_shape, new_system, ds.dims)
        new_ds = xr.Dataset(data_vars=data_vars, coords=new_coords)
        new_ds.attrs = ds.attrs

        return new_ds

    def __call__(self, ds, row, log, db_name):
        # Iterate through each volume and apply the frame of references
        frame = ds.attrs['FrameOfReferenceUID']
        sop = ds.attrs['SOPInstanceUID']

        with contextlib.closing(dbu.make_connection(db_name)) as conn:
            with conn:
                # Find first registration that applies to this instance
                query = f'SELECT * FROM registrations WHERE AlignedFrameOfReferenceUID = "{frame}"'
                registrations = dbu.query_db(db_name, query, True)
                found = False

                # Determine if this image was registered to any other
                for registration in registrations:
                    sop_root = registration['AlignedInstanceUIDRoot']
                    sop_tails = registration['AlignedInstanceUIDTails']
                    aligned_sops = [sop_root + tail for tail in sop_tails]
                    if sop in aligned_sops:
                        found = True
                        break

                if not found:
                    return (ds, row)

                print(sop, registration, type(ds))

                # Find the reference volume that this image was aligned to with this registration
                ref_frame = registration['FrameOfReferenceUID']
                sop_root = registration['ReferenceInstanceUIDRoot']
                sop_tails = registration['ReferenceInstanceUIDTails']
                if sop_tails:
                    reference_sops = [sop_root + tail for tail in sop_tails]
                else:
                    reference_sops = sop_root

                query = f'SELECT * FROM volumes WHERE FrameOfReferenceUID = "{ref_frame}"'
                ref_vols = dbu.query_db(db_name, query, True)

                for ref_vol in ref_vols:
                    if ref_vol['SOPInstanceUID'] in reference_sops:
                        # Reference and aligned image volume exists
                        with xr.open_dataset(ref_vol['FilePath']) as ncfile:
                            ref_ds = ncfile.load()

                        aligned = self._align(ds, registration, ref_ds)
                        return (aligned, row)
                return (ds, row)
