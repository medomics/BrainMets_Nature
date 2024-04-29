import os
import shutil
import multiprocess
import xarray as xr
import numpy as np
import re
import SimpleITK as sitk
import dbutils as dbu
import utils
import pydicom
import contextlib
from glob import glob
from tqdm import tqdm


class FileSaver:
    def __init__(self, save_dir, log, save_transposed=True, overwrite=False, prefixes=True, as_nifti=False):
        self.save_dir = os.path.expanduser(save_dir)
        self.log = log
        self.overwrite = overwrite
        self.prefixes = prefixes
        self.save_transposed = save_transposed
        self.as_nifti = as_nifti

    # We can save as nifti using dicom2nifti or build a simple version myself
    # this will rely on the assembly likely enough.
    def _generate_name(self, row):
        reldir = []
        reldir.append(row.get('Institution', ''))
        reldir.append(row.get('PatientID', ''))
        reldir.append(row.get('FrameOfReferenceUID', ''))
        reldir.append(row.get('StudyInstanceUID', ''))
        reldir.append(row.get('SeriesInstanceUID', ''))

        if self.prefixes:
            prefix = ['', 'Patient_', 'Frame_', 'Study_', 'Series_']
            reldir = [x+y for x, y in zip(prefix, reldir)]

        series_dscp = row.get('SeriesDescription', '')
        series_dscp_clean = re.sub(r'[^a-zA-Z0-9.\_]+', ' ', series_dscp)
        reldir.append(row.get('Modality', '') + '_' + series_dscp_clean)
        reldir.append(row.get('SOPInstanceUID'))
        newpath = os.path.join(self.save_dir, *reldir)
        return newpath

    def _make_dir(self, newpath):
        if self.overwrite or not os.path.exists(newpath):
            if not os.path.exists(os.path.dirname(newpath)):
                try:
                    os.makedirs(os.path.dirname(newpath))
                except FileExistsError:
                    pass

    def _make_nifti(self, row, volume):
        coordinate_system = row['CoordinateSystem'].astype('float64')
        voxel_size = row['VoxelSize'].astype('float64')
        origin = coordinate_system[:3, -1]
        direction = coordinate_system[:3, :3] / voxel_size

        img = sitk.GetImageFromArray(volume.T)
        img.SetOrigin(origin)
        img.SetSpacing(voxel_size)
        img.SetDirection(direction.T.flatten())
        return img

    def save_nifti(self, row, assembled):
        filepath = self._generate_name(row)
        self._make_dir(filepath)

        roi_filenames = {}
        if row['Modality'] == 'RTSTRUCT':
            for name, volume in assembled.items():
                if volume is None or not np.any(volume):
                    self.log.warning(f'{name} in {row["SOPInstanceUID"]} is empty, not writing to .nii.gz')
                    continue
                dirname = os.path.dirname(filepath)
                filename = f'{dirname}/{name}.nii.gz'
                img = self._make_nifti(row, volume)
                img.SetMetaData('descrip', f'SOP_{row["SOPInstanceUID"]}__{row["SeriesDescription"]}')
                sitk.WriteImage(img, filename)
                roi_filenames[name] = filename
            return roi_filenames
        else:
            filename = filepath + '.nii.gz'
            img = self._make_nifti(row, assembled)
            img.SetMetaData('descrip', f'SOP_{row["SOPInstanceUID"]}')
            sitk.WriteImage(img, filename, True)
            return (None, [filename])

    def save_xarray(self, row, coords, assembled):
        newpath = self._generate_name(row) + '.nc'

        # Format for dataset
        axes = list(coords.keys())
        if isinstance(assembled, dict):
            data_vars = {}
            for name, vol in assembled.items():
                if vol is None:
                    data_vars[name] = ([], np.nan)
                else:
                    data_vars[name] = (axes, vol.astype('uint8'))
        else:
            data_vars = {'volume': (
                axes[:assembled.ndim], assembled.astype('float32'))}

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Remove unneeded data
        attrs = row.copy()
        del attrs['VolumeDimensions']
        del attrs['Transposed']
        del attrs['NiftiFilePaths']
        del attrs['FilePath']
        if attrs['StructureNames'] == 'NULL':
            del attrs['StructureNames']
        attrs['CoordinateSystem'] = attrs['CoordinateSystem'].flatten()

        # Removes unaccepted string chars
        cleaned_attrs = {}
        for key, value in attrs.items():
            if type(value) is str:
                value = re.sub(r'[^a-zA-Z0-9.\_]+', ' ', value)
            if value is None:
                value = 'NULL'
            cleaned_attrs[key] = value

        ds.attrs = cleaned_attrs

        row['Transposed'] = False
        if self.save_transposed:
            row['Transposed'] = True
            # Need to update voxel spacing here too
            new_axes = [axes[i] for i in [1, 0, 2]]
            ds = ds.transpose(*new_axes[:len(coords)])

        # Make filepath and save with compression
        self._make_dir(newpath)
        comp = dict(compression='gzip', complevel=3)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(newpath, encoding=encoding, engine='h5netcdf')
        try:
            comp = dict(compression='gzip', complevel=3)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(newpath, encoding=encoding, engine='h5netcdf')
        except Exception:
            self.log.warning(f'Could compress to {newpath}')
            ds.to_netcdf(newpath)
        return newpath

    def save_new_rtstruct(self, row, rtstruct):
        newpath = self._generate_name(row) + '.dcm'
        self._make_dir(newpath)
        rtstruct.save_as(newpath)
        return newpath

    # for nifti we will have this be the create_filepath
    # then we have either save dicom or save_as_nifti
    def save_dicoms(self, rows):
        if not isinstance(rows, list):
            rows = [rows]

        newpaths = []
        for row in rows:
            newpath = self._generate_name(row) + '.dcm'
            self._make_dir(newpath)

            try:
                shutil.copy(row['FilePath'], newpath)
            except FileExistsError:  # duplicate source file
                continue
            except FileNotFoundError:  # missing source file
                continue
            except IOError:
                os.makedirs(os.path.dirname(newpath))
                shutil.copy(row['FilePath'], newpath)
            newpaths.append(newpath)
        return newpaths


class PullHeader:
    def __init__(self, dcm_cols, reg_cols=None, filter_by=None, log=None, copy_dir=None, institution=None):
        self.dcm_cols = dcm_cols
        self.reg_cols = reg_cols
        self.log = log
        self.copy_dir = copy_dir
        self.institution = institution
        if copy_dir is not None:
            self.fs = FileSaver(copy_dir, False, True)

        if isinstance(filter_by, str):
            self.filter_by = utils.read_yaml(filter_by)
        elif callable(filter_by):
            self._filter = filter_by
        else:
            self.filter_by = filter_by

    def _pull_fields(self, ds, dcmfile):
        datafields = {}
        for col in self.dcm_cols:
            datafields[col] = None

        # TODO: Should expand filtering but theres like 100 classes
        # Raw Data Storage SOP Class UID
        if ds.file_meta.MediaStorageSOPClassUID == '1.2.840.10008.5.1.4.1.1.66':
            return None
        for name in self.dcm_cols:
            if name == 'PatientName' and hasattr(ds, 'PatientName'):
                fmt_name = str(getattr(ds, name))
                datafields[name] = re.sub(r'[^a-zA-Z0-9.\_]+', ' ', fmt_name)
            elif name == 'SeriesDescription' and hasattr(ds, 'SeriesDescription'):
                datafields[name] = re.sub(r'[^a-zA-Z0-9.\_]+', ' ', ds.SeriesDescription)
            elif name == 'ReferringPhysicianName' and hasattr(ds, 'ReferringPhysicianName'):
                fmt_name = str(getattr(ds, name))
                datafields[name] = re.sub(r'[^a-zA-Z0-9.\_]+', ' ', fmt_name)
            elif name == 'ImageOrientationPatient' and hasattr(ds, 'DetectorInformationSequence'):
                iop = ds.DetectorInformationSequence[0].ImageOrientationPatient
                datafields[name] = iop
            elif name == 'ImagePositionPatient' and hasattr(ds, 'DetectorInformationSequence'):
                ipp = ds.DetectorInformationSequence[0].ImagePositionPatient
                if ipp is None:
                    ipp = np.array([0, 0, 0])
                datafields[name] = ipp
            elif name == 'FrameOfReferenceUID' and hasattr(ds, 'ReferencedFrameOfReferenceSequence'):
                frame = ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                datafields[name] = frame
            elif name == 'ReferencedSOPInstanceUID' and hasattr(ds, 'ReferencedFrameOfReferenceSequence'):
                ref_frame = ds.ReferencedFrameOfReferenceSequence[0]
                ref_study = ref_frame.RTReferencedStudySequence[0]
                ref_series = ref_study.RTReferencedSeriesSequence[0]
                img_seq = ref_series.ContourImageSequence
                uids = []
                for img in img_seq:
                    uids.append(img.ReferencedSOPInstanceUID)
                uids.sort()
                datafields[name] = uids[0]
            elif name == 'ReferencedSOPInstanceUID' and hasattr(ds, 'ReferencedRTPlanSequence'):
                uids = []
                for ref in ds.ReferencedRTPlanSequence:
                    uids.append(ref.ReferencedSOPInstanceUID)
                uids.sort()
                datafields[name] = uids[0]
            elif name == 'ReferencedSOPInstanceUID' and hasattr(ds, 'ReferencedStructureSetSequence'):
                uids = []
                for ref in ds.ReferencedStructureSetSequence:
                    uids.append(ref.ReferencedSOPInstanceUID)
                uids.sort()
                datafields[name] = uids[0]
            elif name == 'ReferencedSOPInstanceUID' and hasattr(ds, 'RegistrationSequence'):
                # This pulls the SOPInstanceUID which the REG file transforms
                for reg in ds.RegistrationSequence:
                    # Disregards any registrations to the same frame of reference
                    uids = []
                    if reg.FrameOfReferenceUID != ds.FrameOfReferenceUID:
                        uids.append(reg.ReferencedImageSequence[0].ReferencedSOPInstanceUID)
                    uids.sort()
                    datafields[name] = uids[0]
                else:
                    if self.log:
                        self.log.warning(f'Registration file references its own FrameOfReferenceUID: {dcmfile}')
                    datafields[name] = 'SELF-REFERENCED'
                    continue
            elif name == 'GeneratedStruct':
                datafields[name] = False
            elif hasattr(ds, name):
                datafields[name] = getattr(ds, name)
            else:
                datafields[name] = 'NULL'

        structure_names = []
        if ds.Modality == 'RTSTRUCT':
            for contour in ds.StructureSetROISequence:
                cleaned_name = re.sub(r'[^a-zA-Z0-9 ()_]', '', contour.ROIName)
                structure_names.append(cleaned_name)

        if len(structure_names):
            datafields['StructureNames'] = structure_names

        split = ds.SeriesDescription.split('_')
        if len(split) > 2:
            datafields['OriginalPatientID'] = split[0]
            datafields['OriginalSeriesDescription'] = split[-1]
            datafields['OriginalSeriesDate'] = split[1]

        for key, val in datafields.items():
            if isinstance(val, pydicom.uid.UID):
                datafields[key] = str(val)
            elif isinstance(val, pydicom.multival.MultiValue):
                datafields[key] = np.array(val)

        datafields['FilePath'] = dcmfile

        if self.institution:
            datafields['Institution'] = self.institution

        if self.copy_dir is not None and type(self.copy_dir) is str:
            newpath = self.fs.save_dicoms(datafields)
            datafields['FilePath'] = newpath[0]

        return datafields

    def _pull_reg(self, ds):
        datafields = {}
        for col in self.reg_cols:
            datafields[col] = None

        datafields['SOPInstanceUID'] = ds.SOPInstanceUID
        datafields['FrameOfReferenceUID'] = ds.FrameOfReferenceUID

        for reg_seq in ds.RegistrationSequence:
            matrix = reg_seq.MatrixRegistrationSequence[0].MatrixSequence[0]
            transform = matrix.FrameOfReferenceTransformationMatrix
            transform_type = matrix.FrameOfReferenceTransformationMatrixType

            if np.array_equal(np.eye(4), np.array(transform).reshape((4, 4))):
                sops = [x.ReferencedSOPInstanceUID for x in reg_seq.ReferencedImageSequence]
                sop_root, sop_tails = utils.split_sops(sops)
                datafields['ReferenceInstanceUIDRoot'] = sop_root
                datafields['ReferenceInstanceUIDTails'] = sop_tails

            datafields['TransformationMatrix'] = np.array(transform).reshape((4, 4))
            datafields['TransformationType'] = transform_type
            datafields['TransformationApplied'] = False
            datafields['TransformedSeriesUID'] = 'NULL'
            sops = [x.ReferencedSOPInstanceUID for x in reg_seq.ReferencedImageSequence]
            sop_root, sop_tails = utils.split_sops(sops)
            datafields['AlignedInstanceUIDRoot'] = sop_root
            datafields['AlignedInstanceUIDTails'] = sop_tails
            datafields['AlignedFrameOfReferenceUID'] = reg_seq.FrameOfReferenceUID

        return datafields

    def _filter(self, ds):
        if self.filter_by is None:
            return True
        all_none = True
        for key, values in self.filter_by.items():
            if values is not None:
                all_none = False
                tag = getattr(ds, key, None)
                if tag in values:
                    return True
        return all_none

    def from_file(self, dcmfile):
        ds = pydicom.dcmread(dcmfile, stop_before_pixels=True, force=True)
        if len(ds) == 1:
            if self.log:
                self.log.error(
                    f'DICOM File Meta Information header missing, PyDICOM refuses to read: {dcmfile}')
            return None

        datafields = self._pull_fields(ds, dcmfile)
        if datafields is None:
            return None

        if self._filter(ds):
            out = [('ALL', datafields)]
            if ds.Modality == 'REG':
                out.append(('REG', self._pull_reg(ds)))
            return out
        return None

    def from_ds(self, ds):
        name = f'SeriesInstanceUID: {ds.SeriesInstanceUID}'
        datafields = self._pull_fields(ds, name)
        for key, val in datafields.items():
            if isinstance(val, pydicom.uid.UID):
                datafields[key] = str(val)
            elif isinstance(val, pydicom.multival.MultiValue):
                datafields[key] = np.array(val)
        return datafields


@utils.autoset_n_jobs
def digest_dicoms(db_name, dicom_dir, filter_by=None, njobs=None, log=None,
                  progress_bar=True, copy_dir=None, only_digest_new=None,
                  institution=None):
    """Digests all DICOMs contained within dicom_dir

    Args:
        conn (sqlite3.Connection): Connection to the database
        dicom_dir (str): A Posix path to a directory containing DICOMs
    """
    with contextlib.closing(dbu.make_connection(db_name, dicts=True)) as conn:
        with conn:
            dcm_cols = dbu.list_columns(conn, 'dicoms')
            reg_cols = dbu.list_columns(conn, 'registrations')

            ph = PullHeader(dcm_cols=dcm_cols, reg_cols=reg_cols, filter_by=filter_by,
                            log=log, copy_dir=copy_dir, institution=institution)
            files = glob(os.path.join(dicom_dir, '**/*.dcm'), recursive=True)

            if only_digest_new and not copy_dir:
                query = 'SELECT FilePath FROM dicoms'
                existing_files = [x[0] for x in dbu.query_db(db_name, query)]
                files = set(files) - set(existing_files)

            params = {}
            if progress_bar:
                params = dict(desc='Digesting', total=len(files))

            if njobs == 1:
                fields = list(tqdm(map(ph.from_file, files), **params))
            else:
                with multiprocess.Pool(processes=njobs) as P:
                    fields = list(tqdm(P.imap_unordered(ph.from_file, files), **params))

            dcm_rows = []
            reg_rows = []
            for field in fields:
                if field is None or not len(field):
                    continue
                for row in field:
                    if row[0] == 'REG':
                        row[1]['Institution'] = institution
                        reg_rows.append(row[1])
                    else:
                        row[1]['Institution'] = institution
                        dcm_rows.append(row[1])

            dbu.add_rows(conn, table='dicoms', rows=dcm_rows)
            dbu.add_rows(conn, table='registrations', rows=reg_rows)
