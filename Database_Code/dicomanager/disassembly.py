import pydicom
from datetime import datetime
import dbutils as dbu
import dcmio
import numpy as np
import copy
import cv2
import colorsys
import xarray as xr
import contextlib
import utils
from scipy.spatial import cKDTree
from assembly import VolumeUtils


class CoordinateConverter(VolumeUtils):
    def __init__(self, image_series_uid, db_name):
        self.db_name = db_name
        self.image_series_uid = image_series_uid
        self.uids = []
        self._make_ipp_tree()
        with contextlib.closing(dbu.make_connection(db_name)) as conn:
            with conn:
                query = f'SELECT CoordinateSystem FROM volumes WHERE SeriesInstanceUID = "{image_series_uid}"'
                self.coordinate_system = conn.execute(query).fetchone()[0]

    def _make_ipp_tree(self):
        with contextlib.closing(dbu.make_connection(self.db_name, dicts=True)) as conn:
            with conn:
                query = f'SELECT * FROM dicoms WHERE SeriesInstanceUID = "{self.image_series_uid}"'
                image_series = conn.execute(query).fetchall()

        ipps = []
        for row in image_series:
            uid = pydicom.dataset.Dataset()
            uid.ReferencedSOPClassUID = row['SOPClassUID']
            uid.ReferencedSOPInstanceUID = row['SOPInstanceUID']
            self.uids.append(uid)
            ipps.append(row['ImagePositionPatient'])
        self.tree = cKDTree(ipps)

    def k_index_to_uid(self, k_index):
        xyz_coord = self.k_to_coordinate(k_index)
        _, index = self.tree.query(xyz_coord)
        return self.uids[index[0]]


class StructCreator:
    def __init__(self, db_name, log):
        self.db_name = db_name
        self.log = log
        self.uid_prefix = '1.2.826.0.1.3680043.10.771.'
        with contextlib.closing(dbu.make_connection(db_name)) as conn:
            with conn:
                self.columns = dbu.list_columns(conn, 'dicoms')
                self.ph = dcmio.PullHeader(self.columns, filter_by=None, log=self.log)

    def _mask_to_contour_sequence(self, mask, k_index, coord_converter):
        ref_uid = coord_converter.k_index_to_uid(k_index)
        # If hierarchy is unneeded, use cv2.RETR_CCOMP to simplify and save ~15% runtime
        ij_lists, hierarchy = cv2.findContours(mask[..., k_index].astype('uint8'),
                                               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        for ij_list in ij_lists:
            k_vec = np.full((ij_list.shape[0], 1), k_index)
            ijk = np.hstack((ij_list[:, 0, :], k_vec))
            coords = coord_converter.ijk_to_coordinate(ijk)
            # Fill the Contour Sequence
            contour_seq = pydicom.dataset.Dataset()
            contour_seq.ContourImageSequence = pydicom.sequence.Sequence([ref_uid])
            contour_seq.ContourGeometricType = 'CLOSED_PLANAR'
            contour_seq.NumberOfContourPoints = len(coords) // 3
            contour_seq.ContourData = [f'{pt:0.3f}' for pt in coords.flatten()]
            contours.append(contour_seq)
        return contours

    def _append_masks(self, rt_ds, image_series_uid, masks, roi_names=None):
        coord_converter = CoordinateConverter(image_series_uid, self.db_name)

        if isinstance(roi_names, str):
            roi_names = [roi_names]
        if isinstance(masks, list):
            masks = np.array(masks)
        if masks.ndim == 3:
            masks = np.expand_dims(masks, axis=0)
        if not roi_names:
            roi_names = ['NewName' + str(x) for x in range(masks.shape[0])]

        hsv_values = np.linspace(0, 1, masks.shape[0])
        for mask, roi_name, hsv in zip(masks, roi_names, hsv_values):
            rgb_float = np.array(colorsys.hsv_to_rgb(hsv, 1, 1))
            rgb_color = np.round(255 * rgb_float).astype('uint8')
            roi_number = len(rt_ds.StructureSetROISequence) + 1

            # P.3.C.8.8.5 Structure Set Module
            str_set_roi = pydicom.dataset.Dataset()
            str_set_roi.ROINumber = roi_number
            ref_for_seq = rt_ds.ReferencedFrameOfReferenceSequence[0]
            str_set_roi.ReferencedFrameOfReferenceUID = ref_for_seq.FrameOfReferenceUID
            str_set_roi.ROIName = roi_name
            str_set_roi.StructureSetDescription = ''
            str_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
            rt_ds.StructureSetROISequence.append(str_set_roi)

            # P.3.C.8.8.6 ROI Contour Module
            roi_contour_seq = pydicom.dataset.Dataset()
            roi_contour_seq.ROIDisplayColor = list(rgb_color)
            roi_contour_seq.ReferencedROINumber = roi_number
            roi_contour_seq.ContourSequence = pydicom.sequence.Sequence([])

            if isinstance(mask, xr.core.dataarray.DataArray):
                mask = mask.to_numpy()

            # For RTSTRURCTs, a contour sequence item is a unconnected 2D z-axis polygon
            non_zero_slices = np.nonzero(np.any(mask, axis=(0,1)))[0]
            for k_index in non_zero_slices:
                contour_sequences = self._mask_to_contour_sequence(mask, k_index, coord_converter)
                roi_contour_seq.ContourSequence.append(*contour_sequences)

            # Append entire sequence for the given contour
            rt_ds.ROIContourSequence.append(roi_contour_seq)

            # P.3.C.8.8.8 RT ROI Observation Module
            rt_roi_obs = pydicom.dataset.Dataset()
            # This might be different than roi_number...
            rt_roi_obs.ObservationNumber = roi_number
            rt_roi_obs.ReferencedROINumber = roi_number
            rt_roi_obs.ROIObservationDescription = 'Type:Soft, Range:*/*, Fill:0, Opacity:0.0, Thickness:1, LineThickness:2, read-only:false'
            rt_roi_obs.ROIObservationLabel = roi_name
            rt_roi_obs.RTROIInterpretedType = ''
            rt_ds.RTROIObservationsSequence.append(rt_roi_obs)
        return rt_ds

    def _duplicate_rtstruct(self, rtstruct, clear=False):
        # Combines emtpy and update header
        new_rt = copy.deepcopy(rtstruct)

        if clear:
            new_rt.StructureSetROISequence.clear()
            new_rt.ROIContourSequence.clear()
            new_rt.RTROIObservationsSequence.clear()

        date = datetime.now().date().strftime('%Y%m%d')
        time = datetime.now().time().strftime('%H%M%S.%f')[:-3]
        series_instance_uid = pydicom.uid.generate_uid(prefix=self.uid_prefix)
        sop_instance_uid = pydicom.uid.generate_uid(prefix=self.uid_prefix)

        # P.3.C.7.5.1 General Equipment Module
        new_rt.Manufacturer = 'DICOManager'
        new_rt.InstitutionName = 'DICOManager'
        new_rt.ManufacturerModelName = 'DICOManager'
        new_rt.SoftwareVersions = ['0.1.0']

        # P.3.C.8.8.1 RT Series Module
        new_rt.SeriesInstanceUID = series_instance_uid
        new_rt.SeriesDate = date
        new_rt.SeriesTime = time

        # P.3.C.8.8.5 Structure Set Module
        new_rt.StructureSetLabel = 'Auto-Segmented Contours'
        new_rt.StructureSetName = 'Auto-Segmented Contours'
        new_rt.StructureSetDate = date
        new_rt.StructureSetTime = time

        # P.3.C.12.1 SOP Common Module Attributes
        new_rt.SOPInstanceUID = sop_instance_uid
        new_rt.InstanceCreationDate = date
        new_rt.InstanceCreationTime = time

        # P.10.C.7.1 DICOM File Meta Information
        new_rt.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        return new_rt

    def _create_new_rtstruct(self, image_series_uid):
        # Read the ct to build the header from
        with contextlib.closing(dbu.make_connection(self.db_name, dicts=True)) as conn:
            with conn:
                query = f'SELECT * FROM dicoms WHERE SeriesInstanceUID = "{image_series_uid}"'
                image_series = conn.execute(query).fetchall()

        ct_dcm = pydicom.dcmread(image_series[0]['FilePath'], stop_before_pixels=True)

        # Start crafting a fresh RTSTRUCT
        rt_dcm = pydicom.dataset.Dataset()

        date = datetime.now().date().strftime('%Y%m%d')
        time = datetime.now().time().strftime('%H%M%S.%f')[:-3]
        sop_instance_uid = pydicom.uid.generate_uid(prefix=self.uid_prefix)
        series_instance_uid = pydicom.uid.generate_uid(prefix=self.uid_prefix)

        # P.3.C.7.1.1 Patient Module
        if hasattr(ct_dcm, 'PatientName'):
            rt_dcm.PatientName = ct_dcm.PatientName
        else:
            rt_dcm.PatientName = 'UNKNOWN^UNKNOWN^^'

        if hasattr(ct_dcm, 'PatientID'):
            rt_dcm.PatientID = ct_dcm.PatientID
        else:
            rt_dcm.PatientID = '0000000'

        if hasattr(ct_dcm, 'PatientBirthDate'):
            rt_dcm.PatientBirthDate = ct_dcm.PatientBirthDate
        else:
            rt_dcm.PatientBirthDate = ''

        if hasattr(ct_dcm, 'PatientSex'):
            rt_dcm.PatientSex = ct_dcm.PatientSex
        else:
            rt_dcm.PatientSex = ''

        if hasattr(ct_dcm, 'PatientAge'):
            rt_dcm.PatientAge = ct_dcm.PatientAge
        else:
            rt_dcm.PatientAge = ''

        # P.3.C.7.2.1 General Study Module
        rt_dcm.StudyInstanceUID = ct_dcm.StudyInstanceUID

        if hasattr(ct_dcm, 'StudyDate'):
            rt_dcm.StudyDate = ct_dcm.StudyDate
        else:
            rt_dcm.StudyDate = date

        if hasattr(ct_dcm, 'StudyTime'):
            rt_dcm.StudyTime = ct_dcm.StudyTime
        else:
            rt_dcm.StudyTime = time

        if hasattr(ct_dcm, 'StudyID'):
            rt_dcm.StudyID = ct_dcm.StudyID

        if hasattr(ct_dcm, 'StudyDescription'):
            rt_dcm.StudyDescription = ct_dcm.StudyDescription

        # P.3.C.7.5.1 General Equipment Module
        rt_dcm.Manufacturer = 'DICOManager'
        rt_dcm.InstitutionName = 'Beaumont Health'
        rt_dcm.ManufacturerModelName = 'DICOManager'
        rt_dcm.SoftwareVersions = ['0.1.0']

        # P.3.C.8.8.1 RT Series Module
        rt_dcm.Modality = 'RTSTRUCT'
        rt_dcm.SeriesInstanceUID = series_instance_uid

        if hasattr(ct_dcm, 'SeriesNumber'):
            rt_dcm.SeriesNumber = ct_dcm.SeriesNumber

        rt_dcm.SeriesDate = date
        rt_dcm.SeriesTime = time

        if hasattr(ct_dcm, 'SeriesDescription'):
            rt_dcm.SeriesDescription = ct_dcm.SeriesDescription

        # P.3.C.8.8.5 Structure Set Module
        rt_dcm.StructureSetLabel = 'DICOManager Created Contours'
        rt_dcm.StructureSetName = 'DICOManager Created Contours'
        rt_dcm.StructureSetDescription = 'DICOManager Created Contours'
        rt_dcm.StructureSetDate = date
        rt_dcm.StructureSetTime = time

        # Contour Image Sequence (3006, 0016)
        contour_image_sequence = []
        for row in image_series:
            contour_image_item = pydicom.dataset.Dataset()
            contour_image_item.ReferencedSOPClassUID = row['SOPClassUID']
            contour_image_item.ReferencedSOPInstanceUID = row['SOPInstanceUID']
            contour_image_sequence.append(contour_image_item)

        # RT Referenced Series Sequence (3006,0014)
        rt_ref_series_ds = pydicom.dataset.Dataset()
        rt_ref_series_ds.SeriesInstanceUID = ct_dcm.SeriesInstanceUID
        rt_ref_series_ds.ContourImageSequence = pydicom.sequence.Sequence(contour_image_sequence)

        # (3006,0014) attribute of (3006,0012)
        rt_ref_study_ds = pydicom.dataset.Dataset()
        rt_ref_study_ds.ReferencedSOPClassUID = ct_dcm.SOPClassUID
        rt_ref_study_ds.ReferencedSOPInstanceUID = ct_dcm.StudyInstanceUID
        rt_ref_study_ds.RTReferencedSeriesSequence = pydicom.sequence.Sequence([rt_ref_series_ds])

        # (3006,0012) attribute of (3006,0009)
        ref_frame_of_ref_ds = pydicom.dataset.Dataset()
        ref_frame_of_ref_ds.FrameOfReferenceUID = ct_dcm.FrameOfReferenceUID
        ref_frame_of_ref_ds.RTReferencedStudySequence = pydicom.sequence.Sequence([rt_ref_study_ds])

        # (3006,0009) attribute of (3006,0010)
        rt_dcm.ReferencedFrameOfReferenceSequence = pydicom.sequence.Sequence([ref_frame_of_ref_ds])

        # Structure Set Module (3006, 0020)
        rt_dcm.StructureSetROISequence = pydicom.sequence.Sequence()

        # P.3.C.8.8.6 ROI Contour Module
        rt_dcm.ROIContourSequence = pydicom.sequence.Sequence()

        # P.3.C.8.8.8 RT ROI Observation Module
        rt_dcm.RTROIObservationsSequence = pydicom.sequence.Sequence()

        # P.3.C.12.1 SOP Common Module Attributes
        rt_dcm.SOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.481.3')
        rt_dcm.SOPInstanceUID = sop_instance_uid
        rt_dcm.InstanceCreationDate = date
        rt_dcm.InstanceCreationTime = time

        # P.10.C.7.1 DICOM File Meta Information
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 222
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.481.3')
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.ImplementationClassUID = pydicom.uid.UID('1.2.276.0.7230010.3.0.3.6.2')
        file_meta.ImplementationVersionName = 'OFFIS_DCMTK_362'
        file_meta.SourceApplicationEntityTitle = 'RO_AE_MIM'

        # Make a pydicom.dataset.Dataset() into a pydicom.dataset.FileDataset()
        inputs = {'filename_or_obj': None,
                  'dataset': rt_dcm,
                  'file_meta': file_meta,
                  'preamble': b'\x00'*128}
        return pydicom.dataset.FileDataset(**inputs)

    def _check_rtstruct(self, rtstruct):
        if isinstance(rtstruct, str):
            try:
                rtstruct = pydicom.dcmread(rtstruct)
            except pydicom.errors.InvalidDicomError:
                rtstruct = xr.load_dataset(rtstruct)
            except FileNotFoundError:
                series_instance = rtstruct

        if isinstance(rtstruct, xr.core.dataset.Dataset):
            series_instance = rtstruct.attrs['SeriesInstanceUID']

        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                query = f'SELECT FilePath FROM dicoms WHERE SeriesInstanceUID = "{series_instance}"'
                filepath = conn.execute(query).fetchone()[0]
        try:
            rtstruct = pydicom.dcmread(filepath)
        except FileNotFoundError:
            self.log.error(f'Series {series_instance} not an RTSTRUCT, skipping')
            return None
        return rtstruct

    def _apply_fn_and_write(self, fn, *args, **kwargs):
        new_rt = fn(*args, **kwargs)
        new_row = self.ph.from_ds(new_rt)
        new_row['GeneratedStruct'] = True

        # Save new RTSTRUCT
        fs = dcmio.FileSaver(kwargs['save_dir'], kwargs['overwrite'], kwargs['prefixes'])
        filepath = fs.save_new_rtstruct(new_row, new_rt)
        new_row['FilePath'] = filepath

        # Add row
        with contextlib.closing(dbu.make_connection(self.db_name)) as conn:
            with conn:
                dbu.add_rows(conn, table='dicoms', rows=new_row)

    def new_from_image(self, image_series_uid, masks, roi_names=None, **kwargs):
        new_rt = self._create_new_rtstruct(image_series_uid)
        new_rt = self._append_masks(new_rt, image_series_uid, masks, roi_names)
        return new_rt

    def append_to_rtstruct(self, rtstruct, image_series_uid, masks, roi_names=None, **kwargs):
        rtstruct = self._check_rtstruct(rtstruct)
        if rtstruct is None:
            return None
        new_rt = self._duplicate_rtstruct(rtstruct)
        return self._append_masks(new_rt, image_series_uid, masks, roi_names)

    def new_from_rtstruct(self, rtstruct, image_series_uid, masks, roi_names=None, **kwargs):
        rtstruct = self._check_rtstruct(rtstruct)
        if rtstruct is None:
            return None
        new_rt = self._duplicate_rtstruct(rtstruct, clear=True)
        return self._append_masks(new_rt, image_series_uid, masks, roi_names)


def numpy_to_dicom(volume: np.ndarray, referenced_xarray: str,
                   series_description: str, study_description: str, save_dir: str) -> list:
    """Converts a numpy array to a DICOM file

    Args:
        volume (numpy.ndarray): Numpy volume with similar coordinates to the nc_file
        series_description (str): Series description to use in the numpy file
        save_dir (str): A POSIX path to the save location

    Returns:
        list: Returns a list of dicoms files saved to the save_dir
    """
    xrds = xr.load_dataset(referenced_xarray)
    x0 = xrds.coords['x'][0]
    y0 = xrds.coords['y'][0]

    series_uid = pydicom.uid.generate_uid('1.2.826.0.1.3680043.10.771.')
    date = datetime.now().date().strftime('%Y%m%d')

    files = []

    for z_slice, z_loc in enumerate(xrds.coords['z']):
        sop = pydicom.uid.generate_uid('1.2.826.0.1.3680043.10.771.')
        ds = pydicom.Dataset()
        ds.PatientID = xrds.attrs['PatientID']
        ds.FrameOfReferenceUID = xrds.attrs['FrameOfReferenceUID']
        ds.StudyInstanceUID = xrds.attrs['StudyInstanceUID']
        ds.SeriesInstanceUID = series_uid
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        ds.SliceThickness = xrds.attrs['VoxelSize'][2]
        ds.Rows = volume.shape[0]
        ds.Columns = volume.shape[1]
        ds.SOPInstanceUID = sop
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.StudyDescription = f'{study_description}'
        ds.SeriesDescription = f'{series_description}'
        ds.PatientName = xrds.attrs['PatientName'].replace(' ', '^')
        ds.Modality = 'MR'
        ds.InstanceNumber = z_slice + 1
        ds.AcquisitionDate = date
        ds.StudyDate = date
        ds.PixelSpacing = list(xrds.attrs['VoxelSize'][:2])
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.SliceLocation = z_loc
        ds.ImagePositionPatient = [x0, y0, z_loc]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PatientPosition = 'HFS'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15

        # P.10.C.7.1 DICOM File Meta Information
        file_meta = pydicom.dataset.FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 222
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.481.3')
        file_meta.MediaStorageSOPInstanceUID = sop
        file_meta.ImplementationClassUID = pydicom.uid.UID('1.2.276.0.7230010.3.0.3.6.2')
        file_meta.ImplementationVersionName = 'OFFIS_DCMTK_362'

        inputs = {'filename_or_obj': None,
                  'dataset': ds,
                  'file_meta': file_meta,
                  'preamble': b'\x00'*128}
        fullds = pydicom.dataset.FileDataset(**inputs)

        # Workaround for issue 1075
        fullds.PixelData = utils.to_16bit(volume[:, :, z_slice]).tobytes()
        fullds.is_implict_VR = True
        fullds['PixelData'].VR = 'OB'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filename = f'{save_dir}/{sop}.dcm'
        fullds.save_as(filename)
        files.append(filename)
    return files
