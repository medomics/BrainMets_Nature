import nibabel
import scipy
import numpy as np
import csv
import os
import sphericity


def compute_volume(ds):
    volume = ds.get_fdata()
    voxel_size = ds.header['pixdim'][:3]
    voxel_volume = np.product(voxel_size)
    return np.sum(volume) * voxel_volume


def unnest(nested):
    out = []
    for n in nested:
        if n is None:
            continue
        elif not len(n):
            continue
        elif isinstance(n[0], list):
            out.extend(unnest(n))
        else:
            out.extend([n])
    return out


def pre_compute_gw_distance(grey_white_niftis):
    roi_nii = {}
    for nii in grey_white_niftis:
        roi = os.path.basename(nii).replace('.nii.gz', '')
        roi_nii[roi] = nii

    grey_ds = nibabel.load(roi_nii['GreyMatter'])
    white_ds = nibabel.load(roi_nii['WhiteMatter'])
    brain_ds = nibabel.load(roi_nii['BrainBSCer'])

    grey_vol = np.array(grey_ds.get_fdata(), dtype='bool')
    white_vol = np.array(white_ds.get_fdata(), dtype='bool')
    brain_vol = np.array(brain_ds.get_fdata(), dtype='bool')

    voxel_size = grey_ds.header['pixdim'][:3]
    grey_mask = np.ma.masked_where(brain_vol == 0, grey_vol * brain_vol)
    white_mask = np.ma.masked_where(brain_vol == 0, (white_vol * np.invert(grey_vol)) * brain_vol)

    grey_edt = scipy.ndimage.distance_transform_edt(np.invert(grey_mask), sampling=voxel_size)
    grey_distance = np.ma.masked_where(brain_vol == 0, grey_edt)
    white_edt = scipy.ndimage.distance_transform_edt(np.invert(white_mask), sampling=voxel_size)
    white_distance = np.ma.masked_where(brain_vol == 0, white_edt)

    interface_ventricles = scipy.ndimage.binary_dilation(grey_mask) ^ grey_mask
    interface = np.invert(interface_ventricles * (white_mask + grey_mask))
    interface_distance = np.ma.masked_where(brain_vol == 0, scipy.ndimage.distance_transform_edt(interface, sampling=voxel_size))

    grey_distance.dump('../helper_data/grey_edt.npy')
    white_distance.dump('../helper_data/white_edt.npy')
    interface_distance.dump('../helper_data/interface_edt.npy')
    np.save('../helper_data/gw_brain_mask', brain_vol)


def pre_compute_masks(nii_files, expansions, atlas):
    names = []
    masks = []

    for nii_file in nii_files:
        ds = nibabel.load(nii_file)
        roi = os.path.basename(nii_file).replace('.nii.gz', '')
        names.append(roi)
        masks.append(ds.get_fdata())

    masks = np.array(masks)
    masks_edt = np.zeros_like(masks)

    for i, mask in enumerate(masks):
        masks_edt[i] = scipy.ndimage.distance_transform_edt(np.ones_like(mask) - mask, sampling=ds.header['pixdim'][:3])

    for d in expansions:
        expanded_roi = np.zeros_like(masks)
        for i, m_edt in enumerate(masks_edt):
            expanded_roi[i] = np.copy(m_edt) <= d
        np.savez_compressed(f'../helper_data/masks_{atlas}_{d}mm.npz', expanded_roi.astype(bool))

    with open(f'../helper_data/names_{atlas}.csv', 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for name in names:
            csvwriter.writerow([name])

    return names, masks


class FindOverlap:
    def __init__(self, columns, atlases, expansions):
        super().__init__()
        self.columns = columns
        self.atlases = atlases
        self.expansions = expansions

    def _load_names(self):
        self.atlases_names = {}
        for atlas in self.atlases:
            atlas_names = []
            with open(f'../helper_data/names_{atlas}.csv', 'r') as f:
                for row in f:
                    atlas_names.append(row.replace('\n', ''))
            self.atlases_names[atlas] = atlas_names

    def _load_masks(self):
        self.atlases_masks = {}

        for atlas in self.atlases:
            atlas_masks = {}
            for distance in self.expansions:
                atlas_masks[distance] = np.load(f'../helper_data/masks_{atlas}_{distance}mm.npz')['arr_0']
            self.atlases_masks[atlas] = atlas_masks

    def _load_gw_distances(self):
        self.interface_distance = np.load('../helper_data/interface_edt.npy', allow_pickle=True)
        self.brain_mask = np.load('../helper_data/gw_brain_mask.npy')

    def _clear_loaded(self):
        self.atlases_names = None
        self.atlases_masks = None
        self.interface_distance = None
        self.brain_mask = None

    # Unnest, put expansions in the class init
    def run(self, volume_row):
        rows = []

        self._load_gw_distances()
        self._load_masks()

        for roi, filepath in zip(volume_row['NiftiROIs'], volume_row['NiftiFilePaths']):
            ds = nibabel.load(filepath)

            # Computes volume of the GTV
            volume = compute_volume(ds)
            sphericity_value = sphericity.extract(nifti=ds,
                                                  voxel_dim=[1, 1, 1],
                                                  roi_interp='linear',
                                                  roi_pv=0.5)
            sx, sy, sz = ds.get_fdata().shape

            for distance in self.expansions:
                for atlas in self.atlases:
                    masks_atlas = self.atlases_masks[atlas][distance]

                    # Computes overlap with masks
                    if not volume:
                        roi_overlap_atlas = np.zeros((masks_atlas.shape[0])).astype(bool)
                        centroid = None
                        interface_centroid = None
                        interface_roi = None
                    else:
                        vol = ds.get_fdata().astype(bool)
                        com_x, com_y, com_z = scipy.ndimage.center_of_mass(vol)
                        ix = np.round(max(min(com_x, sx-1), 0)).astype(int)
                        iy = np.round(max(min(com_y, sy-1), 0)).astype(int)
                        iz = np.round(max(min(com_z, sz-1), 0)).astype(int)

                        roi_overlap_atlas = masks_atlas[:, ix, iy, iz]

                        centroid = [ix, iy, iz]

                        if distance == 0:
                            # Computes minimum surface separation and centroid distance to grey matter
                            roi_rind = vol ^ scipy.ndimage.binary_erosion(vol)

                            to_exclude = ['skull', 'align', 'skin', 'body', 'outer', 'external', 'surface', 'rind']
                            exclude = any([x in roi.lower() for x in to_exclude])

                            interface_centroid = self.interface_distance[ix, iy, iz]
                            if type(interface_centroid) is np.ma.core.MaskedConstant or exclude:
                                interface_centroid = None

                            interface_roi = None
                            if not np.sum(np.invert(self.brain_mask) * roi_rind):
                                interface_roi = np.min(self.interface_distance * roi_rind)
                        else:
                            interface_centroid = None
                            interface_roi = None

                    row = {}
                    for key, value in volume_row.items():
                        if key in self.columns:
                            row[key] = value

                    row['NiftiROI'] = roi
                    row['NiftiFilePath'] = filepath
                    row['Volume'] = volume
                    row['Sphericity'] = sphericity_value
                    row['AtlasExpansion'] = distance
                    row['Centroid'] = centroid
                    row['GMtoCentroid'] = interface_centroid
                    row['GMtoSurface'] = interface_roi
                    row['AtlasUsed'] = atlas
                    row['AtlasOverlap'] = roi_overlap_atlas
                    row['UniqueName'] = f'{distance}mm_of_{filepath}_for_{atlas}'

                    reordered = {k: row[k] for k in self.columns}

                    rows.append(reordered)

        self._clear_loaded()
        return rows


class FindOverlapShared:
    def __init__(self, columns, atlas_shapes, interface_shape, brain_shape):
        self.columns = columns
        self.atlas_shapes = atlas_shapes
        self.interface_shape = interface_shape
        self.brain_shape = brain_shape

    # Unnest, put expansions in the class init
    def run(self, volume_row):
        rows = []

        atlases = {}
        for atlas, shape in self.atlas_shapes.items():
            atlases[atlas] = np.memmap(f'../helper_data/{atlas}.array', dtype='uint8', mode='r', shape=shape)

        interface_distance = np.memmap('../helper_data/distance.array', dtype='float32', mode='r', shape=self.interface_shape)
        brain_mask = np.memmap('../helper_data/brain.array', dtype='uint8', mode='r', shape=self.brain_shape)

        for roi, filepath in zip(volume_row['NiftiROIs'], volume_row['NiftiFilePaths']):
            ds = nibabel.load(filepath)

            # Computes volume of the GTV
            volume = compute_volume(ds)
            sphericity_value = sphericity.extract(nifti=ds,
                                                  voxel_dim=[1, 1, 1],
                                                  roi_interp='linear',
                                                  roi_pv=0.5)
            sx, sy, sz = ds.get_fdata().shape
            vol = ds.get_fdata().astype(bool)
            com_x, com_y, com_z = scipy.ndimage.center_of_mass(vol)
            ix = np.round(max(min(com_x, sx-1), 0)).astype(int)
            iy = np.round(max(min(com_y, sy-1), 0)).astype(int)
            iz = np.round(max(min(com_z, sz-1), 0)).astype(int)
            centroid = [ix, iy, iz]

            for atlas_name, atlas_masks in atlases.items():
                overlap = atlas_masks[:, ix, iy, iz]
                atlas, distance = atlas_name.split('-')

                if not volume:
                    overlap = np.zeros(atlas_masks.shape[0]).astype(bool)
                    centroid = None
                    interface_centroid = None
                    interface_roi = None
                else:
                    if distance == 0:
                        # Computes minimum surface separation and centroid distance to grey matter
                        roi_rind = vol ^ scipy.ndimage.binary_erosion(vol)

                        to_exclude = ['skull', 'align', 'skin', 'body', 'outer', 'external', 'surface', 'rind']
                        exclude = any([x in roi.lower() for x in to_exclude])

                        interface_centroid = interface_distance[ix, iy, iz]
                        if type(interface_centroid) is np.ma.core.MaskedConstant or exclude:
                            interface_centroid = None

                        interface_roi = None
                        if not np.sum(np.invert(brain_mask) * roi_rind):
                            interface_roi = np.min(interface_distance * roi_rind)
                    else:
                        interface_centroid = None
                        interface_roi = None

                row = {}
                for key, value in volume_row.items():
                    if key in self.columns:
                        row[key] = value

                row['NiftiROI'] = roi
                row['NiftiFilePath'] = filepath
                row['Volume'] = volume
                row['Sphericity'] = sphericity_value
                row['AtlasExpansion'] = distance
                row['Centroid'] = centroid
                row['GMtoCentroid'] = interface_centroid
                row['GMtoSurface'] = interface_roi
                row['AtlasUsed'] = atlas
                row['AtlasOverlap'] = np.array(overlap, dtype=bool)
                row['UniqueName'] = f'{distance}mm_of_{filepath}_for_{atlas}'

                reordered = {k: row[k] for k in self.columns}

                rows.append(reordered)

        atlases = None
        brain_mask = None
        interface_distance = None

        return rows
