import numpy as np
import SimpleITK as sitk


def n4_bias(image_group, nbins=200):
    for i in range(image_group.volume.shape[-1]):
        sitk_image = sitk.GetImageFromArray(image_group.volume[..., i].astype('float32'))
        sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1, nbins)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output = corrector.Execute(sitk_image, sitk_mask)
        image_group.volume[..., i] = sitk.GetArrayFromImage(output).astype('int16')
    return image_group


def suv_bw(image_group):
    ds = image_group.ds
    rescaled = image_group.volume * ds.RescaleSlope + ds.RescaleIntercept
    patient_weight = 1000 * float(ds.PatientWeight)
    radiopharm = ds.RadiopharmaceuticalInformationSequence[0]
    total_dose = float(radiopharm.RadionuclideTotalDose)
    suv_values = rescaled * patient_weight / total_dose
    image_group.volume = suv_values
    image_group.units = 'SUVbw'
    return image_group


def counts(image_group):
    try:
        map_seq = image_group.ds.RealWorldValueMappingSequence[0]
        slope = float(map_seq.RealWorldValueSlope)
        intercept = float(map_seq.RealWorldValueIntercept)
        count_vol = np.array(image_group.volume * slope + intercept, dtype='float32')
        image_group.volume = count_vol
        image_group.units = 'Counts'
        return image_group
    except Exception:
        return image_group


def padding(image_group):
    # Handles different encoding types
    pad = getattr(image_group.ds, 'PixelPaddingValue', None)
    limit = getattr(image_group.ds, 'PixelPaddingRangeLimit', pad)
    pad_value = getattr(image_group.ds, 'SmallestPixelValue', 0)
    if pad is not None:
        if image_group.ds.PixelRepresentation:
            pad = pad - (1 << image_group.ds.BitsStored)
            limit = limit - (1 << image_group.ds.BitsStored)
        image_group.volume[(image_group.volume >= pad) & (image_group.volume <= limit)] = pad_value
    return image_group


def hounsfield(image_group):
    # Rescales to HU values
    ds = image_group.ds
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept
    hu_array = np.array(image_group.volume * slope + intercept, dtype='int16')
    hu_array[hu_array < -1000] = -1000
    image_group.volume = hu_array
    image_group.units = 'Hounsfield'
    return image_group


def interpolate(raw_array, raw_dims, reffed_img, interpolator=sitk.sitkLinear):
    coordinate_system = reffed_img['CoordinateSystem'].astype('float64')
    volume_dimensions = reffed_img['VolumeDimensions'].astype('int32')
    voxel_size = reffed_img['VoxelSize'].astype('float64')
    direction = coordinate_system[:3, :3] / voxel_size

    # SITK Dose Image
    # ITK assumes [z, y, x] coordinates, rt dose saves as z, y, x
    raw_sitk = sitk.GetImageFromArray(raw_array)

    raw_sitk.SetDirection(raw_dims.direction.astype('float64'))
    raw_sitk.SetSpacing(raw_dims.voxel_size)
    raw_sitk.SetOrigin(raw_dims.origin.astype('float64'))

    # Configuration Interpolator
    interp = sitk.ResampleImageFilter()
    interp.SetOutputDirection(direction.T.flatten())
    interp.SetOutputSpacing(voxel_size)
    interp.SetSize(volume_dimensions.tolist())
    interp.SetOutputPixelType(sitk.sitkFloat32)
    interp.SetOutputOrigin(coordinate_system[:3, -1])
    interp.SetInterpolator(interpolator)

    # Interpolate
    new_sitk = interp.Execute(raw_sitk)
    new_arr = sitk.GetArrayFromImage(new_sitk).T
    return new_arr
