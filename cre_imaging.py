"""Modifications based on create_test_image()

"""

__all__ = ['create_test_image']

import logging

import numpy

from rascil.data_models.memory_data_models import Image

from rascil.processing_components.image.operations import create_image_from_array
from rascil.processing_components.image.operations import import_image_from_fits

from rascil.processing_components.util.installation_checks import check_data_directory
from rascil.processing_components.simulation.testing_support import replicate_image

check_data_directory()
log = logging.getLogger("rascil-logger")


def create_test_image(
        cellsize=None,
        frequency=None,
        channel_bandwidth=None,
        phasecentre=None,
        polarisation_frame=None,
        fits_id=None,
        image_path=None,
) -> Image:
    """Create a useful test image

    This is the test image M31 widely used in ALMA and other simulations. It is actually part of an Halpha region in
    M31.

    :param cellsize:
    :param frequency: Frequency (array) in Hz
    :param channel_bandwidth: Channel bandwidth (array) in Hz
    :param phasecentre: Phase centre of image (SkyCoord)
    :param polarisation_frame: Polarisation frame
    :param fits_id: Advice cellsize, like 8, 16, 32, corresponds to the result of resize_fits
    :param image_path: Optional, path to the resize_fits result
    :return: Image
    """
    log.debug('dqw_create_image')
    check_data_directory()

    # Rewrite by dqw
    if image_path is None:
        image_path = "m31_%d.fits" % fits_id
    im = import_image_from_fits(image_path)

    if frequency is None:
        frequency = [1e8]
    if polarisation_frame is None:
        polarisation_frame = im.image_acc.polarisation_frame
    im = replicate_image(im, frequency=frequency, polarisation_frame=polarisation_frame)

    wcs = im.image_acc.wcs.deepcopy()

    if cellsize is not None:
        wcs.wcs.cdelt[0] = -180.0 * cellsize / numpy.pi
        wcs.wcs.cdelt[1] = +180.0 * cellsize / numpy.pi
    if frequency is not None:
        wcs.wcs.crval[3] = frequency[0]
    if channel_bandwidth is not None:
        wcs.wcs.cdelt[3] = channel_bandwidth[0]
    else:
        if len(frequency) > 1:
            wcs.wcs.cdelt[3] = frequency[1] - frequency[0]
        else:
            wcs.wcs.cdelt[3] = 0.001 * frequency[0]
    wcs.wcs.radesys = "ICRS"
    wcs.wcs.equinox = 2000.00

    if phasecentre is None:
        phasecentre = im.image_acc.phasecentre
    else:
        wcs.wcs.crval[0] = phasecentre.ra.deg
        wcs.wcs.crval[1] = phasecentre.dec.deg
        # WCS is 1 relative
        wcs.wcs.crpix[0] = im["pixels"].data.shape[3] // 2 + 1
        wcs.wcs.crpix[1] = im["pixels"].data.shape[2] // 2 + 1

    return create_image_from_array(
        im["pixels"].data, wcs=wcs, polarisation_frame=polarisation_frame
    )
