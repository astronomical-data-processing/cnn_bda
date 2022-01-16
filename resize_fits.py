"""Resize M31 images with skimage
According to the characteristics of M31.model.bits (the information is only in the middle part),
the expansion and preservation is only for this part. That is, if the advice cellsize for the array is 16k,
use the result of expanding to 8k (corresponding to the file name m31_16.fit)

"""

from astropy.io import fits
from skimage import transform


def extend_k(n=8):
    hdulist = fits.open('M31.model.fits')
    data = hdulist[0].data
    header = hdulist[0].header
    hdulist.close()

    npixel = 512 * n
    large_data = transform.resize(data[129:385, 129:385], (npixel, npixel))
    large_data = large_data.reshape(1, 1, npixel, npixel)
    large_data = large_data.astype('float64')

    header['CRPIX1'] = npixel / 2
    header['CRPIX2'] = npixel / 2
    header['NAXIS1'] = npixel
    header['NAXIS2'] = npixel

    fits.writeto(filename='m31_%d.fits' % n, data=large_data, header=header,
                 overwrite=True)


if __name__ == '__main__':
    import numpy

    li = numpy.array([4, 8, 16, 32])
    for i in li:
        extend_k(i)
