""" Average BlockVisibility using numba (RASCIL version tag 0.1.11)

"""

__all__ = ['coalesce_visibility', 'decoalesce_visibility']

import logging

import numpy
import pandas as pd
from astropy import constants
import numba

from rascil.data_models.memory_data_models import Visibility, BlockVisibility
from rascil.processing_components.util.array_functions import average_chunks, average_chunks2
from rascil.processing_components.visibility.base import vis_summary, copy_visibility
from rascil.processing_components.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger('logger')


def coalesce_visibility(vis: BlockVisibility, time_coal=0.0, frequency_coal=0.0, max_time_coal=100,
                        max_frequency_coal=100, **kwargs) -> Visibility:
    """ Coalesce the BlockVisibility data_models. The output format is a Visibility, as needed for imaging

    Coalesce by baseline-dependent averaging (optional). The number of integrations averaged goes as the ratio of the
    maximum possible baseline length to that for this baseline. This number can be scaled by coalescence_factor and
    limited by max_coalescence.

    When faceting, the coalescence factors should be roughly the same as the number of facets on one axis.

    If coalescence_factor=0.0 then just a format conversion is done

    :param vis: BlockVisibility to be coalesced
    :param time_coal: Number of times to coalesce
    :param frequency_coal: Number of frequencies to coalesce
    :param max_time_coal: Maximum number of integrations to coalesce
    :param max_frequency_coal: Maximum number of frequency channels to coalesce
    :return: Coalesced visibility with  cindex and blockvis filled in
    """

    log.debug('coalesce_visibility: comp_numba')
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    if time_coal == 0.0 and frequency_coal == 0.0:
        return convert_blockvisibility_to_visibility((vis))

    # if time_coal == 0.0:
    #     max_time_coal = 1.0
    # if frequency_coal == 0.0:
    #     max_frequency_coal = 1.0
    # if max_time_coal == 1.0 and max_frequency_coal == 1.0:
    #     return convert_blockvisibility_to_visibility((vis))

    cvis, cflags, cuvw, cwts, cimwt, ctime, cfrequency, cchannel_bandwidth, ca1, ca2, cintegration_time, cindex \
        = average_in_blocks(vis.data['vis'], vis.data['flags'], vis.data['uvw'], vis.data['weight'],
                            vis.data['imaging_weight'],
                            vis.time, vis.integration_time,
                            vis.frequency, vis.channel_bandwidth, time_coal, max_time_coal,
                            frequency_coal, max_frequency_coal)
    coalesced_vis = Visibility(uvw=cuvw, flags=cflags, time=ctime, frequency=cfrequency,
                               channel_bandwidth=cchannel_bandwidth,
                               phasecentre=vis.phasecentre, antenna1=ca1, antenna2=ca2, vis=cvis,
                               weight=cwts, imaging_weight=cimwt,
                               configuration=vis.configuration, integration_time=cintegration_time,
                               polarisation_frame=vis.polarisation_frame, cindex=cindex,
                               blockvis=vis, meta=vis.meta)

    log.debug(
        'coalesce_visibility: Created new Visibility for coalesced data_models, coalescence factors (t,f) = (%.3f,%.3f)'
        % (time_coal, frequency_coal))
    log.debug('coalesce_visibility: Maximum coalescence (t,f) = (%d, %d)' % (max_time_coal, max_frequency_coal))
    log.debug('coalesce_visibility: Original %s, coalesced %s' % (vis_summary(vis),
                                                                  vis_summary(coalesced_vis)))

    return coalesced_vis


def average_in_blocks(vis, flags, uvw, wts, imaging_wts, times, integration_time, frequency, channel_bandwidth,
                      time_coal=1.0, max_time_coal=100, frequency_coal=1.0, max_frequency_coal=100):
    """ Average visibility in blocks

    :param vis:
    :param flags:
    :param uvw:
    :param wts:
    :param imaging_wts:
    :param times:
    :param integration_time:
    :param frequency:
    :param channel_bandwidth:
    :param time_coal:
    :param max_time_coal:
    :param frequency_coal:
    :param max_frequency_coal:
    :return:
    """

    log.debug('average_in_block: comp_numba')

    ntimes, nant, _, nchan, npol = vis.shape
    times.dtype = numpy.float64

    assert wts.shape == flags.shape
    flagwts = wts
    flagwts[flags > 0] = 0.0
    allpwtsgrid = numpy.einsum('ijklm->ijkl', flagwts, optimize=True)

    # Calculate the average length of time and frequency from the baseline
    time_average = numpy.ones([nant, nant], dtype='int')
    frequency_average = numpy.ones([nant, nant], dtype='int')

    uvwd = uvw[..., 0:2]
    uvdist = numpy.einsum('ijkm,ijkm->ijk', uvwd, uvwd, optimize=True)
    uvmax = numpy.sqrt(numpy.max(uvdist))
    uvdist_max = numpy.sqrt(numpy.max(uvdist, axis=0))

    allpwtsgrid_bool = numpy.einsum('ijklm->jk', flagwts, optimize=True)
    mask = numpy.where(uvdist_max > 0.)
    mask0 = numpy.where(uvdist_max <= 0.)

    time_average[mask] = numpy.round((time_coal * uvmax / uvdist_max[mask]))
    time_average.dtype = numpy.int64
    time_average[mask0] = max_time_coal
    numpy.putmask(time_average, allpwtsgrid_bool == 0, 0)
    numpy.putmask(time_average, time_average < 1, 1)
    numpy.putmask(time_average, time_average > max_time_coal, max_time_coal)
    frequency_average[mask] = numpy.round((frequency_coal * uvmax / uvdist_max[mask]))

    frequency_average.dtype = numpy.int64
    frequency_average[mask0] = max_frequency_coal
    numpy.putmask(frequency_average, allpwtsgrid_bool == 0, 0)
    numpy.putmask(frequency_average, frequency_average < 1, 1)
    numpy.putmask(frequency_average, frequency_average > max_frequency_coal, max_frequency_coal)

    numpy.putmask(time_average, time_average > ntimes, ntimes)
    numpy.putmask(frequency_average, frequency_average > nchan, nchan)

    # Calculate the size of the remaining data after averaging
    cnvis = 0
    time_chunk_len = numpy.ones([nant, nant], dtype='int')
    frequency_chunk_len = numpy.ones([nant, nant], dtype='int')
    for a1 in range(nant):
        for a2 in range(a1 + 1, nant):
            if (time_average[a2, a1] > 0) & (frequency_average[a2, a1] > 0) & (allpwtsgrid[:, a2, a1, ...].any() > 0.0):
                time_chunk_len[a2, a1] = (ntimes - 1) // time_average[a2, a1] + 1
                frequency_chunk_len[a2, a1] = (nchan - 1) // frequency_average[a2, a1] + 1
                nrows = time_chunk_len[a2, a1] * frequency_chunk_len[a2, a1]
                cnvis += nrows

    frequency_grid, time_grid = numpy.meshgrid(frequency, times)
    channel_bandwidth_grid, integration_time_grid = numpy.meshgrid(channel_bandwidth, integration_time)

    # Define the output parameters
    ctime = numpy.zeros([cnvis])
    cfrequency = numpy.zeros([cnvis])
    cchannel_bandwidth = numpy.zeros([cnvis])
    cvis = numpy.zeros([cnvis, npol], dtype='complex')
    cflags = numpy.zeros([cnvis, npol], dtype='int')
    cwts = numpy.zeros([cnvis, npol])
    cimwts = numpy.zeros([cnvis, npol])
    cuvw = numpy.zeros([cnvis, 3])
    ca1 = numpy.zeros([cnvis], dtype='int')
    ca2 = numpy.zeros([cnvis], dtype='int')
    cintegration_time = numpy.zeros([cnvis])

    uvw_grid = numpy.zeros([ntimes, nant, nant, nchan, 3])
    cindex = numpy.zeros([ntimes, nant, nant, nchan, 1], dtype='int')

    # Preparation
    visstart = 0
    nrows = 0
    for a2 in range(1, nant):
        for a1 in range(a2):
            nrows = time_chunk_len[a2, a1] * frequency_chunk_len[a2, a1]
            rows = slice(visstart, visstart + nrows)
            ca1[rows] = a1
            ca2[rows] = a2

            m = numpy.arange(visstart, visstart + nrows)
            m = m.reshape((time_chunk_len[a2, a1], frequency_chunk_len[a2, a1]))
            m = m.repeat(frequency_average[a2, a1], axis=1)
            m = m.repeat(time_average[a2, a1], axis=0)
            cindex[:, a2, a1, :, 0].flat = m[0:ntimes, 0:nchan].flatten()

            uvw_grid[:, a2, a1, :, 0] = numpy.outer(uvw[:, a2, a1, 0], frequency / constants.c.value)
            uvw_grid[:, a2, a1, :, 1] = numpy.outer(uvw[:, a2, a1, 1], frequency / constants.c.value)
            uvw_grid[:, a2, a1, :, 2] = numpy.outer(uvw[:, a2, a1, 2], frequency / constants.c.value)

            visstart += nrows

    mask = numpy.tri(nant, nant, -1, dtype=bool)
    cindex_flat = cindex[:, mask, :, 0].flatten()
    allpwtsgrid = numpy.einsum('tijf->ijtf', allpwtsgrid)
    allpwtsgrid_flat = allpwtsgrid[mask, :, :].flatten()

    # Repeat subsequent gird data with baseline
    baseline = nant * (nant - 1) // 2
    time_grid = numpy.tile(time_grid.flatten(), baseline)
    frequency_grid = numpy.tile(frequency_grid.flatten(), baseline)
    integration_time_grid = numpy.tile(integration_time_grid.flatten(), baseline)
    channel_bandwidth_grid = numpy.tile(channel_bandwidth_grid.flatten(), baseline)

    def get_in_allwts(arr):
        result = average_chunks_jit(arr.flatten(), allpwtsgrid_flat, cnvis, cindex_flat)
        return result[0]

    ctime = get_in_allwts(time_grid)
    cfrequency = get_in_allwts(frequency_grid)
    cintegration_time = get_in_allwts(integration_time_grid)
    cchannel_bandwidth = get_in_allwts(channel_bandwidth_grid)

    for axis in range(3):
        cuvw[:, axis] = get_in_allwts(uvw_grid[:, mask, :, axis])

    for pol in range(npol):
        wts_flat = wts[:, mask, :, pol].flatten()

        def get_in_wts(arr):
            result = average_chunks_jit(arr.flatten(), wts_flat, cnvis, cindex_flat)
            return result[0]

        cwts[:, pol] = get_in_wts(wts[:, mask, :, pol])
        cvis[:, pol] = get_in_wts(vis[:, mask, :, pol])
        cimwts[:, pol] = get_in_wts(imaging_wts[:, mask, :, pol])

    cflags[cwts <= 0.0] = 1

    assert cnvis == visstart, "Mismatch between number of rows in coalesced visibility %d and index %d" % \
                              (cnvis, visstart)

    return cvis, cflags, cuvw, cwts, cimwts, ctime, cfrequency, cchannel_bandwidth, ca1, ca2, cintegration_time, cindex.flatten()


@numba.jit(nopython=True)
def average_chunks_jit(arr, wts, fullsize, index):
    chunks = numpy.zeros(fullsize, dtype=arr.dtype)
    weights = numpy.zeros(fullsize, dtype=wts.dtype)

    for i in range(arr.shape[0]):
        poi = index[i]
        chunks[poi] += arr[i] * wts[i]
        weights[poi] += wts[i]

    chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]

    return chunks, weights


def decoalesce_visibility(vis: Visibility, **kwargs) -> BlockVisibility:
    """ Decoalesce the visibilities to the original values (opposite of coalesce_visibility)

    This relies upon the block vis and the index being part of the vis. Needs the index generated by coalesce_visibility

    :param vis: (Coalesced visibility)
    :return: BlockVisibility with vis and weight columns overwritten
    """

    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis
    assert isinstance(vis.blockvis, BlockVisibility), "No blockvisibility in vis %r" % vis
    assert vis.cindex is not None, "No reverse index in Visibility %r" % vis

    log.debug('decoalesce_visibility: Created new Visibility for decoalesced data_models')
    decomp_vis = copy_visibility(vis.blockvis)

    vshape = decomp_vis.data['vis'].shape

    dvis = numpy.zeros(vshape, dtype='complex')
    assert numpy.max(vis.cindex) < dvis.size
    assert numpy.max(vis.cindex) < vis.vis.shape[0], "Incorrect template used in decoalescing"

    ntimes, nant, _, nchan, npol = decomp_vis.vis.shape
    nvis = npol * ntimes * nchan * nant * (nant - 1) // 2

    log.debug('decoalesce_visibility: tri-size of block and size of vis: %d, %d' % (nvis, vis.data['vis'].size))

    # This approach is slow when the cindex is large
    decomp_vis.data['vis'].flat = vis.data['vis'][vis.cindex, :]
    decomp_vis.data['flags'].flat = vis.data['flags'][vis.cindex, :]
    decomp_vis.data['weight'].flat = vis.data['weight'][vis.cindex, :]
    decomp_vis.data['imaging_weight'].flat = vis.data['imaging_weight'][vis.cindex, :]

    log.debug('decoalesce_visibility: Coalesced %s, decoalesced %s' % (vis_summary(vis),
                                                                       vis_summary(
                                                                           decomp_vis)))
    return decomp_vis
