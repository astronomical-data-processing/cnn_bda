""" Average BlockVisibility in the original way(RASCIL version tag 0.1.11)

"""

__all__ = ['coalesce_visibility']

import logging

import numpy
from astropy import constants

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

    log.debug('coalesce_visibility: comp_original')
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    if time_coal == 0.0 and frequency_coal == 0.0:
        return convert_blockvisibility_to_visibility((vis))

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
    # Calculate the averaging factors for time and frequency making them the same for all times
    # for this baseline
    # Find the maximum possible baseline and then scale to this.

    # The input visibility is a block of shape [ntimes, nant, nant, nchan, npol]. We will map this
    # into rows like vis[npol] and with additional columns antenna1, antenna2, frequency

    log.debug('average_in_block: comp_original')
    ntimes, nant, _, nchan, npol = vis.shape

    times.dtype = numpy.float64

    # Original
    # Pol independent weighting
    # allpwtsgrid = numpy.sum(wts, axis=4)
    # # Pol and frequency independent weighting
    # allcpwtsgrid = numpy.sum(allpwtsgrid, axis=3)
    # # Pol and time independent weighting
    # alltpwtsgrid = numpy.sum(allpwtsgrid, axis=0)

    # Optimized
    assert wts.shape == flags.shape
    flagwts = wts
    flagwts[flags > 0] = 0.0
    allpwtsgrid = numpy.einsum('ijklm->ijkl', flagwts, optimize=True)
    # allcpwtsgrid = numpy.einsum('ijkl->ijk', allpwtsgrid, optimize=True)
    # alltpwtsgrid = numpy.einsum('ijkl->jkl', allpwtsgrid, optimize=True)

    # Now calculate on a baseline basis the time and frequency averaging. We do this by looking at
    # the maximum uv distance for all data and for a given baseline. The integration time and
    # channel bandwidth are scale appropriately.
    time_average = numpy.ones([nant, nant], dtype='int')
    frequency_average = numpy.ones([nant, nant], dtype='int')
    # ua = numpy.arange(nant)

    # Original
    # uvmax = numpy.sqrt(numpy.max(uvw[..., 0] ** 2 + uvw[..., 1] ** 2 + uvw[..., 2] ** 2))
    # for a2 in ua:
    #     for a1 in ua:
    #         if allpwtsgrid[:, a2, a1, :].any() > 0.0:
    #             uvdist = numpy.max(numpy.sqrt(uvw[:, a2, a1, 0] ** 2 + uvw[:, a2, a1, 1] ** 2), axis=0)
    #             if uvdist > 0.0:
    #                 time_average[a2, a1] = min(max_time_coal,
    #                                            max(1, int(round((time_coal * uvmax / uvdist)))))
    #                 frequency_average[a2, a1] = min(max_frequency_coal,
    #                                                 max(1, int(round(frequency_coal * uvmax / uvdist))))
    #             else:
    #                 time_average[a2, a1] = max_time_coal
    #                 frequency_average[a2, a1] = max_frequency_coal

    # Optimized
    # Calculate uvdist instead of uvwdist
    uvwd = uvw[..., 0:2]
    uvdist = numpy.einsum('ijkm,ijkm->ijk', uvwd, uvwd, optimize=True)
    uvmax = numpy.sqrt(numpy.max(uvdist))

    # uvdist = numpy.sqrt(numpy.einsum('ijkm,ijkm->ijk', uvw, uvw, optimize=True))
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

    # See how many time chunks and frequency we need for each baseline. To do this we use the same averaging that
    # we will use later for the actual data_models. This tells us the number of chunks required for each baseline.
    frequency_grid, time_grid = numpy.meshgrid(frequency, times)
    channel_bandwidth_grid, integration_time_grid = numpy.meshgrid(channel_bandwidth, integration_time)
    cnvis = 0
    time_chunk_len = numpy.ones([nant, nant], dtype='int')
    frequency_chunk_len = numpy.ones([nant, nant], dtype='int')

    time_len = len(times)
    frequency_len = len(frequency)

    # for a2 in ua:
    #     for a1 in ua:
    for a1 in range(nant):
        for a2 in range(a1 + 1, nant):
            if (time_average[a2, a1] > 0) & (frequency_average[a2, a1] > 0 & (allpwtsgrid[:, a2, a1, ...].any() > 0.0)):
                # time_chunks, _ = average_chunks(times, allcpwtsgrid[:, a2, a1], time_average[a2, a1])
                # time_chunk_len[a2, a1] = time_chunks.shape[0]
                time_chunk_len[a2, a1] = (time_len - 1) // time_average[a2, a1] + 1

                # frequency_chunks, _ = average_chunks(frequency, alltpwtsgrid[a2, a1, :], frequency_average[a2, a1])
                # frequency_chunk_len[a2, a1] = frequency_chunks.shape[0]
                frequency_chunk_len[a2, a1] = (frequency_len - 1) // frequency_average[a2, a1] + 1

                nrows = time_chunk_len[a2, a1] * frequency_chunk_len[a2, a1]
                cnvis += nrows

    # Now we know enough to define the output coalesced arrays. The output will be
    # successive a1, a2: [len_time_chunks[a2,a1], a2, a1, len_frequency_chunks[a2,a1]]
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

    # For decoalescence we keep an index to map back to the original BlockVisibility
    rowgrid = numpy.zeros([ntimes, nant, nant, nchan], dtype='int')
    rowgrid.flat = range(rowgrid.size)

    cindex = numpy.zeros([rowgrid.size], dtype='int')

    # Now go through, chunking up the various arrays. Everything is converted into an array with
    # axes [time, channel] and then it is averaged over time and frequency chunks for
    # this baseline.
    # To aid decoalescence we will need an index of which output elements a given input element
    # contributes to. This is a many to one. The decoalescence will then just consist of using
    # this index to extract the coalesced value that a given input element contributes towards.

    visstart = 0
    nrows = 0
    # for a2 in ua:
    #     for a1 in ua:
    for a1 in range(nant):
        for a2 in range(a1 + 1, nant):
            nrows = time_chunk_len[a2, a1] * frequency_chunk_len[a2, a1]
            rows = slice(visstart, visstart + nrows)

            # cindex.flat[rowgrid[:, a2, a1, :]] = numpy.array(range(visstart, visstart + nrows))
            index_r = numpy.arange(visstart, visstart + nrows)
            index_r = index_r.reshape(time_chunk_len[a2, a1], frequency_chunk_len[a2, a1])
            index_r = index_r.repeat(frequency_average[a2, a1], axis=1)
            index_r = index_r.repeat(time_average[a2, a1], axis=0)
            cindex.flat[rowgrid[:, a2, a1, :]] = index_r[0:ntimes, 0:nchan].flatten()

            ca1[rows] = a1
            ca2[rows] = a2

            # Average over time and frequency for case where polarisation isn't an issue
            def average_from_grid(arr):
                return average_chunks2(arr, allpwtsgrid[:, a2, a1, :],
                                       (time_average[a2, a1], frequency_average[a2, a1]))[0]

            ctime[rows] = average_from_grid(time_grid).flatten()
            cfrequency[rows] = average_from_grid(frequency_grid).flatten()

            for axis in range(3):
                uvwgrid = numpy.outer(uvw[:, a2, a1, axis], frequency / constants.c.value)
                cuvw[rows, axis] = average_from_grid(uvwgrid).flatten()

            # For some variables, we need the sum not the average
            def sum_from_grid(arr):
                result = average_chunks2(arr, allpwtsgrid[:, a2, a1, :],
                                         (time_average[a2, a1], frequency_average[a2, a1]))
                return result[0] * result[0].size

            cintegration_time[rows] = sum_from_grid(integration_time_grid).flatten()
            cchannel_bandwidth[rows] = sum_from_grid(channel_bandwidth_grid).flatten()

            # For the polarisations we have to perform the time-frequency average separately for each polarisation
            for pol in range(npol):
                result = average_chunks2(vis[:, a2, a1, :, pol], wts[:, a2, a1, :, pol],
                                         (time_average[a2, a1], frequency_average[a2, a1]))
                cvis[rows, pol], cwts[rows, pol] = result[0].flatten(), result[1].flatten()

            for pol in range(npol):
                result = average_chunks2(wts[:, a2, a1, :, pol], wts[:, a2, a1, :, pol],
                                         (time_average[a2, a1], frequency_average[a2, a1]))
                cwts[rows, pol] = result[0].flatten()
                cflags[rows, pol][cwts[rows, pol] <= 0.0] = 1

            for pol in range(npol):
                result = average_chunks2(imaging_wts[:, a2, a1, :, pol], wts[:, a2, a1, :, pol],
                                         (time_average[a2, a1], frequency_average[a2, a1]))
                cimwts[rows, pol] = result[0].flatten()

            visstart += nrows

    assert cnvis == visstart, "Mismatch between number of rows in coalesced visibility %d and index %d" % \
                              (cnvis, visstart)

    return cvis, cflags, cuvw, cwts, cimwts, ctime, cfrequency, cchannel_bandwidth, ca1, ca2, cintegration_time, cindex


def average_chunks2(arr, wts, chunksize):
    """ Average the two dimensional array arr with weights by chunks

    Array len does not have to be multiple of chunksize.

    :param arr: 2D array of values
    :param wts: 2D array of weights
    :param chunksize: 2-tuple of averaging region e.g. (2,3)
    :return: 2D array of averaged data_models, 2d array of weights
    """
    # Do each axis to determine length
    #    assert arr.shape == wts.shape, "Shapes of arrays must be the same"
    # It is possible that there is a dangling null axis on wts
    wts = wts.reshape(arr.shape)

    # Original
    # l0 = len(average_chunks(arr[:, 0], wts[:, 0], chunksize[0])[0])
    # l1 = len(average_chunks(arr[0, :], wts[0, :], chunksize[1])[0])

    # dqw
    l0 = (arr.shape[0] - 1) // chunksize[0] + 1
    l1 = (arr.shape[1] - 1) // chunksize[1] + 1

    tempchunks = numpy.zeros([arr.shape[0], l1], dtype=arr.dtype)
    tempwt = numpy.zeros([arr.shape[0], l1])

    tempchunks *= tempwt
    for i in range(arr.shape[0]):
        result = average_chunks(arr[i, :], wts[i, :], chunksize[1])
        tempchunks[i, :], tempwt[i, :] = result[0].flatten(), result[1].flatten()

    chunks = numpy.zeros([l0, l1], dtype=arr.dtype)
    weights = numpy.zeros([l0, l1])

    for i in range(l1):
        result = average_chunks(tempchunks[:, i], tempwt[:, i], chunksize[0])
        chunks[:, i], weights[:, i] = result[0].flatten(), result[1].flatten()

    return chunks, weights


def average_chunks(arr, wts, chunksize):
    """ Average the array arr with weights by chunks

    Array len does not have to be multiple of chunksize

    This version is optimised for plain numpy. It is roughly ten times faster that average_chunks_jit when used
    without numba jit. It cannot (yet) be used with numba because the add.reduceat is not support in numba
    0.31

    :param arr: 1D array of values
    :param wts: 1D array of weights
    :param chunksize: averaging size
    :return: 1D array of averaged data_models, 1d array of weights
    """
    if chunksize <= 1:
        return arr, wts

    # Original codes
    # places = range(0, len(arr), chunksize)
    # chunks = numpy.add.reduceat(wts * arr, places)
    # weights = numpy.add.reduceat(wts, places)
    # chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]

    # Codes optimized

    mask = numpy.zeros(((len(arr) - 1) // chunksize + 1, arr.shape[0]), dtype=bool)
    for enumerate_id, i in enumerate(range(0, len(arr), chunksize)):
        mask[enumerate_id, i:i + chunksize] = 1
    chunks = mask.dot(wts * arr)
    weights = mask.dot(wts)
    # chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]
    numpy.putmask(chunks, weights > 0.0, chunks / weights)

    return chunks, weights
