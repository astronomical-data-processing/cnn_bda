"""Add functions in the new RASCIL version
The function name remains the same as in tag 0.1.11.
The calculation of index is now a separate function. Not stored in visibility class

Example：
compression： cvt = coalesce_visibility(vis, time_coal=1.0, max_time_coal=8, frequency_coal=1.0, max_frequency_coal=1.0)
decompression： cvis = decoalesce_visibility(cvt)

"""

__all__ = ['coalesce_visibility', 'decoalesce_visibility']

import numpy
import numba

from add_data_models import Visibility
from rascil.data_models.memory_data_models import BlockVisibility

import logging

log = logging.getLogger('logger')


def coalesce_visibility(vis: BlockVisibility, time_coal=0.0, frequency_coal=0.0,
                        max_time_coal=100, max_frequency_coal=100, self_corr_coal=1.0, **kwargs):
    """ Coalesce the BlockVisibility data_models

    :param vis: BlockVisibility to be coalesced
    :param time_coal: Number of times to coalesce
    :param frequency_coal: Number of frequencies to coalesce
    :param max_time_coal: Maximum number of integrations to coalesce
    :param max_frequency_coal: Maximum number of frequency channels to coalesce
    :param self_corr_coal: Number of self-correlated data to coalesce，Not used for now
    :param kwargs:
    :return:
    """
    log.debug('coalesce_visibility: new_rascil')
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis

    # if time_coal == 0.0 and frequency_coal == 0.0:
    #     return vis

    cvis, cflags, cwts, cimwt, cnvis, allpwtsgrid_bool, time_coal, max_time_coal, \
    frequency_coal, max_frequency_coal = average_in_blocks(vis["vis"].values, vis["flags"].values,
                                                           vis["weight"].values, vis["imaging_weight"].values,
                                                           vis.blockvisibility_acc.uvdist, time_coal, max_time_coal,
                                                           frequency_coal, max_frequency_coal, self_corr_coal)

    coalesced_vis = Visibility(vis=cvis, flags=cflags, weight=cwts, imaging_weight=cimwt,
                               cnvis=numpy.arange(cnvis), time=vis.time.data,
                               integration_time=vis.integration_time.data,
                               frequency=vis.frequency.data, channel_bandwidth=vis.channel_bandwidth.data,
                               source=vis.source, phasecentre=vis.phasecentre, configuration=vis.configuration,
                               uvw=vis.uvw.data, baselines=vis.baselines, meta=vis.meta,
                               polarisation_frame=vis.blockvisibility_acc.polarisation_frame,
                               allpwtsgrid_bool=allpwtsgrid_bool,
                               tf_coal=(time_coal, max_time_coal, frequency_coal, max_frequency_coal), )

    log.debug(
        'coalesce_visibility: Created new Visibility for coalesced data_models, coalescence factors (t,f) = (%.3f,%.3f)' %
        (time_coal, frequency_coal))
    log.debug('coalesce_visibility: Maximum coalescence (t,f) = (%d, %d)' % (max_time_coal, max_frequency_coal))
    log.debug('coalesce_visibility: Original %s, coalesced %s' % \
              (vis.blockvisibility_acc.datasizes(), coalesced_vis.visibility_acc.datasizes()))

    return coalesced_vis


def decoalesce_visibility(vis: Visibility, liner=False):
    """ Decoalesce the visibilities to the original values (opposite of coalesce_visibility)

    :param vis: Visibility to be decoalesced
    :return:
    """
    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis

    log.debug('decoalesce_visibility: new_rascil')

    ntimes, nbaseline, nchan, npol = vis.vis_orignal

    dvis = numpy.zeros([ntimes, nbaseline, nchan, npol], dtype='complex')
    dflags = numpy.zeros([ntimes, nbaseline, nchan, npol], dtype='int')
    dwts = numpy.zeros([ntimes, nbaseline, nchan, npol])
    dimwts = numpy.zeros([ntimes, nbaseline, nchan, npol])

    cindex = compute_cindex(allpwtsgrid_bool=vis["allpwtsgrid_bool"].values, uvdist=vis.visibility_acc.uvdist,
                            tf_coal=vis.tf_coal, nchan=nchan, npol=npol)[0]

    if liner:
        time_average, time_chunk_len = compute_cindex(allpwtsgrid_bool=vis["allpwtsgrid_bool"].values,
                                                        uvdist=vis.visibility_acc.uvdist,
                                                        tf_coal=vis.tf_coal, nchan=nchan, npol=npol)[4:]
        time_chunk_len = numpy.cumsum(time_chunk_len)
        time_chunk_len = numpy.insert(time_chunk_len, 0, 0)
        for bl in range(nbaseline):
            nrows = slice(time_chunk_len[bl], time_chunk_len[bl+1])
            dvis[:, bl, 0, 0] = recover_liner(vis["vis"].values[nrows, 0], ntimes, time_average[bl])

    else:
        dvis[...].flat = vis["vis"].values[cindex, :].flatten()

    dflags[...].flat = vis["flags"].values[cindex, :].flatten()
    dwts[...].flat = vis["weight"].values[cindex, :].flatten()
    dimwts[...].flat = vis["imaging_weight"].values[cindex, :].flatten()

    decomp_vis = BlockVisibility(vis=dvis, flags=dflags, weight=dwts, imaging_weight=dimwts,
                                 time=vis.time.data, integration_time=vis.integration_time.data,
                                 frequency=vis.frequency.data, channel_bandwidth=vis.channel_bandwidth.data,
                                 source=vis.source, phasecentre=vis.phasecentre, configuration=vis.configuration,
                                 uvw=vis.uvw.data, baselines=vis.baselines, meta=vis.meta,
                                 polarisation_frame=vis.visibility_acc.polarisation_frame, )

    log.debug('decoalesce_visibility: Coalesced %s, decoalesced %s' % (
        vis.blockvisibility_acc.datasizes(), decomp_vis.visibility_acc.datasizes()))

    return decomp_vis


@numba.jit(nopython=True)
def recover_liner(data_ave, ntimes, chunksize):
    rec_liner = numpy.zeros(ntimes, dtype=data_ave.dtype)

    chunk = (chunksize - 1) / 2
    power = numpy.linspace(-chunk, chunk, chunksize)

    start = 0
    for i in range(data_ave.shape[0]):
        if i < data_ave.shape[0] - 1:
            lenstep = 2 * ((chunksize + 1) // 2)
            step = (data_ave[i + 1] - data_ave[i]) / lenstep
        else:
            lis = ntimes - start
            chunk = (lis - 1) / 2
            power = numpy.linspace(-chunk, chunk, lis)

        rec_liner[start:start + chunksize] = data_ave[i] + step * power
        start += chunksize

    return rec_liner


def average_in_blocks(vis, flags, wts, imwts, uvdist, time_coal=1.0, max_time_coal=100,
                      frequency_coal=1.0, max_frequency_coal=100, self_corr_coal=None):
    """ Average visibility of blocks

    :param vis:
    :param flags:
    :param wts:
    :param imwts:
    :param uvdist:
    :param time_coal:
    :param max_time_coal:
    :param frequency_coal:
    :param max_frequency_coal:
    :param self_corr_coal:
    :return:
    """
    ntimes, nbaseline, nchan, npol = vis.shape

    assert wts.shape == flags.shape
    flagwts = wts
    flagwts[flags > 0] = 0.0
    allpwtsgrid_bool = numpy.einsum('ijkm -> j', flagwts)

    # Calculate the average length of time and frequency from the baseline
    tf_coal = (time_coal, max_time_coal, frequency_coal, max_frequency_coal)

    cindex, allpwtsgrid_bool, tf_coal, cnvis = \
        compute_cindex(allpwtsgrid_bool, uvdist, tf_coal, nchan, npol, self_corr_coal)[0:4]

    # Initial settings
    cvis = numpy.zeros([cnvis, npol], dtype='complex')
    cflags = numpy.zeros([cnvis, npol], dtype='int')
    cwts = numpy.zeros([cnvis, npol])
    cimwts = numpy.zeros([cnvis, npol])

    # Start calculation
    for pol in range(npol):
        wts_flat = wts[..., pol].flatten()

        def get_in_wts(arr):
            result = average_chunks_jit(arr.flatten(), wts_flat, cnvis, cindex)
            return result

        cwts[:, pol] = get_in_wts(wts[..., pol])
        cvis[:, pol] = get_in_wts(vis[..., pol])
        cimwts[:, pol] = get_in_wts(imwts[..., pol])

    cflags[cwts <= 0.0] = 1

    return cvis, cflags, cwts, cimwts, cnvis, allpwtsgrid_bool, time_coal, max_time_coal, frequency_coal, max_frequency_coal


@numba.jit(nopython=True)
def average_chunks_jit(arr, wts, fullsize, index):
    """ Average calculation with parameter data

    :param arr: Parameter data
    :param wts: Weights
    :param fullsize: Output data size
    :param index: cindex
    :return:
    """
    chunks = numpy.zeros(fullsize, dtype=arr.dtype)
    weights = numpy.zeros(fullsize, dtype=wts.dtype)

    for i in range(arr.shape[0]):
        poi = index[i]
        chunks[poi] += arr[i] * wts[i]
        weights[poi] += wts[i]

    chunks[weights > 0.0] = chunks[weights > 0.0] / weights[weights > 0.0]

    return chunks


def compute_cindex(allpwtsgrid_bool, uvdist, tf_coal, nchan, npol, self_corr_coal=None):
    """ get cindex

    :param allpwtsgrid_bool:
    :param uvdist:
    :param tf_coal: tuple: [time_coal, max_time_coal, frequency_coal, max_frequency_coal]
    :param nchan: Number of channels
    :param npol: Number of polarizations
    :param self_corr_coal:
    :return:
    """
    ntimes, nbaseline = uvdist.shape

    time_coal, max_time_coal, frequency_coal, max_frequency_coal = tf_coal

    time_average = numpy.ones([nbaseline], dtype='int')
    frequency_average = numpy.ones([nbaseline], dtype='int')

    uvmax = numpy.max(uvdist)
    uvdist_max = numpy.max(uvdist, axis=0)
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

    time_coal = numpy.min(time_average)
    max_time_coal = numpy.max(time_average)
    frequency_coal = numpy.min(frequency_average)
    max_frequency_coal = numpy.max(frequency_average)

    tf_coal = (time_coal, max_time_coal, frequency_coal, max_frequency_coal)

    # Calculate the size of the remaining data after averaging
    cnvis = 0
    time_chunk_len = numpy.ones([nbaseline], dtype='int')
    frequency_chunk_len = numpy.ones([nbaseline], dtype='int')
    for bl in range(nbaseline):
        if allpwtsgrid_bool[bl] > 0.0:
            time_chunk_len[bl] = (ntimes - 1) // time_average[bl] + 1
            frequency_chunk_len[bl] = (nchan - 1) // frequency_average[bl] + 1
        nrows = time_chunk_len[bl] * frequency_chunk_len[bl]
        cnvis += nrows

    cindex = numpy.zeros([ntimes, nbaseline, nchan, npol], dtype='int')

    visstart = 0
    for bl in range(nbaseline):
        nrows = time_chunk_len[bl] * frequency_chunk_len[bl]

        index_r = numpy.arange(visstart, visstart + nrows)
        index_r = index_r.reshape(time_chunk_len[bl], frequency_chunk_len[bl])
        index_r = index_r.repeat(frequency_average[bl], axis=1)
        index_r = index_r.repeat(time_average[bl], axis=0)

        cindex[:, bl, :].flat = index_r[0:ntimes, 0:nchan].flatten()
        visstart += nrows

    assert cnvis == visstart, "Mismatch between number of rows in coalesced visibility %d and index %d" % \
                              (cnvis, visstart)

    return cindex.flatten(), allpwtsgrid_bool, tf_coal, cnvis, time_average, time_chunk_len
