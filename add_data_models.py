""" Create Visibility class to save the compressed data, refer to Block class
Relative to block, compress the four parameters (vis, weight, imaging_weight, flags)
and leave the other quantities unchanged (while adding some parameters).

The cindex is not stored directly in the Vis class,
but is also recalculated by other reserved parameters (allpwtsgrid_bool, tf_coal, vis_original).

"""

__all__ = ['Visibility']

import numpy
import xarray
from astropy import constants as const
from astropy.time import Time

from rascil.data_models.polarisation import (PolarisationFrame, )


class XarrayAccessorMixin:
    """Convenience methods to access the fields of the xarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def size(self):
        """Return size in GB"""
        size = self._obj.nbytes
        return size / 1024.0 / 1024.0 / 1024.0

    def datasizes(self):
        """Return string describing sizes of data variables
        :return: string
        """
        s = "Dataset size: {:.3f} GB\n".format(self._obj.nbytes / 1024 / 1024 / 1024)
        for var in self._obj.data_vars:
            s += "\t[{}]: \t{:.3f} GB\n".format(
                var, self._obj[var].nbytes / 1024 / 1024 / 1024
            )
        return s


class Visibility(xarray.Dataset):
    """Visibility xarray Dataset class

    Visibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre.

    Polarisation frame is the same for the entire data set and can be stokesI, circular, circularnp, linear, linearnp

    The configuration is stored as an attribute.

    Here is an example::

        <xarray.Visibility>
        Dimensions:            (baselines: 6670, frequency: 3, polarisation: 4, time: 3, uvw_index: 3, cnvis: ***)
        Coordinates:
          * time               (time) float64 5.085e+09 5.085e+09 5.085e+09
          * baselines          (baselines) MultiIndex
          - antenna1           (baselines) int64 0 0 0 0 0 0 ... 112 112 112 113 113 114
          - antenna2           (baselines) int64 0 1 2 3 4 5 ... 112 113 114 113 114 114
          * frequency          (frequency) float64 1e+08 1.05e+08 1.1e+08
          * polarisation       (polarisation) <U2 'XX' 'XY' 'YX' 'YY'
          * uvw_index          (uvw_index) <U1 'u' 'v' 'w'
          * cnvis              (cnvis) int64 0 1 2 3 4 5 .....
        Data variables:
            integration_time   (time) float32 99.72697 99.72697 99.72697
            datetime           (time) datetime64[ns] 2000-01-01T03:54:07.843184299 .....
            vis                (cnvis, polarisation) complex128 ...
            weight             (cnvis, polarisation) float32 0.0...
            imaging_weight     (cnvis, polarisation) float32 0.0...
            flags              (cnvis, polarisation) float32 0.0...
            allpwtsgrid_bool   (baselines) float32 0.0...
            uvw                (time, baselines, uvw_index) float64 0.0 0.0 ... 0.0 0.0
            uvw_lambda         (time, baselines, frequency, uvw_index) float64 0.0 .....
            uvdist_lambda      (time, baselines, frequency) float64 0.0 0.0 ... 0.0 0.0
            channel_bandwidth  (frequency) float64 1e+07 1e+07 1e+07
        Attributes:
            phasecentre:         <SkyCoord (ICRS): (ra, dec) in deg    (180., -35.)>
            configuration:       <xarray.Configuration>Dimensions:   (id: 115, spat...
            polarisation_frame:  linear
            source:              unknown
            meta:                None
            tf_coal:             (time_coal, max_time_coal, frequency_coal, max_frequency_coal)
            vis_orginal:         (time, baselines, frequency, polarisation)


    """

    __slots__ = ()

    def __init__(
            self,
            frequency=None,
            channel_bandwidth=None,
            phasecentre=None,
            configuration=None,
            uvw=None,
            time=None,
            vis=None,
            weight=None,
            integration_time=None,
            flags=None,
            cnvis=None,
            baselines=None,
            polarisation_frame=PolarisationFrame("stokesI"),
            imaging_weight=None,
            source="anonymous",
            meta=None,
            low_precision="float32",
            allpwtsgrid_bool=None,
            tf_coal=None,
    ):
        """Visibility

        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        :param uvw: UVW coordinates (m) [:, nant, nant, 3]
        :param time: Time (UTC) [:]
        :param baselines: List of baselines
        :param flags: Flags [:, nant, nant, nchan]
        :param weight: [:, nant, nant, nchan, npol]
        :param imaging_weight: [:, nant, nant, nchan, npol]
        :param integration_time: Integration time [:]
        :param polarisation_frame: Polarisation_Frame e.g. Polarisation_Frame("linear")
        :param source: Source name
        :param meta: Meta info
        """

        super().__init__()

        if weight is None:
            weight = numpy.ones(vis.shape)
        else:
            assert weight.shape == vis.shape

        if imaging_weight is None:
            imaging_weight = weight
        else:
            assert imaging_weight.shape == vis.shape

        if integration_time is None:
            integration_time = numpy.ones_like(time)
        else:
            assert len(integration_time) == len(time)

        k = (frequency / const.c).value
        if len(frequency) == 1:
            uvw_lambda = (uvw * k)[..., numpy.newaxis, :]
        else:
            uvw_lambda = numpy.einsum("tbs,k->tbks", uvw, k)

        ntimes = len(time)
        nbaseline = len(baselines)
        nchan = len(frequency)
        npol = polarisation_frame.npol
        # Define the names of the dimensions

        coords = {
            "time": time,
            "baselines": baselines,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "spatial": ["u", "v", "w"],
            "cnvis": cnvis,
        }

        datavars = dict()
        datavars["integration_time"] = xarray.DataArray(
            integration_time.astype(low_precision), dims=["time"], attrs={"units": "s"}
        )
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims=["time"],
            attrs={"units": "s"},
        )

        datavars["vis"] = xarray.DataArray(
            vis,
            dims=["cnvis", "polarisation"],
            attrs={"units": "Jy"},
        )
        datavars["weight"] = xarray.DataArray(
            weight.astype(low_precision),
            dims=["cnvis", "polarisation"],
        )
        datavars["imaging_weight"] = xarray.DataArray(
            imaging_weight.astype(low_precision),
            dims=["cnvis", "polarisation"],
        )
        datavars["flags"] = xarray.DataArray(
            flags.astype(low_precision),
            dims=["cnvis", "polarisation"],
        )
        datavars["allpwtsgrid_bool"] = xarray.DataArray(
            allpwtsgrid_bool.astype(low_precision),
            dims=["baselines"],
        )

        datavars["uvw"] = xarray.DataArray(
            uvw, dims=["time", "baselines", "spatial"], attrs={"units": "m"}
        )

        datavars["uvw_lambda"] = xarray.DataArray(
            uvw_lambda,
            dims=["time", "baselines", "frequency", "spatial"],
            attrs={"units": "lambda"},
        )
        datavars["uvdist_lambda"] = xarray.DataArray(
            numpy.hypot(uvw_lambda[..., 0], uvw_lambda[..., 1]),
            dims=["time", "baselines", "frequency"],
            attrs={"units": "lambda"},
        )

        datavars["channel_bandwidth"] = xarray.DataArray(
            channel_bandwidth, dims=["frequency"], attrs={"units": "Hz"}
        )

        attrs = dict()
        attrs["rascil_data_model"] = "Visibility"
        attrs["configuration"] = configuration  # Antenna/station configuration
        attrs["source"] = source
        attrs["phasecentre"] = phasecentre
        attrs["_polarisation_frame"] = polarisation_frame.type
        attrs["meta"] = meta
        attrs["tf_coal"] = tf_coal
        attrs["vis_orignal"] = (ntimes, nbaseline, nchan, npol)

        super().__init__(datavars, coords=coords, attrs=attrs)


@xarray.register_dataset_accessor("visibility_acc")
class VisibilityAccessor(XarrayAccessorMixin):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def rows(self):
        """Rows"""
        return range(len(self._obj.time))

    @property
    def ntimes(self):
        """Number of times (i.e. rows) in this table"""
        return len(self._obj["time"])

    @property
    def nchan(self):
        """Number of channels"""
        return len(self._obj["frequency"])

    @property
    def npol(self):
        """Number of polarisations"""
        return len(self._obj.polarisation)

    @property
    def polarisation_frame(self):
        """Polarisation frame (from coords)

        :return:
        """
        return PolarisationFrame(self._obj.attrs["_polarisation_frame"])

    @property
    def nants(self):
        """Number of antennas"""
        return self._obj.configuration.configuration_acc.nants

    @property
    def nbaselines(self):
        """Number of Baselines"""
        return len(self._obj["baselines"])

    @property
    def u(self):
        """u coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 0]

    @property
    def v(self):
        """v coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 1]

    @property
    def w(self):
        """w coordinate (metres) [nrows, nbaseline]"""
        return self._obj["uvw"][..., 2]

    @property
    def u_lambda(self):
        """u coordinate (wavelengths) [nrows, nbaseline]"""
        return self._obj["uvw_lambda"][..., 0]

    @property
    def v_lambda(self):
        """v coordinate (wavelengths) [nrows, nbaseline]"""
        return self._obj["uvw_lambda"][..., 1]

    @property
    def w_lambda(self):
        """w coordinate (wavelengths) [nrows, nbaseline]"""
        return self._obj["uvw_lambda"][..., 2]

    @property
    def uvdist(self):
        """uv distance (metres) [nrows, nbaseline]"""
        return numpy.hypot(self.u, self.v)

    @property
    def uvdist_lambda(self):
        """uv distance (metres) [nrows, nbaseline]"""
        return numpy.hypot(self.u_lambda, self.v_lambda)

    @property
    def uvwdist(self):
        """uv distance (metres) [nrows, nbaseline]"""
        return numpy.hypot(self.u, self.v, self.w)

    @property
    def flagged_vis(self):
        """Flagged complex visibility [nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["vis"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_weight(self):
        """Weight [: npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["weight"].data * (1 - self._obj["flags"].data)

    @property
    def flagged_imaging_weight(self):
        """Flagged Imaging_weight[nrows, nbaseline, nchan, npol]

        Note that a numpy or dask array is returned, not an xarray dataarray
        """
        return self._obj["imaging_weight"].data * (1 - self._obj["flags"].data)

    @property
    def nvis(self):
        """Number of visibilities (in total)"""
        return numpy.product(self._obj.vis.shape)
