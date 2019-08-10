import numpy as np

def bin_coord(x, res=1):
    import numpy as np
    import pandas as pd

    x = np.array(x)
    x0 = np.floor(x.min())
    x1 = np.ceil(x.max()) + res

    xbin = np.arange(x0, x1, res)
    xlbl = np.c_[xbin[1:], xbin[:-1]].mean(1)

    xcut = pd.cut(x, xbin, labels=xlbl)

    return xcut


def bin_data(time, lat, lon, time_res='M', space_res=1):
    """
    Returns binned values of coordinates that can be used to grid data

    Parameters
    ----------
    time : np.datetime64
        times of measurements
    lat : np.ndarray
        latitudes in degrees N
    lon : np.ndarray
        latitudes in degrees E

    Returns
    -------
    time, lat, lon : np.ndarray
    """
    from pandas import Series

    assert np.issubdtype(time, np.datetime64), 'time is not np.datetime64 format'
    assert all((lat >= -90) & (lat <= 90)), 'lat is outside logical bounds'
    assert all((lon >= -180) & (lon <= 180)), 'lon is outside logical bounds'

    x = lon.values
    x[x >= 180] -= 360
    lon = x

    time = Series(time.values.astype('datetime64[M]'), name='time')
    lats = Series(bin_coord(lat, res=1).get_values(), name='lat')
    lons = Series(bin_coord(lon, res=1).get_values(), name='lon')

    return time, lats, lons


def insert_empty_coordinates(xds, res=1):
    """
    Pass a xr.Dataset or xr.DataArray through this function if there are
    missing values in the dimensions that are empty.

    e.g. Lat = [33, 35, 38, 39] --> [33, 34, 35, 36, 37, 38, 39]

    Parameters
    ----------
    xds : xr.DataArray / xr.Dataset
        must have time, lat, lon as dimensions and coordinates.
        time must be monthly and lat/lon must be 1deg
    res : float [1]
        the spatial resolution can be set to different resolutions.

    Returns
    -------
    xds : xr.Dataset / xr.DataArray
        a dataset with missing coordinates inserted.
    """

    import numpy as np

    t0 = xds.time.min().values.astype('datetime64[Y]').astype('datetime64[M]')
    t1 = xds.time.max().values.astype('datetime64[Y]') + np.timedelta64(1, 'Y')
    time = np.arange(t0, t1, 1, dtype='datetime64[M]')

    r = res
    x0 = -180 + r / 2
    x1 = 180

    y0 = np.floor(xds.lat.min()) - r / 2
    y1 = np.ceil(xds.lat.max()) + r / 2

    x = np.arange(x0, x1, r)
    y = np.arange(y0, y1, r)

    xds = xds.reindex(time=time, lon=x, lat=y)

    return xds


def grid_1deg_monthly(var, time, lat, lon):
    import pandas as pd
    import xarray as xr
    import numpy as np

    name = getattr(var, 'name', 'data')

    dt, dy, dx = bin_data(time, lat, lon)
    df = pd.Series(var).groupby(
        [dt.values, dy.values, dx.values]).agg(['mean', 'std'])
    df = df.loc[(slice('1982', '2017'), slice(None), slice(None)), :]
    df = df.dropna(subset=['mean'])

    df.index.levels[0].name = 'time'
    df.index.levels[1].name = 'lat'
    df.index.levels[2].name = 'lon'

    a = df.reset_index()
    ti = (a.time - np.datetime64('1982-01-01')).astype('timedelta64[M]')
    yi = a.lat.copy().astype(float)
    xi = a.lon.copy().astype(float)
    yi -= -89.5
    xi -= -179.5
    nt = int(ti.max()//12 + 1) * 12
    ti, yi, xi = [a.values.astype(int) for a in (ti, yi, xi)]

    grid_avg = np.ndarray([nt, 180, 360]) * np.nan
    grid_std = np.ndarray([nt, 180, 360]) * np.nan

    grid_avg[ti, yi, xi] = a['mean'].values
    grid_std[ti, yi, xi] = a['std'].values

    dt = np.arange('1982-01', '2019-01', 1, dtype='datetime64[M]')[:nt]
    dy = np.arange(-89.5, 90, 1)
    dx = np.arange(-179.5, 180, 1)

    xds = xr.Dataset()
    xds[name + '_mean'] = xr.DataArray(
        data=grid_avg,
        coords={'time': dt.astype('datetime64[ns]'), 'lat': dy, 'lon': dx},
        dims=['time', 'lat', 'lon'],
        encoding={'complevel': 4, 'zlib': True})
    xds[name + '_std'] = xr.DataArray(
        data=grid_std,
        coords={'time': dt.astype('datetime64[ns]'), 'lat': dy, 'lon': dx},
        dims=['time', 'lat', 'lon'],
        encoding={'complevel': 4, 'zlib': True})

    return xds
