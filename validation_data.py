import numpy as np
import pandas as pd
import xarray as xr
from glob import glob


def unzip(filename):
    import zipfile
    import os

    new_dir = os.path.splitext(filename)[0]
    try:
        os.mkdir(new_dir)
    except FileExistsError as e:
        pass

    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(new_dir)

    return new_dir


def download_file(url, prefix='./', overwrite=True):
    """
    Downloads and saves any url to a file at the specified prefix.

    Paramters
    ---------
    url : str
        location of the file you would like to download
    prefix : str
        the path where you would like to download the file to.

    Returns
    -------
    save_path : str
        The path where the file is saved to, else None if unsuccessful
    """
    from tqdm import tqdm
    import requests
    import math
    import os

    save_path = os.path.join(prefix, url.split('/')[-1])

    if os.path.isfile(save_path) and (not overwrite):
        print("'{}' already exists. Rename or delete local file to download again".format(
            save_path))
        return save_path

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    block_size = 2**20
    total_size = int(r.headers.get('content-length', 0))
    total_scaled = math.ceil(total_size // block_size)
    wrote = 0

    with open(save_path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), desc=url, total=total_scaled, unit='MB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
        return save_path

    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        return None


def carioca_processing(working_dir='/Users/luke/Desktop/'):
    """
    Downloads, concatenates and grids CARIOCA data downloaded from PANGEA
    """
    from binning_utils import bin_data, insert_empty_coordinates

    def carioca_download(prefix='./'):
        import os

        data_source = [  # files manually found
            'http://doi.pangaea.de/10.1594/PANGAEA.814995',
            'http://doi.pangaea.de/10.1594/PANGAEA.814996',
            'http://doi.pangaea.de/10.1594/PANGAEA.728786',
            'http://doi.pangaea.de/10.1594/PANGAEA.728787',
            'http://doi.pangaea.de/10.1594/PANGAEA.728788',
            'http://doi.pangaea.de/10.1594/PANGAEA.728789',
            'http://doi.pangaea.de/10.1594/PANGAEA.728790',
            'http://doi.pangaea.de/10.1594/PANGAEA.728791',
            'http://doi.pangaea.de/10.1594/PANGAEA.728792',
            'http://doi.pangaea.de/10.1594/PANGAEA.728793',
            'http://doi.pangaea.de/10.1594/PANGAEA.813637',
            'http://doi.pangaea.de/10.1594/PANGAEA.728794',
        ]

        flist = []
        for url in data_source:
            url += "?format=textfile"

            fname = download_file(url, prefix)
            sname = fname.replace('?format=textfile', '.txt')
            os.rename(fname, sname)
            flist += sname,

        return flist

    def carioca_concat_txt(filelist):
        """
        Reads in CARIOCA text files and concatenates them into a pandas.DataFrame

        Pramaeters
        ----------
        filelist : list
            a list of paths to carioca files as downloaded from PANGEA

        Returns
        -------
        data : pd.DataFrame
            concatenated carioca data
        meta : pd.DataFrame
            metadata information from the header of each file.
        """
        import re

        data, meta = [], {}
        for sname in filelist:
            ID = sname.split('.')[-2]

            # Look for the end of the header
            fobj = open(sname)
            header_lines = int(np.where([line.startswith('*/') for line in fobj])[0]) + 1

            # read in the file with the header lines skipped
            df = pd.read_csv(sname,
                            sep='\t',
                            skiprows=header_lines,
                            parse_dates=True,
                            index_col=0)
            df['Float'] = ID  # create a new column for float name
            data += df,

            # reset the file object and read until header_lines and make metadata
            fobj.seek(0)
            meta[ID] = ''.join([fobj.readline() for i in range(header_lines)])

        meta = pd.Series(meta)
        data = pd.concat(data, ignore_index=False, sort=False)

        pattern = '[^A-Za-z0-9_]+'
        cols = [re.sub(pattern, '', c) for c in data.columns]
        data.columns = cols

        return data, meta

    flist = carioca_download(working_dir)

    # meta also returned but we never use it so set it to _
    dat, _ = carioca_concat_txt(flist)

    dat = dat[dat.Latitude < -35]

    # create indicies on which to bin the data
    # bin data groups data monthly by 1deg
    time, lat, lon = bin_data(dat.index, dat.Latitude, dat.Longitude)

    print('VARIABLES IN CARIOCA: ', dat.columns.values.tolist())
    key = input('Specify the variable you want to grid: ')

    xda = (dat[key]
          .groupby([time.values, lat.values, lon.values])
          .agg(['mean', 'std'])
          .to_xarray()
          .rename(
              {'level_0': 'time', 'level_1': 'lat', 'level_2': 'lon',
               'mean': '{}_mean'.format(key), 'std': '{}_std'.format(key),
              })
          )

    xda = insert_empty_coordinates(xda)

    return xda


def GLODAPv2_processing(working_dir='/Users/luke/Desktop', max_depth=15):

    def glodap_download(working_dir):
        url = "https://www.glodap.info/glodap_files/v2.2019/GLODAPv2.2019_Merged_Master_File.csv.zip"

        fname_zipped = download_file(url, working_dir, overwrite=False)
        fname_folder = unzip(fname_zipped)

        return fname_folder

    def glodap_read_txt(fname):
        def parse_dates(*args):
            from datetime import datetime
            args = [float(d) for d in args]
            string = '{:.0f}-{:.0f}-{:.0f}'.format(*args)
            format = '%Y-%m-%d'
            date = datetime.strptime(string, format)
            return date

        fobj = open(fname)
        for l, line in enumerate(fobj):
            iscomment = line.startswith('//')
            if not iscomment:
                break

        df = pd.read_csv(
            filepath_or_buffer=fname, sep=',',  # nrows=50000,
            parse_dates={'date': ['year', 'month', 'day']},
            date_parser=parse_dates, na_values=-9999,
        )

        return df

    def calc_pco2_from_dic_alk(dic, alk, pres, temp, salt):
        from cbsyst import CBsys
        dic, alk, pres, temp, salt = [
            np.array(a) for a in [dic, alk, pres, temp, salt]]

        m = np.any([np.isnan(a) for a in [dic, alk, pres, temp, salt]], axis=0)

        i = (pres < 15) & ~m

        out = CBsys(
            P_in=pres[i],
            DIC=dic[i],
            TA=alk[i],
            T_in=temp[i],
            S_in=salt[i],
            P_out=0.01,
        )

        pco2 = dic * np.nan
        pco2[i] = out['pCO2']

        return pco2

    from binning_utils import grid_1deg_monthly

    folder_name = glodap_download(working_dir)

    file_name = '{}/GLODAPv2.2019_Merged_Master_File.csv'.format(folder_name)

    df = glodap_read_txt(file_name)

    df = df[df.depth <= max_depth]

    df['pco2'] = calc_pco2_from_dic_alk(
        df.tco2, df.talk, df.pressure, df.temperature, df.salinity)

    xds = grid_1deg_monthly(df.pco2, df.date, df.latitude, df.longitude)

    return xds
