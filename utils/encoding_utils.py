import numpy as np
import sys; sys.path.append("./")


def decode_date(encoded_date):
    doy_sin,doy_cos = encoded_date
    doy = np.arctan2((2*doy_sin-1),(2*doy_cos-1))*365/(2*np.pi)
    if doy<1:
        doy+=365
    return np.array([np.round(doy)])


def decode_coordinates(encoded_coords):
    '''
    Encoded coordinates must be of shape (n_coords, 3)
    Output is lat-long coordinates of shape (n_coords, 2)
    '''

    lat_enc,long_sin,long_cos = encoded_coords.T
    lat = -lat_enc*180+90
    long = np.arctan2((2*long_sin-1),(2*long_cos-1))*360/(2*np.pi)
    return np.array([lat,long]).T

def encode_coordinates(coords):
    '''
    coordinates must be in lat-long of shape (n_coords, 2)
    Output is encoded coordinates of shape (n_coords, 3)
    '''
    lat,long = coords
    lat = (-lat + 90)/180
    long_sin = (np.sin(long*2*np.pi/360)+1)/2
    long_cos = (np.cos(long*2*np.pi/360)+1)/2

    return np.array([lat,long_sin,long_cos], dtype=np.float32)