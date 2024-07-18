import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
from cartopy.io.shapereader import Reader
import argparse


shapefile = Reader(
    '/home/santimtco/Downloads/Departamentos/departamento_small.shp').geometries()


def process_band(band, meta, crop=False):
    res = band.shape

    # x0 = int(res[1] / 2)
    # y0 = int(res[0] * 3 / 4)
    # xf = int(res[1] / 8) # same
    # yf = int(res[0] / 8) # 4 for suthamerica 8 for cordoba

    # little fire
    x0 = int(res[1] * 9 / 16)
    y0 = int(res[0] * 3 / 4)
    xf = int(res[1] / 16)
    yf = int(res[0] / 16)

    hight = meta['goes_imager_projection'].perspective_point_height
    xmin = meta['x'][:].min() * hight
    xmax = meta['x'][:].max() * hight
    ymin = meta['y'][:].min() * hight
    ymax = meta['y'][:].max() * hight

    # Calculate scale factors
    scale_x = (xmax - xmin) / res[1]
    scale_y = (ymax - ymin) / res[0]

    if crop:
        img = band[:].data[y0:y0+yf, x0:x0+xf]
        cropped_xmin = xmin + x0 * scale_x
        cropped_xmax = xmin + (x0 + xf) * scale_x
        cropped_ymin = ymax - (y0 + yf) * scale_y
        cropped_ymax = ymax - y0 * scale_y
        zoom_img_extent = (cropped_xmin, cropped_xmax,
                           cropped_ymin, cropped_ymax)
        return img, zoom_img_extent, meta

    img = band[:].data
    extent = (xmin, xmax, ymin, ymax)
    return img, extent, meta

def array_from_band(filename, data='CMI', crop=False):
    img_obj = Dataset(filename, 'r')
    meta = img_obj.variables

    return process_band(meta[data], meta, crop)

def array_of_arrays_from_band(filename, bands=[7,6,5], crop=False):
    img_obj = Dataset(filename, 'r')
    meta = img_obj.variables

    arrays = [img_obj.variables['CMI_C{:02d}'.format(band)] for band in bands]
    return [process_band(arr, meta, crop) for arr in arrays]


def calibrate(img, meta):
    ichanel = int(meta['band_id'][:] or 0)
    if ichanel >= 7:
        fk1 = meta['planck_fk1'][0]
        fk2 = meta['planck_fk2'][0]
        bc1 = meta['planck_bc1'][0]
        bc2 = meta['planck_bc2'][0]
        img_cal = (fk2 / (np.log((fk1 / img) + 1)) - bc1) / bc2-273.15
        return img_cal
    slope = meta['kappa0'].data
    return img * slope


def normalize(value, lower_limit, upper_limit, clip=True):
    norm = (value - lower_limit) / (upper_limit - lower_limit)
    if clip:
        norm = np.clip(norm, 0, 1)
    return norm


def get_band_file(images_list, band_number):
    for img in images_list:
        if f'C{band_number:02}_G16' in img:
            return img
    return None


def rebin(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def fire_detection_composite(R, G, B, meta):
    RGB = np.dstack((normalize(R-273.15, 0, 60)**(1/0.4),
                     normalize(G, 0, 1), normalize(rebin(B, R.shape), 0, 0.75)))
    lon_cen = meta['goes_imager_projection'].longitude_of_projection_origin
    height = meta['goes_imager_projection'].perspective_point_height
    return RGB, lon_cen, height


def expanded_filter(filter): return np.expand_dims(filter, -1)


def apply_mask(image):
    def fire_mask(vector): return vector[:, :, 0] > 0.8

    def fire_medium(vector): return (
        vector[:, :, 1] > 0.8) & (vector[:, :, 1] > 0.8)

    def fire_top(vector): return (vector[:, :, 2] > 0.8) & (
        vector[:, :, 1] > 0.8) & (vector[:, :, 0] > 0.8)

    def cloud(vector): return (vector[:, :, 2] > vector[:, :, 0]) & (
        vector[:, :, 1] > vector[:, :, 0]) & ~fire_mask(vector)

    def water(vector): return (vector[:, :, 2] < 0.15) & (
        vector[:, :, 1] < 0.15) & (vector[:, :, 0] < 0.15) & ~cloud(vector)

    def soil(vector): return ~fire_mask(vector) & ~fire_medium(
        vector) & ~fire_top(vector) & ~water(vector) & ~cloud(vector)

    expanded = expanded_filter(fire_mask(image))
    image = image * ~expanded_filter(soil(image)) + \
        expanded_filter(soil(image)) * np.array([0, .3, 0])
    image = image * ~(expanded |
                      expanded_filter(fire_medium(image)) |
                      expanded_filter(fire_top(image)) |
                      expanded_filter(cloud(image)) |
                      expanded_filter(water(image))) + \
        expanded * np.array([1, 0, 0]) + \
        expanded_filter(fire_medium(image)) * np.array([1, 1, 0]) + \
        expanded_filter(fire_top(image)) * np.array(
            [1, 1, 1]) + expanded_filter(cloud(image)) * np.array([.3, .3, .3]) + \
        expanded_filter(water(image)) * np.array([0, 0, .3])

    return image


def apply_plot(RGB, show, lon_cen, height, extent):
    plt.figure(dpi=100, frameon=False)
    crs = ccrs.Geostationary(central_longitude=lon_cen,
                             satellite_height=height)
    ax = plt.axes(projection=crs)
    ax.gridlines()
    ax.add_feature(ccrs.cartopy.feature.BORDERS, linewidth=0.5, color='green')
    ax.coastlines(resolution='10m', color='blue')
    ax.add_geometries(shapefile, ccrs.PlateCarree(),
                      edgecolor='white', facecolor='none', linewidth=0.2)
    img = plt.imshow(RGB, extent=extent)
    # plt.title(f'Fire Temperature - {images_list[7-1][:-3]}')
    if show:
        plt.show()
    else:
        plt.savefig(fname=export_dir + "MASK-P" +
                    dir.replace("/", "-") + ".png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a fire composite image from GOES16 CMI data',
                                     epilog='Example: python3 final.py -i ~/data/noaa-goes16/ABI-L2-CMIPF/2020/267/15/ -o ~/facu/ProcesamientoImagenes/seq_fire/')

    parser.add_argument('-i', '--input', type=str,
                        help='Input directory \nExample: ~/data/noaa-goes16/ABI-L2-CMIPF/2020/267/15/', required=True)
    parser.add_argument('-o', '--output', type=str,
                        help='Output directory \nExample: ~/facu/ProcesamientoImagenes/seq_fire', required=False, default="./")
    parser.add_argument('-p', '--plot', type=bool,
                        help='Plot the image', required=False, default=True)
    parser.add_argument('-c', '--crop', type=bool,
                        help='Crop the image', required=False, default=True)
    parser.add_argument('-m', '--mask', type=bool,
                        help='Apply mask', required=False, default=False)
    parser.add_argument('-s', '--show-plot', type=bool,
                        help='Shows matplotlib insted of saving', required=False, default=False)

    dir = parser.parse_args().input
    mask = parser.parse_args().mask
    plot = parser.parse_args().plot
    crop = parser.parse_args().crop
    export_dir = parser.parse_args().output

    images_list = os.listdir(dir)
    images_list.sort()

    R, extent, meta = array_from_band(
        dir + get_band_file(images_list, 7), data='CMI', crop=crop)
    G, _, _ = array_from_band(
        dir + get_band_file(images_list, 6), data='CMI', crop=crop)
    B, _, _ = array_from_band(
        dir + get_band_file(images_list, 5), data='CMI', crop=crop)

    RGB, lon_cen, height = fire_detection_composite(R, G, B, meta)

    if mask:
        RGB = apply_mask(RGB)

    if plot:
        apply_plot(RGB, parser.parse_args().show_plot, lon_cen, height, extent)
    else:
        Image.fromarray((RGB * 255).astype(np.uint8)
                        ).save(export_dir + "P" + dir.replace("/", "-") + ".png")
