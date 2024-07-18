from datetime import datetime, timezone, timedelta
import time
from final import get_band_file, array_from_band, fire_detection_composite, apply_plot
import s3fs

start_t = time.perf_counter()

bucket_name = 'noaa-goes16'
product_name = 'ABI-L2-CMIPF'

now = datetime.now(timezone.utc) - timedelta(minutes=10)
year = now.year
day_of_year = now.timetuple().tm_yday
hour = now.hour


fs = s3fs.S3FileSystem(anon=True)
path = f'{bucket_name}/{product_name}/{year}/{day_of_year}/{hour:02.0f}/'
files = fs.ls(path)

prefix = 'http://noaa-goes16.s3.amazonaws.com/'
sufix = '#mode=bytes'

url_7 = prefix + '/'.join(get_band_file(files, 7).split('/')[1:]) + sufix
url_6 = prefix + '/'.join(get_band_file(files, 6).split('/')[1:]) + sufix
url_5 = prefix + '/'.join(get_band_file(files, 5).split('/')[1:]) + sufix

end_t = time.perf_counter()

print(f'Dowloading: {url_7}')
R, extent, meta = array_from_band(url_7, data='CMI', crop=False)
print(time.perf_counter()-end_t)

print(f'Dowloading: {url_6}')
G, _, _ = array_from_band(url_6, data='CMI', crop=False)
print(time.perf_counter()-end_t)

print(f'Dowloading: {url_5}')
B, _, _ = array_from_band(url_5, data='CMI', crop=False)
print(time.perf_counter()-end_t)

print("Producing RGB composite")
RGB, lon_cen, height = fire_detection_composite(R, G, B, meta)
print(time.perf_counter()-end_t)

print("Showing ...")
apply_plot(RGB, show=True, lon_cen=lon_cen, height=height, extent=extent)
