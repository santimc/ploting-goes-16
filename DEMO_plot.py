from datetime import datetime, timezone, timedelta
import time
from final import array_of_arrays_from_band, fire_detection_composite, apply_plot
import s3fs

start_t = time.perf_counter()

bucket_name = 'noaa-goes16'
product_name = 'ABI-L2-MCMIPF'

now = datetime.now(timezone.utc) - timedelta(minutes=10)
year = now.year
day_of_year = now.timetuple().tm_yday
hour = now.hour


fs = s3fs.S3FileSystem(anon=True)
path = f'{bucket_name}/{product_name}/{year}/{day_of_year}/{hour:02.0f}/'
files = fs.ls(path)

prefix = 'http://noaa-goes16.s3.amazonaws.com/'
sufix = '#mode=bytes'

url = prefix + '/'.join(files[0].split('/')[1:]) + sufix

end_t = time.perf_counter()

print(f'Dowloading: {url}')
r, g, b = array_of_arrays_from_band(url, bands=[7, 6, 5], crop=False)

R, extent, meta = r
G, _, _ = g
B, _, _ = b

print(time.perf_counter()-end_t)

print("Producing RGB composite")
RGB, lon_cen, height = fire_detection_composite(R, G, B, meta)
print(time.perf_counter()-end_t)

print("Showing ...")
apply_plot(RGB, show=True, lon_cen=lon_cen, height=height, extent=extent)
