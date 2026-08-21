"""
Satellite tile fetcher and georeferencing service.
Fetches high-resolution satellite imagery tiles for a given bounding box / GeoJSON polygon
and calculates affine transformation matrices for GPS-to-pixel conversions.
"""
import math
import io
import requests
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any

class SatelliteService:
    def __init__(self, zoom: int = 18, provider: str = "google"):
        self.zoom = zoom
        self.provider = provider
        # Tile endpoints
        self.tile_providers = {
            "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "google_hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
        }

    def get_tile_url(self, provider: str = None) -> str:
        prov = provider or self.provider
        return self.tile_providers.get(prov, self.tile_providers["google"])

    def deg_to_num(self, lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = (lon_deg + 180.0) / 360.0 * n
        ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
        return xtile, ytile

    def num_to_deg(self, xtile: float, ytile: float, zoom: int) -> Tuple[float, float]:
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg

    def fetch_satellite_image_for_polygon(
        self,
        coordinates: List[List[float]],
        zoom: int = 18,
        provider: str = "google"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fetches and stitches tiles covering the polygon bounding box.
        coordinates: list of [lon, lat] points
        Returns: (stitched_image_bgr_np, geo_metadata)
        """
        lons = [p[0] for p in coordinates]
        lats = [p[1] for p in coordinates]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        # Convert to tile coordinates
        x_min, y_min = self.deg_to_num(max_lat, min_lon, zoom)
        x_max, y_max = self.deg_to_num(min_lat, max_lon, zoom)

        x_start, x_end = int(math.floor(x_min)), int(math.floor(x_max))
        y_start, y_end = int(math.floor(y_min)), int(math.floor(y_max))

        # Limit max tiles for performance (3x3 = 9 tiles max per single crop)
        x_end = min(x_end, x_start + 3)
        y_end = min(y_end, y_start + 3)

        width_tiles = (x_end - x_start + 1)
        height_tiles = (y_end - y_start + 1)
        
        stitched = Image.new('RGB', (width_tiles * 256, height_tiles * 256))
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        tile_template = self.get_tile_url(provider)

        for i, x in enumerate(range(x_start, x_end + 1)):
            for j, y in enumerate(range(y_start, y_end + 1)):
                url = tile_template.format(z=zoom, y=y, x=x)
                try:
                    res = requests.get(url, headers=headers, timeout=5)
                    if res.status_code == 200:
                        tile_img = Image.open(io.BytesIO(res.content)).convert('RGB')
                        stitched.paste(tile_img, (i * 256, j * 256))
                except Exception as e:
                    print(f"Failed to fetch tile {x},{y}: {e}")

        # Geographic bounds of the stitched canvas
        nw_lat, nw_lon = self.num_to_deg(x_start, y_start, zoom)
        se_lat, se_lon = self.num_to_deg(x_end + 1, y_end + 1, zoom)
        canvas_w, canvas_h = stitched.size
        
        # Exact Web Mercator forward & inverse projection (eliminates any spatial distortion)
        def geo_to_pixel(lon, lat):
            xt, yt = self.deg_to_num(lat, lon, zoom)
            px = (xt - x_start) * 256.0
            py = (yt - y_start) * 256.0
            return [float(px), float(py)]

        def pixel_to_geo(px, py):
            xt = x_start + (px / 256.0)
            yt = y_start + (py / 256.0)
            lat, lon = self.num_to_deg(xt, yt, zoom)
            return [float(lon), float(lat)]

        polygon_pixels = [geo_to_pixel(p[0], p[1]) for p in coordinates]
        
        # Convert PIL RGB to OpenCV BGR
        img_np = np.array(stitched)[:, :, ::-1].copy()

        geo_meta = {
            "nw_bounds": [nw_lat, nw_lon],
            "se_bounds": [se_lat, se_lon],
            "canvas_size": [canvas_w, canvas_h],
            "polygon_pixels": polygon_pixels,
            "geo_to_pixel": geo_to_pixel,
            "pixel_to_geo": pixel_to_geo
        }

        return img_np, geo_meta
