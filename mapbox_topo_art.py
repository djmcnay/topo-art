import os
import io
import json
import requests
import mercantile
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import plotly.graph_objects as go


class TopoArt:
    """ """

    def __init__(
            self,
            mapbox_token: str = None
    ):
        """ """

        # mapbox api key can either be provided as a string or set as an env variable
        if mapbox_token is None:
            load_dotenv()
            mapbox_token = os.getenv("MAPBOX_TOKEN", None)
        self.mapbox_token = mapbox_token

        #
        self.centre: tuple = None
        self.bbox: tuple | list = None
        self.Z: np.ndarray = None
        self.zoom: int = None


    def load_geojson(self, json_path):
        """
        Load a cached elevation grid from JSON and return (Z, bbox, zoom).
        """
        with open(json_path, "r") as f:
            payload = json.load(f)

        self.Z = np.array(payload["z"], dtype=float)
        self.bbox = tuple(payload["bbox"])
        self.zoom = payload["zoom"]

    def save_geojson(self, json_path: str = "topo_art.geojson"):
        """
        """

        # Save as JSON (simple container, not true GeoJSON)
        payload = {
            "bbox": list(self.bbox),
            "zoom": self.zoom,
            "z": self.Z.tolist(),
        }

        with open(json_path, "w") as f:
            json.dump(payload, f)


    def bbox_from_coords(self, lon: float, lat: float, size_km: float = 0.2, aspect: list = (150, 80)):
        """
            Create a bbox centred on (lon, lat).

            Args:
                lon, lat: float: coordinates of centre point.
                aspect: (width_units, height_units), e.g. (150, 80) for a 150cm x 80cm piece.
                size_km: km per aspect unit.

            Example:
              aspect=(150, 80), size_km=1  -> 150km x 80km
              aspect=(150, 80), size_km=0.2 -> 30km x 16km
            """

        from math import cos, radians

        if lon is None or lat is None:
            lon, lat = self.centre

        # Half extents in km
        half_w_km = (aspect[0] * size_km) / 2.0
        half_h_km = (aspect[1] * size_km) / 2.0

        # Approx conversion rates at this latitude
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * cos(radians(lat))

        # Convert km -> degrees
        dlon = half_w_km / km_per_deg_lon
        dlat = half_h_km / km_per_deg_lat

        self.bbox = (
            lon - dlon,     # min lon
            lat - dlat,     # min latitude
            lon + dlon,     # max long
            lat + dlat,     # max latitude
        )

        return self.bbox

    def bbox_from_coords_with_rotation(
            self, lon: float,
            lat: float, size_km:
            float = 0.2,
            aspect: list = (150, 80),
            rotation_deg: float = 0,
    ):
        """
        Create a bbox centred on (lon, lat), with optional rotation.
    
        Args:
            lon, lat: float: coordinates of centre point.
            aspect: (width_units, height_units), e.g. (150, 80) for a 150cm x 80cm piece.
            size_km: km per aspect unit.
            rotation_deg: rotation angle in degrees (clockwise from north)
    
        Example:
          aspect=(150, 80), size_km=1  -> 150km x 80km
          aspect=(150, 80), size_km=0.2 -> 30km x 16km
          rotation_deg=45 -> 45-degree clockwise rotation
        """

        from math import cos, sin, radians

        if lon is None or lat is None:
            lon, lat = self.centre

        # Store these values for later use in other methods
        self.rotation_deg = rotation_deg
        self.aspect = aspect

        # Width and height in km
        width_km = aspect[0] * size_km
        height_km = aspect[1] * size_km

        # Approx conversion rates at this latitude
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * cos(radians(lat))

        # Convert half-widths from km to degrees (different scales for lon and lat)
        half_width_lon_deg = (width_km / 2.0) / km_per_deg_lon
        half_height_lat_deg = (height_km / 2.0) / km_per_deg_lat

        if rotation_deg == 0:
            # No rotation, use simple calculation
            self.bbox = (
                lon - half_width_lon_deg,     # min lon
                lat - half_height_lat_deg,    # min latitude
                lon + half_width_lon_deg,     # max long
                lat + half_height_lat_deg,    # max latitude
            )

            # Store corners for consistency
            self.rotated_corners = [
                (lon - half_width_lon_deg, lat - half_height_lat_deg),  # bottom left
                (lon + half_width_lon_deg, lat - half_height_lat_deg),  # bottom right
                (lon + half_width_lon_deg, lat + half_height_lat_deg),  # top right
                (lon - half_width_lon_deg, lat + half_height_lat_deg),  # top left
            ]
        else:
            # For rotation, we need to be careful with the different scales of lon and lat
            # First define corners in "natural" coordinates where 1 unit = 1 km
            corners_km = [
                (-width_km/2, -height_km/2),   # bottom left
                (width_km/2, -height_km/2),    # bottom right
                (width_km/2, height_km/2),     # top right
                (-width_km/2, height_km/2),    # top left
            ]

            # Apply rotation in km-space (where distances are uniform)
            rot_rad = radians(rotation_deg)
            cos_rot = cos(rot_rad)
            sin_rot = sin(rot_rad)

            rotated_corners_km = []
            for x_km, y_km in corners_km:
                # Apply rotation matrix in km space
                rot_x_km = x_km * cos_rot - y_km * sin_rot
                rot_y_km = x_km * sin_rot + y_km * cos_rot
                rotated_corners_km.append((rot_x_km, rot_y_km))

            # Convert from km to degrees (different scales for lon and lat)
            rotated_corners = []
            for x_km, y_km in rotated_corners_km:
                corner_lon = lon + (x_km / km_per_deg_lon)
                corner_lat = lat + (y_km / km_per_deg_lat)
                rotated_corners.append((corner_lon, corner_lat))

            # Store the rotated corners for visualization
            self.rotated_corners = rotated_corners

            # Calculate bounding box that contains all rotated corners
            corner_lons = [c[0] for c in rotated_corners]
            corner_lats = [c[1] for c in rotated_corners]

            self.bbox = (
                min(corner_lons),  # min lon
                min(corner_lats),  # min lat
                max(corner_lons),  # max lon
                max(corner_lats),  # max lat
            )

        return self.bbox

    @staticmethod
    def terrain_rgb_to_elevation(rgb_arr: np.ndarray) -> np.ndarray:
        """
        Convert Mapbox terrain-rgb tile to elevation in metres.

        rgb_arr: (H, W, 3) uint8
        elevation = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
        """
        R = rgb_arr[:, :, 0].astype(np.float64)
        G = rgb_arr[:, :, 1].astype(np.float64)
        B = rgb_arr[:, :, 2].astype(np.float64)
        elevation = -10000.0 + ((R * 256 * 256 + G * 256 + B) * 0.1)
        return elevation

    def mapbox_geojson_from_bbox(self, zoom, tile_size=256, downsample=1):
        """
        Fetch Mapbox terrain-rgb tiles for a bbox, build an elevation grid, and
        save it to a JSON file.

        bbox: (min_lon, min_lat, max_lon, max_lat)
        zoom: Web Mercator zoom level (11–13 sensible for this use case)
        downsample: integer step to subsample rows/cols (>=1)
        """

        min_lon, min_lat, max_lon, max_lat = self.bbox

        tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=[zoom]))
        if not tiles:
            raise RuntimeError("No tiles for bbox/zoom combination")

        xs = [t.x for t in tiles]
        ys = [t.y for t in tiles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        tiles_w = max_x - min_x + 1
        tiles_h = max_y - min_y + 1

        width_px = tiles_w * tile_size
        height_px = tiles_h * tile_size

        Z = np.full((height_px, width_px), np.nan, dtype=float)

        for t in tiles:
            url = (
                f"https://api.mapbox.com/v4/mapbox.terrain-rgb/"
                f"{t.z}/{t.x}/{t.y}.pngraw?access_token={self.mapbox_token}"
            )
            r = requests.get(url)
            r.raise_for_status()

            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            rgb = np.array(img)
            elev = self.terrain_rgb_to_elevation(rgb)

            x_idx = (t.x - min_x) * tile_size
            y_idx = (t.y - min_y) * tile_size

            Z[y_idx:y_idx + tile_size, x_idx:x_idx + tile_size] = elev

        # Remove all-NaN rows/cols (e.g., pure ocean)
        valid_rows = ~np.all(np.isnan(Z), axis=1)
        valid_cols = ~np.all(np.isnan(Z), axis=0)
        Z = Z[valid_rows][:, valid_cols]

        # Flip vertically so north is “up” in the final plot
        Z = Z[::-1, :]

        # Optional downsampling for smoother / lighter plots
        if downsample > 1:
            Z = Z[::downsample, ::downsample]

        self.Z = Z


    def mapbox_geojson_from_bbox_rotated(self, zoom, tile_size=256, downsample=1):
        """
        Fetch Mapbox terrain-rgb tiles for a rotated bounding box, build an elevation grid,
        and crop/rotate the data to match the desired rectangle.

        zoom: Web Mercator zoom level (11–13 sensible for this use case)
        tile_size: Size of Mapbox tiles in pixels
        downsample: Integer step to subsample rows/cols (>=1)
        """

        from scipy.ndimage import rotate, map_coordinates
        import numpy as np

        # Get the axis-aligned bounding box (the one that contains the rotated rectangle)
        min_lon, min_lat, max_lon, max_lat = self.bbox

        # First, fetch all the tiles covering the axis-aligned bounding box
        tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=[zoom]))
        if not tiles:
            raise RuntimeError("No tiles for bbox/zoom combination")

        xs = [t.x for t in tiles]
        ys = [t.y for t in tiles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        tiles_w = max_x - min_x + 1
        tiles_h = max_y - min_y + 1

        width_px = tiles_w * tile_size
        height_px = tiles_h * tile_size

        Z = np.full((height_px, width_px), np.nan, dtype=float)

        for t in tiles:
            url = (
                f"https://api.mapbox.com/v4/mapbox.terrain-rgb/"
                f"{t.z}/{t.x}/{t.y}.pngraw?access_token={self.mapbox_token}"
            )
            r = requests.get(url)
            r.raise_for_status()

            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            rgb = np.array(img)
            elev = self.terrain_rgb_to_elevation(rgb)

            x_idx = (t.x - min_x) * tile_size
            y_idx = (t.y - min_y) * tile_size

            Z[y_idx:y_idx + tile_size, x_idx:x_idx + tile_size] = elev

        # We now have the elevation data for the entire axis-aligned bounding box
        # Flip vertically so north is "up" in the final plot
        Z = Z[::-1, :]

        # Check if we have rotation and rotated corners
        if hasattr(self, 'rotation_deg') and self.rotation_deg != 0 and hasattr(self, 'rotated_corners'):
            # Create a mask for the rotated rectangle area
            # Convert the rotated corners to pixel coordinates in our grid
            h, w = Z.shape

            # Conversion from lon/lat to pixel coordinates
            def lonlat_to_pixel(lon, lat):
                x_norm = (lon - min_lon) / (max_lon - min_lon)
                y_norm = (max_lat - lat) / (max_lat - min_lat)  # Flipped because we flipped Z
                return int(x_norm * w), int(y_norm * h)

            # Get pixel coordinates of rotated corners
            pixel_corners = [lonlat_to_pixel(lon, lat) for lon, lat in self.rotated_corners]

            # Create a mask for the rotated rectangle
            mask = np.zeros_like(Z, dtype=bool)
            from matplotlib import path
            poly_path = path.Path(pixel_corners)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            points = np.vstack((xx.flatten(), yy.flatten())).T
            mask_flat = poly_path.contains_points(points)
            mask = mask_flat.reshape(h, w)

            # Apply mask to keep only data within the rotated rectangle
            Z_masked = Z.copy()
            Z_masked[~mask] = np.nan

            # Now we need to "un-rotate" the data to get a straight rectangle
            # Calculate dimensions for the target grid (maintain aspect ratio)
            aspect_ratio = self.aspect[0] / self.aspect[1]  # width / height

            # Choose a reasonable target size based on the input data dimensions
            target_height = int(np.sqrt(np.sum(mask) / aspect_ratio))
            target_width = int(target_height * aspect_ratio)

            # Create a target grid
            Z_rotated = np.full((target_height, target_width), np.nan)

            # Find the center of the masked area
            y_indices, x_indices = np.where(mask)
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)

            # Counter-rotate the grid and sample from the masked data
            for y in range(target_height):
                for x in range(target_width):
                    # Convert to normalized coordinates (-1 to 1) from the center
                    x_norm = 2 * (x / target_width - 0.5)
                    y_norm = 2 * (y / target_height - 0.5)

                    # Apply counter-rotation
                    rot_rad = np.radians(-self.rotation_deg)  # Counter-rotate
                    x_rot = x_norm * np.cos(rot_rad) - y_norm * np.sin(rot_rad)
                    y_rot = x_norm * np.sin(rot_rad) + y_norm * np.cos(rot_rad)

                    # Scale to the original masked area size
                    x_orig = center_x + x_rot * (np.max(x_indices) - np.min(x_indices)) / 2
                    y_orig = center_y + y_rot * (np.max(y_indices) - np.min(y_indices)) / 2

                    # Check if coordinates are within bounds and sample
                    if 0 <= x_orig < w and 0 <= y_orig < h and mask[int(y_orig), int(x_orig)]:
                        # Use bilinear interpolation
                        Z_rotated[y, x] = map_coordinates(Z_masked, [[y_orig], [x_orig]], order=1)[0]

            # Use the rotated and cropped data
            Z = Z_rotated

        # Optional downsampling for smoother / lighter plots
        if downsample > 1:
            Z = Z[::downsample, ::downsample]

        self.Z = Z


    def contour_config_from_interval(self, metres_per_contour, z_clip_min=None, z_clip_max=None):
        """
        From an elevation grid Z (metres), work out zmin, zmax and ncontours
        for a desired vertical spacing in metres.
        """
        zmin = float(np.nanmin(self.Z))
        zmax = float(np.nanmax(self.Z))

        if z_clip_min is not None:
            zmin = max(zmin, z_clip_min)
        if z_clip_max is not None:
            zmax = min(zmax, z_clip_max)

        if zmax <= zmin:
            raise ValueError("zmax must be > zmin after clipping")

        ncontours = max(1, int(np.floor((zmax - zmin) / metres_per_contour)))
        return zmin, zmax, ncontours


    def plot_contour(
            self,
            colorscale="Viridis",
            showscale=False,
            metres_per_contour=25,
            contour_width=0.5,
            contour_colour="rgba(0, 0, 0, 0.15)",
            title=None,
    ):
        """
        Plot an elevation grid as a filled contour plot using Plotly,
        with the correct geographic aspect ratio.
        """
        min_lon, min_lat, max_lon, max_lat = self.bbox
        h, w = self.Z.shape

        _, _, n_contours = self.contour_config_from_interval(
            metres_per_contour=metres_per_contour,
        )

        # Build coordinate arrays so Plotly stretches correctly
        lons = np.linspace(min_lon, max_lon, w)
        lats = np.linspace(min_lat, max_lat, h)

        fig = go.Figure(
            go.Contour(
                z=self.Z,
                x=lons,    # longitude axis
                y=lats,    # latitude axis
                colorscale=colorscale,
                contours=dict(
                    coloring="heatmap",
                    showlines=False,
                ),
                line=dict(
                    width=contour_width,
                    color=contour_colour,
                ),
                showscale=showscale,
                ncontours=n_contours,
            )
        )

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        fig.update_layout(
            title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig

# --- Example usage for your Skye bbox ---

if __name__ == "__main__":

    art = TopoArt()

    skye_bbox = (
        -6.323133091151334,   # min_lon
        57.11935340999355,    # min_lat
        -5.93944402991653,    # max_lon
        57.25935008445512     # max_lat
    )
    zoom = 13

    print(art.mapbox_bbox_from_coords(-6.088117854601649, 57.209790, 0.2))

    # cache_path = "skye_contours.geojson"
    #
    # if not os.path.exists(cache_path):
    #     Z = mapbox_geojson_from_bbox(skye_bbox, zoom, cache_path, downsample=1)
    # else:
    #     Z, _, _ = load_geojson(cache_path)
    #
    # custom_colorscale = [
    #     [0.0, "rgba(0, 128, 128, 0.15)"],  # teal, semi-transparent
    #     [0.2, "rgba(255, 255, 255, 0.25)"],  # white
    #     [1.0, "rgba(128,   0, 128, 0.3)"]  # purple
    # ]
    #
    # fig = plot_contour(
    #     Z,
    #     bbox=skye_bbox,
    #     colorscale=custom_colorscale,
    #     showscale=False,
    #     title="Skye elevation art",
    # )
    #
    # fig.show()
    #
    # fig.write_image("skye_contours.svg", scale=2.0)
