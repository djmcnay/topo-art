import os
import io
import json
import requests
import mercantile
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import plotly.graph_objects as go
from typing import List, Tuple


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

        # critical co-ordinates
        self.centre: tuple = None
        self.bbox: tuple | list = None
        self.Z: np.ndarray = None
        self.zoom: int = None

        # plot parameters
        self.metres_per_contour: float = 20.0
        self.contour_width: float = 0.5
        self.contour_colour: str = "rgba(0, 0, 0, 0.15)"
        self.colour_scale: str | list = [
            (0.0, "rgba(0, 128, 128, 0.5)"),
            (0.25, "rgba(255, 255, 255, 0.5)"),
            (1.0, "rgba(128, 0, 128, 0.5)")
        ]

        # actual plot
        self.fig: go.Figure = None

    @property
    def metres_per_contour(self):
        return self._metres_per_contour
    @metres_per_contour.setter
    def metres_per_contour(self, value):
        try:
            self._metres_per_contour = float(value)
        except Exception as e:
            raise TypeError("metres_per_contour must be a float") from e

    @property
    def contour_width(self):
        return self._contour_width
    @contour_width.setter
    def contour_width(self, value):
        try:
            self._contour_width = float(value)
        except Exception as e:
            raise TypeError("contour_width must be a float") from e



    @property
    def centre(self):
        return self._centre

    @centre.setter
    def centre(self, value):
        if value is None or isinstance(value, dict):
            self._centre = value
        else:
            raise TypeError("centre must be a dict with keys lon and lat")

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        if value is None:
            self._bbox = value
        elif isinstance(value, (list, tuple)):
            self._bbox = tuple(float(v) for v in value)
        else:
            raise TypeError("bbox must be a list or tuple of (min_lon, min_lat, max_lon, max_lat)")

    @property
    def Z(self):
        return self._Z

    @Z.setter
    def Z(self, value):
        if value is None:
            self._Z = value
        else:
            try:
                self._Z = np.array(value, dtype=float)
            except Exception as e:
                raise TypeError("Z must be a numpy array of floats") from e

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        if value is None:
            self._zoom = value
        else:
            try:
                self._zoom = int(value)
            except Exception as e:
                raise TypeError("zoom must be an integer") from e




    def _load_from_payload(self, payload: dict):
        """ iterates over items in the (json) payload and will overwrite if class already has that attribute """
        for k, v in payload.items():
            if hasattr(self, k):
                print(f"Overwriting {k} from payload")
                setattr(self, k, v)

    def load_geojson_from_stream(self, content: str):
        """ Built for Streamlit, loads cached data """
        payload = json.loads(content)
        self._load_from_payload(payload)

    def load_geojson(self, json_path):
        """ Load a cached elevation grid from JSON and return (Z, bbox, zoom)."""
        with open(json_path, "r") as f:
            payload = json.load(f)
        self._load_from_payload(payload)


    def payload_geojson(self):
        """ """

        # Save as JSON (simple container, not true GeoJSON)
        payload_map = {
            "centre": self.centre,
            "bbox": list(self.bbox),
            "zoom": self.zoom,
            "Z": self.Z.tolist(),
        }

        payload_plot = {
            "colour_scale": self.colour_scale,
            "contour_colour": self.contour_colour,
            "metres_per_contour": self.metres_per_contour,
            "contour_width": self.contour_width,
        }

        return {**payload_map, **payload_plot}


    def save_geojson(self, json_path: str = "topo_art.geojson"):
        """ Save GeoJSON to File Path """
        with open(json_path, "w") as f:
            json.dump(self.payload_geojson(), f)


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

    @staticmethod
    def hex_to_rgba_str(hex_color: str, opacity: float = 0.5) -> str:
        """
        Convert hex color to rgba() string.

        Parameters
        ----------
        hex_color : str
            '#RRGGBB' or 'RRGGBB'
        opacity : float, optional
            Alpha in [0, 1], default 0.5

        Returns
        -------
        str
            'rgba(r, g, b, a)'
        """
        hex_color = hex_color.lstrip("#")

        if len(hex_color) != 6:
            raise ValueError("hex_color must be in format #RRGGBB")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        opacity = max(0.0, min(1.0, opacity))  # clamp
        return f"rgba({r}, {g}, {b}, {opacity})"

    @staticmethod
    def rgba_to_hex_and_opacity(rgba_str):
        """
        Convert an rgba string to a hex color code and separate opacity value.

        Args:
            rgba_str (str): RGBA string in the format 'rgba(r, g, b, a)' where
                             r, g, b are integers in the range 0-255 and
                             a is a float in the range 0-1.

        Returns:
            tuple: (hex_color, opacity) where hex_color is a string in the format '#RRGGBB'
                   and opacity is a float in the range 0-1.

        Example:
            >>> rgba_to_hex_and_opacity('rgba(255, 0, 128, 0.5)')
            ('#ff0080', 0.5)
        """
        # Extract the RGBA components from the string
        try:
            # Remove 'rgba(' and ')', then split by commas
            rgba_parts = rgba_str.replace('rgba(', '').replace(')', '').split(',')

            # Convert r, g, b to integers
            r = int(rgba_parts[0].strip())
            g = int(rgba_parts[1].strip())
            b = int(rgba_parts[2].strip())

            # Extract opacity as float
            opacity = float(rgba_parts[3].strip())

            # Convert RGB to hex format
            hex_color = f'#{r:02x}{g:02x}{b:02x}'

            return hex_color, opacity

        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid RGBA string format: {rgba_str}. Expected format: 'rgba(r, g, b, a)'") from e


    def colour_scale_from_hex(
            self,
            c_low: str, o_low: float = 0.5,
            midpoint: float = 0.5,
            c_mid: str = None, c_high: str = None,
            o_mid: float = None, o_high: float = None,
            update_self: bool = True,
    ) -> list:
        """
        Creates a Plotly colourscale from three hex colours & opacity values.

        Doesn't specifically need to be a 3-colour scale, could be 1-3.
        If 1, then we just set colour low for all 3.
        If 2, then we set colur mid to colour low - there is logic to this because it still allows for a midpoint.

        Args:
            c_low: hex colour string, e.g. '#000000'
            c_mid: hex colour string, e.g. '#FFFFFF' (optional)
            c_high: hex colour string, e.g. '#FFFFFF' (optional)
            midpoint: float between 0 and 1, e.g. 0.5
            o_low, o_mid, o_high: float between 0 and 1, e.g. 0.5

        Returns:
            List[Tuple[float, str]]: list of tuples of (value, colour)

        """

        # update colour to match c1 if missing
        c_mid = c_low if c_mid is None else c_mid
        c_high = c_low if c_high is None else c_high

        # update opacity to match o1 if missing
        o_mid = o_low if o_mid is None else o_mid
        o_high = o_low if o_high is None else o_high

        colour_scale = [
            (0.0, self.hex_to_rgba_str(c_low, o_low)),
            (midpoint, self.hex_to_rgba_str(c_mid, o_mid)),
            (1.0, self.hex_to_rgba_str(c_high, o_high))
        ]

        # update stored value
        if update_self:
            self.colour_scale = colour_scale
            return self.colour_scale
        else:
            return colour_scale

    def contour_from_hex(self, colour: str, opacity: float = 0.5, update_self: bool = True):
        """ Update contour colour from hex colour string."""
        rgba_colour = self.hex_to_rgba_str(colour, opacity)
        if update_self:
            self.contour_colour = rgba_colour
            return self.contour_colour
        else:
            return rgba_colour

    def plot_contour(
            self,
            colorscale=None,
            metres_per_contour=None,
            contour_width=None,
            contour_colour=None,
            update_params: bool = True,
    ):
        """
        Plot an elevation grid as a filled contour plot using Plotly,
        with the correct geographic aspect ratio.
        """

        # parameters required to stored internally
        min_lon, min_lat, max_lon, max_lat = self.bbox
        h, w = self.Z.shape

        # optional from self
        colorscale = colorscale if colorscale is not None else self.colour_scale
        metres_per_contour = metres_per_contour if metres_per_contour is not None else self.metres_per_contour
        contour_width = contour_width if contour_width is not None else self.contour_width
        contour_colour = contour_colour if contour_colour is not None else self.contour_colour

        if update_params:
            self.metres_per_contour = metres_per_contour
            self.contour_width = contour_width
            self.contour_colour = contour_colour
            self.colour_scale = colorscale

        #
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
                showscale=False,
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

        if update_params:
            self.fig = fig
            return self.fig
        else:
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
