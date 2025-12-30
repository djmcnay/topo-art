# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
from mapbox_topo_art import TopoArt


@st.cache_resource
def init_topo_art():
    return TopoArt()

art = init_topo_art()

# init a folium zoom level in the session state
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 2

st.set_page_config(layout="wide")
st.title("Pick a point on the map")

# inputs: determine the width, height and grid-square size
with st.sidebar:
    width = st.number_input("width", value=150, min_value=100, max_value=200)
    height = st.number_input("height", value=80, min_value=50, max_value=200)
    grid_size = st.number_input("grid_size", value=0.2, min_value=0.01, max_value=10.0)


# Choose initial map centre
if art.centre is not None:
    map_center = [art.centre["lat"], art.centre["lon"]]
    zoom = st.session_state["map_zoom"]
else:
    map_center = [0, 0]
    zoom = st.session_state["map_zoom"]


# create map & allow Lat/Lon popups on click
m = folium.Map(location=map_center, zoom_start=zoom, tiles="OpenStreetMap",)
m.add_child(folium.LatLngPopup())

# If we have a centre, draw marker + bbox rectangle
if art.centre is not None:

    lat_c, lon_c = art.centre["lat"], art.centre["lon"]
    # lon_c = art.centre["lon"]

    # marker at centre
    folium.Marker(
        location=[lat_c, lon_c],
        popup=f"Centre: {lat_c:.4f}, {lon_c:.4f}",
    ).add_to(m)

    # compute bbox using art & draw rectangle
    min_lon, min_lat, max_lon, max_lat = art.bbox_from_coords(
        lon=lon_c, lat=lat_c, size_km=grid_size, aspect=(width, height)
    )

    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],    # note: (lat, lon) order for Folium
        color="red",
        weight=2,
        fill=False,
    ).add_to(m)

# Render map & capture clicks
st_data = st_folium(m, width=900, height=550)

# Update centre on click (and cause rerun -> rectangle updates)
if st_data:

    if "zoom" in st_data:
        st.session_state["map_zoom"] = st_data["zoom"]

    if st_data.get("last_clicked"):
        art.centre = {
            "lat": st_data["last_clicked"]["lat"],
            "lon": st_data["last_clicked"]["lng"],
        }
        st.rerun()

with st.sidebar:
    if st.button("click"):
        art.mapbox_geojson_from_bbox(zoom=11)

if art.Z is None:
    st.stop()

# colour styling
c1 = st.sidebar.color_picker("low colour", value="#008080")
o1 = st.sidebar.slider("low opacity", 0.0, 1.0, 0.5, step=0.01)
c2 = st.sidebar.color_picker("mid colour", value="#FFFFFF")
o2 = st.sidebar.slider("mid opacity", 0.0, 1.0, 0.5, step=0.01)
c3 = st.sidebar.color_picker("high colour", value="#800080")
o3 = st.sidebar.slider("high opacity", 0.0, 1.0, 0.5, step=0.01)

scale_mid = st.slider("midpoint", 0.0, 1.0, 0.5, step=0.01)

metres_per_contour = st.slider("metres per contour", 5.0, 100.0, 25.0, step=5.0)
contour_width = st.slider("contour width", 0.0, 1.0, 0.5, step=0.01)
contour_colour = st.color_picker("contour colour", value="#000000")
contour_opacity = st.slider("contour opacity", 0.0, 1.0, 0.5, step=0.01)

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


colourscale = [
    [0.0, hex_to_rgba_str(c1, opacity=o1)],
    [scale_mid, hex_to_rgba_str(c2, opacity=o2)],
    [1.0, hex_to_rgba_str(c3, opacity=o3)],
]


fig = art.plot_contour(
    colorscale=colourscale,
    showscale=False,
    metres_per_contour=metres_per_contour,
    contour_width=contour_width,
    contour_colour=hex_to_rgba_str(contour_colour, contour_opacity),
)

st.plotly_chart(fig)

