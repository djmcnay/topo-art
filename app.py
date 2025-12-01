# app.py
import streamlit as st
import folium
from streamlit_folium import st_folium
from mapbox_topo_art import TopoArt


@st.cache_resource
def init_topo_art():
    return TopoArt()

art = init_topo_art()

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
    zoom = 8
else:
    map_center = [0, 0]
    zoom = 2


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
if st_data and st_data.get("last_clicked"):
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

fig = art.plot_contour()
st.plotly_chart(fig)

