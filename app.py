# app.py
import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from mapbox_topo_art import TopoArt

# set app config stuff... can't be bothered to make it too pretty
st.set_page_config(
    layout="centered",
    page_title="Topo Art",
    page_icon="🗺️",
)

st.title("Topo Art")

load_dotenv()
assert "MAPBOX_TOKEN" in os.environ
print(os.environ["MAPBOX_TOKEN"][:6])

@st.cache_resource
def init_topo_art():
    """ create an instance of TopoArt and load cached data if available """
    return TopoArt()

art = init_topo_art()

# session state
# init a folium zoom level in the session state
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 7

if 'map_center' not in st.session_state:
    st.session_state['map_center'] = [0, 0]

# inputs: determine the width, height and grid-square size
with st.sidebar:

    st.markdown("### Map Settings")

    cols = st.columns(3)
    with cols[0]:
        width = st.number_input(
            "width",
            value=150,
            min_value=50,
            max_value=200,
            help="width of desktop in cm",
        )
    with cols[1]:
        height = st.number_input(
            "height",
            value=80,
            min_value=50,
            max_value=200,
            help="height of desktop in cm",
        )
    with cols[2]:
        grid_size = st.number_input(
            "grid size",
            value=0.15,
            min_value=0.01,
            max_value=10.0,
            help="size of grid squares in km i.e. 0.2 implies 1cm is 200m x 200m",
        )

# Choose the initial map centre
if art.centre is not None:
    map_center = [art.centre["lat"], art.centre["lon"]]
else:
    map_center = [51.4266, 0]

zoom = st.session_state["map_zoom"]

# create a map and allow Lat/Lon popups on click
m = folium.Map(location=map_center, zoom_start=zoom, tiles="OpenStreetMap",)
m.add_child(folium.LatLngPopup())

# If we have a centre, draw marker + bbox rectangle
if art.centre is not None:

    lat_c, lon_c = art.centre["lat"], art.centre["lon"]

    # marker at centre
    folium.Marker(
        location=[lat_c, lon_c],
        popup=f"Centre: {lat_c:.4f}, {lon_c:.4f}",
    ).add_to(m)

    # compute bbox using art & draw rectangle
    min_lon, min_lat, max_lon, max_lat = art.bbox_from_coords(
        lon=lon_c,
        lat=lat_c,
        size_km=grid_size,
        aspect=(width, height),
    )

    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],    # note: (lat, lon) order for Folium
        color="red",
        weight=2,
        fill=False,
    ).add_to(m)


tabs = st.tabs(["Map", "Topo Graph"])

# Render map & capture clicks
with tabs[0]:

    st_data = st_folium(m, width=900, height=550)

    if art.centre is not None:
        st.markdown(
            f"Centre: {map_center[0]:.4f}, {map_center[1]:.4f}; Bounding Box: {[f'{i:.4f}' for i in art.bbox]}",
        )

    # Update centre on click (and cause rerun -> rectangle updates)
    if st_data:

        if st_data.get("last_clicked"):

            if "zoom" in st_data:
                st.session_state["map_zoom"] = st_data["zoom"]

            art.centre = {
                "lat": st_data["last_clicked"]["lat"],
                "lon": st_data["last_clicked"]["lng"],
            }
            st.rerun()

with st.sidebar:
    if st.button("Download GeoJSON", use_container_width=True, type="primary"):
        art.mapbox_geojson_from_bbox(zoom=11)

if art.Z is None:
    st.stop()

with st.sidebar:

    st.markdown("### Colour scale")

    # selection of colour scales:
    # Artemis Custom is where we build our own
    # Others are imported from Plotly
    colour_scale_options = (
        "Artemis Custom",
        "Viridis",
        "Tealrose",
        "Tealrose_r"
    )

    # dropdown menu to select colour scale; set colourscale
    dd_colour_scales = st.selectbox("Colour Scale", colour_scale_options, index=0)
    colourscale = dd_colour_scales

    # only if we have selected Artemis Custom do we need all the rest of the selection faff
    # in the fullness of time we can param this into a widget if required
    if dd_colour_scales == "Artemis Custom":

            # find how many colours we are having in the colour gradient [max 3]
            n_colours = st.slider("number of colours", 1, 3, 3, step=1)

            # widget for colour pickers (with defaults)
            cols = st.columns(3)
            with cols[0]:
                # must always have at least `1-colour & c1 is the primary
                c1 = st.color_picker("low", value="#008080")
            with cols[1]:

                # only require the mid-colour in a 3-point scale
                # in a 2-point scale we just set mid as c1
                c2 = st.color_picker("mid", value="#FFFFFF") if n_colours == 3 else c1
            with cols[2]:

                # for high, we need it for 2 or 3 colours
                # set as c1 only picking a single colour
                c3 = c1 if n_colours == 1 else st.color_picker("high", value="#800080")

            # Initial values for the opacities
            opacity_df = pd.DataFrame({
                'point': ['low', 'mid', 'high'],
                'opacity': [0.5, 0.5, 0.5]
            })

            # Use data editor to adjust all three values in one place
            updated_df = st.data_editor(
                opacity_df,
                column_config={
                    "point": st.column_config.TextColumn("Position"),
                    "opacity": st.column_config.NumberColumn(
                        "Opacity",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01,
                        format="%.2f",
                    ),
                },
                hide_index=True,
                use_container_width=True
            )

            # Get the three opacity values
            o1 = updated_df.loc[updated_df['point'] == 'low', 'opacity'].values[0]
            o2 = updated_df.loc[updated_df['point'] == 'mid', 'opacity'].values[0]
            o3 = updated_df.loc[updated_df['point'] == 'high', 'opacity'].values[0]

            if n_colours == 1:
                o2 = o1
                o3 = o1
            elif n_colours == 2:
                o3 = o1

            scale_mid = st.slider("midpoint", 0.0, 1.0, 0.25, step=0.01)

            # for the colourscale we always use 3 colours
            # this is because with 2-we have no control over the midpoint
            # obviously if n_colours==1 these are all the same;
            # n_colours==2 implies c1 and c2 are the same so midpoint is between c2 and c3
            colourscale = [
                [0.0, art.hex_to_rgba_str(c1, opacity=o1)],
                [scale_mid, art.hex_to_rgba_str(c2, opacity=o2)],
                [1.0, art.hex_to_rgba_str(c3, opacity=o3)],
            ]


with st.sidebar:

    st.markdown("### Contour")
    metres_per_contour = st.slider("metres per contour", 5.0, 100.0, 20.0, step=5.0)
    contour_width = st.slider("contour width", 0.0, 1.0, 0.25, step=0.01)

    cols = st.columns(2)
    with cols[0]:
        contour_colour = st.color_picker("contour colour", value="#000000")
    with cols[1]:
        contour_opacity = st.slider("contour opacity", 0.0, 1.0, 0.35, step=0.01)

with tabs[1]:

    fig = art.plot_contour(
        colorscale=colourscale,
        showscale=False,
        metres_per_contour=metres_per_contour,
        contour_width=contour_width,
        contour_colour=art.hex_to_rgba_str(contour_colour, contour_opacity),
    )

    if st.sidebar.button(
            "confirm center",
            type="secondary",
            help="validate centroids; will show X as paper centre & O as target lat / long",
    ):
        # add an X in the centre of the plotly plot by paper reference,
        # then add an O at the target x and y co-ordinates (which are the lattitude and longitude)
        fig.add_annotation(x=0.5, xref="paper", y=0.5, yref="paper", text="X", showarrow=False)
        fig.add_annotation(x=art.centre["lon"], y=art.centre["lat"], text="O", showarrow=False)


    st.plotly_chart(
        fig,
        use_container_width=False,
        config = {
          'toImageButtonOptions': {
            'format': 'svg',
            'filename': 'topo_art',
            'scale': 5}
        }
    )

