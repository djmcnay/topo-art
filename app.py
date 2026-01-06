# app.py
import json
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from mapbox_topo_art import TopoArt

# %% Streamlit Page Config

# set page config (first) and give the page a title
st.set_page_config(layout="centered", page_title="Topo Art", page_icon="🗺️")
st.title("Topo Art")

# %% Default Variables: define upfront but can be updated via loading saved object

# originally main desk size David wants
# now including important paper sizes
default_size_options = {
    "desktop X Large": (150.0, 80.0, 0.15),
    "A3": (42.0, 29.7, 0.15),
    "A2": (59.4, 42.0, 0.15),
    "A1": (84.1, 59.4, 0.15),
    "A0": (118.9, 84.1, 0.15),
    "desktop Large": (140.0, 70.0, 0.15),
    "desktop Standard": (120.0, 60.0, 0.15),
    "desktop Deep": (140.0, 80.0, 0.15),
    "desktop Compact": (100.0, 50.0, 0.15),
}

# selection of colour scales:
# Custom is where we build our own (default state)
# Others are imported from Plotly
colour_scale_options = (
    "Custom",
    "Viridis",
    "Tealrose",
    "Tealrose_r"
)

default_custom_scale = [
    (0.0, "rgba(0, 128, 128, 0.5)"),
    (0.25, "rgba(255, 255, 255, 0.5)"),
    (1.0, "rgba(128, 0, 128, 0.5)"),
]

# %% App Authentication

# import streamlit_authenticator as stauth
# from streamlit_authenticator.utilities import LoginError

# # Load authenticator config file: contains usernames and hashed passwords
# with open("./.streamlit/auth_config.yaml") as file:
#     config = yaml.load(file, Loader=yaml.loader.SafeLoader)

# # Initialize authenticator object
# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days']
# )
#
# ### Authentication Block
# # check if the user is already authenticated in the session state
# # if not trigger the authentication widget
# if not st.session_state.get("authentication_status", None):
#     try:
#         authenticator.login(location="main", key="login-widget-home")
#         st.rerun()      # not required: trigger removes the widget on success
#     except LoginError as e:
#          st.error(e)
#
# # check authentication status before continuing
# if st.session_state["authentication_status"] is None:
#     st.warning("Please enter your username and password")
#     st.stop()
# elif st.session_state["authentication_status"] is False:
#     st.error("Username/password is incorrect")
#     st.stop()

# %% Cached Resources & Session State variables

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

# %% App: layout

tabs = st.tabs(["Map", "Topo Graph", "Load Saved Work"])

# start with load saved work
# this updates art with saved data but also overrides some app default params,
# for example, the colour scale
with tabs[2]:

    st.markdown(f"""
    ### Load Saved Work
    Need to be very specific, this is only for GeoJSON files saved from the Topo Graph tab.
    Those are JSON files containing centre, bbox, elevation data and zoom level for the map, 
    they also contain the colour scale and contour settings for the artwork.
    """)

    uploaded_file = st.file_uploader(
        "Upload GeoJSON file",
        type=["geojson", "json"],
        accept_multiple_files=False,
        help="Upload a very specific GeoJSON file containing map and art data to load into the app.",
    )

    if uploaded_file is not None:

        # read the file and use built in function to update params in art
        file_content = uploaded_file.read()
        art.load_geojson_from_stream(content=file_content)

        # remember that colour_scale can be Custom or Named,
        # Named is a string like 'Tealrose' but Custom needs to be List[Tuple[Float, Str(RGBA)]]
        # So if the colour scale uploaded to art is a list, then override the defaults for the colour_scale
        if isinstance(art.colour_scale, list):
            default_custom_scale = art.colour_scale

# %% App: Map, Central Point and Bounding Box Data

# inputs: determine the width, height and grid-square size
with st.sidebar:

    st.markdown("### Map Settings")

    # the default size dropdown for the main sizes used for desks and artwork
    default_size = st.selectbox("Default Sizing", default_size_options.keys(), index=0,)

    # width, height and granularity
    cols = st.columns(3)
    with cols[0]:
        width = st.number_input(
            "width",
            value=default_size_options[default_size][0],
            min_value=1.0,
            max_value=1000.0,
            help="width of image in cm",
        )
    with cols[1]:
        height = st.number_input(
            "height",
            value=default_size_options[default_size][1],
            min_value=1.0,
            max_value=1000.0,
            help="height of image in cm",
        )
    with cols[2]:
        grid_size = st.number_input(
            "grid size",
            value=default_size_options[default_size][2],
            min_value=0.01,
            max_value=100.0,
            help="size of grid squares in km i.e. 0.2 implies 1cm is 200m x 200m",
        )

# %% Map Tab

# Render map & capture clicks
with tabs[0]:

    # Choose the initial map centre: otherwise default to London
    map_center = [51.4266, 0] if art.centre is None else [art.centre["lat"], art.centre["lon"]]
    zoom = st.session_state["map_zoom"]

    # create a map and allow Lat/Lon popups on click
    m = folium.Map(location=map_center, zoom_start=zoom, tiles="OpenStreetMap",)
    m.add_child(folium.LatLngPopup())

    # If we have a centre, draw marker + bbox rectangle
    if art.centre is not None:
        lat_c, lon_c = art.centre["lat"], art.centre["lon"]

        # marker at centre
        folium.Marker(location=[lat_c, lon_c], popup=f"Centre: {lat_c:.4f}, {lon_c:.4f}").add_to(m)

        # compute bbox using art & draw rectangle
        min_lon, min_lat, max_lon, max_lat = art.bbox_from_coords(
            lon=lon_c,
            lat=lat_c,
            size_km=grid_size,
            aspect=(width, height),
        )

        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],  # note: (lat, lon) order for Folium
            color="red",
            weight=2,
            fill=False,
        ).add_to(m)

    # streamlit Folium map object... the actual map
    st_data = st_folium(m, width=900, height=550)

    if art.centre is not None:
        st.markdown(f"Centre: {map_center[0]:.4f}, {map_center[1]:.4f}; BBox: {[f'{i:.4f}' for i in art.bbox]}")

    # Update centre on click (and cause rerun -> rectangle updates)
    # Utility here is to store the zoom level and centres when we click
    # Streamlit must be re-run, and we don't want the map to move from the current view
    if st_data:
        if st_data.get("last_clicked"):
            if "zoom" in st_data:
                st.session_state["map_zoom"] = st_data["zoom"]

            art.centre = {
                "lat": st_data["last_clicked"]["lat"],
                "lon": st_data["last_clicked"]["lng"],
            }
            st.rerun()

# Make Call to Mapbox API
with st.sidebar:
    if st.button("Download GeoJSON", width="stretch", type="primary"):
        art.mapbox_geojson_from_bbox(zoom=11)

# If there is no elevation data, then stop running the app
if art.Z is None:
    st.stop()

# %% App: Topo Art itself

with (st.sidebar):

    st.markdown("### Colour scale")

    # dropdown menu to select colour scale; set colour scale
    # this is oddly complicated - the options are above, but Custom means a bespoke scale
    # because we can load the default (could be a saved name) or we could be working in bespoke space
    # if bespoke we need to check it's a list (not a string) and then name it "Custom"
    dd_colour_scales = st.selectbox(
        "Colour Scale",
        colour_scale_options,
        index=colour_scale_options.index("Custom" if isinstance(art.colour_scale, list) else art.colour_scale),
    )
    art.colour_scale = dd_colour_scales

    # only if we have selected Custom do we need all the rest of the selection faff
    # in the fullness of time we can param this into a widget if required
    if dd_colour_scales == "Custom":

            # find how many colours we are having in the colour gradient [max 3]
            # won't work properly when reloading saved data
            n_colours = st.slider("number of colours", 1, 3, len(default_custom_scale), step=1)

            # this is a hack to get a list of the hex codes and opacities from the rgba codes
            # remember a colour scale will be like [(0.0, 'rgba(r, g, b, a)')]
            # so we use that to extract the hex from rgba and the opacity
            # these then become the defaults for the colour picker and the input to the opacity dataframe
            hex_opacity = []
            for i, v in enumerate(default_custom_scale):
                ci, oi = art.rgba_to_hex_and_opacity(v[1])
                hex_opacity.append((ci, oi))

            # widget for colour pickers (with defaults)
            cols = st.columns(3)
            with cols[0]:
                # must always have at least `1-colour & c1 is the primary
                c1 = st.color_picker("low", value=hex_opacity[0][0])
            with cols[1]:
                # only require the mid-colour in a 3-point scale
                # in a 2-point scale we just set mid as c1
                c2 = st.color_picker("mid", value=hex_opacity[1][0]) if n_colours == 3 else c1
            with cols[2]:
                # for high, we need it for 2 or 3 colours
                # set as c1 only picking a single colour
                c3 = c1 if n_colours == 1 else st.color_picker("high", value=hex_opacity[2][0])

            # Initial values for the opacities
            # use the hex_opacity list (above) which is created from the rgba colour scale
            opacity_df = pd.DataFrame({
                'point': ['low', 'mid', 'high'],
                'opacity': [i[1] for i in hex_opacity]
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
                width='stretch',
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

            # we can pick the default value from the 2nd entry in the scale
            scale_mid = st.slider("midpoint", 0.0, 1.0, value=default_custom_scale[1][0], step=0.01)

            # for the colour scale we always use 3 colours
            # this is because with 2-we have no control over the midpoint
            # obviously if n_colours==1 these are all the same;
            # n_colours==2 implies c1 and c2 are the same so midpoint is between c2 and c3
            art.colour_scale_from_hex(
                c_low=c1, o_low=o1,
                c_mid=c2, o_mid=o2,
                c_high=c3, o_high=o3,
                midpoint=scale_mid
            )


with st.sidebar:

    st.markdown("### Contour")

    # streamlit inputs for meters per contour and contour width; updated from art
    mpc = st.slider("metres per contour", 5.0, 100.0, art.metres_per_contour, step=5.0)
    contour_width = st.slider("contour width", 0.0, 1.0, art.contour_width, step=0.01)

    # update art with the streamlit elements
    art.metres_per_contour = mpc
    art.contour_width = contour_width

    # set contour colour and opacity
    cols = st.columns(2)
    c_hex, c_opacity = art.rgba_to_hex_and_opacity(art.contour_colour)
    with cols[0]:
        contour_colour = st.color_picker("contour colour", value=c_hex)
    with cols[1]:
        contour_opacity = st.slider("contour opacity", 0.0, 1.0, value=c_opacity, step=0.01)

    # set the internal RGBA contour colour
    art.contour_from_hex(contour_colour, contour_opacity)

# %% Create Figure

fig = art.plot_contour()

# %% Saving and Validation

with st.sidebar:

    with st.expander("Advanced Settings", expanded=False):

        save_format = st.selectbox("save format", ("svg", "png"), index=0)

        # scaling factor for image export
        n_scale = st.sidebar.number_input(
            "scale factor",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="scale multiplier when downloading SVG image",
        )

        pixels_per_cm = st.number_input(
            "pixels per cm",
            value=37.8,
            help="Assuming a standard 96 DPI (dots per inch), converted to cm; 96 pixels/inch ÷ 2.54cm = 37.8 pixels/cm"
        )

        if st.button(
                "confirm center",
                type="secondary",
                width="stretch",
                help="validate centroids; will show X as paper centre & O as target lat / long",
        ):
            # add an X in the centre of the plotly plot by paper reference,
            # then add an O at the target x and y co-ordinates (which are the latitude and longitude)
            fig.add_annotation(x=0.5, xref="paper", y=0.5, yref="paper", text="X", showarrow=False)
            fig.add_annotation(x=art.centre["lon"], y=art.centre["lat"], text="O", showarrow=False)

        if st.button(
                "clear cache",
                type="secondary",
                width="stretch",
                help="clears centre, bbox, elevation data & zoom from the art object; requires fresh data call.",
        ):
            art.centre = None
            art.bbox = None
            art.Z = None
            art.zoom = None
            st.rerun()

# %% Art Tab

with tabs[1]:

    # plot the figure
    st.plotly_chart(
        fig,
        width="stretch",
        config = {
          'toImageButtonOptions': {
            'format': save_format,
            'filename': 'topo_art',
            'height': int(height * pixels_per_cm),
            'width': int(width * pixels_per_cm),
            'scale': n_scale}
        }
    )


    # button to download data as a JSON object
    geojson = art.payload_geojson()
    if st.download_button(
            "save GeoJson",
            data=json.dumps(geojson),
            file_name="topo_art.geojson",
            mime="application/json",
    ):
        st.success(f"saved to topo_art.geojson")
