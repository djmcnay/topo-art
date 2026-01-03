---
title: TopoArt
emoji: "🗺"
colorFrom: blue
colorTo: green
python_version: 3.12.9
sdk: streamlit
sdk_version: 1.51.0
app_file: app.py
pinned: true
header: mini
---

# Topo Art 
Project is an attempt to create "art" from a topographical map. 
Originally to build a custom desktop to be printed at [The Printed Find](https://theprintedfind.com/).
We have turned this into a Streamlit App which can pull from 

The workflow is to use [Mapbox](https://www.mapbox.com/) as a source of topographical data. 
We then convert this into a custom contour map in [Plotly](https://plotly.com/python/contour-plots/), 
which we save as an SVG file.
