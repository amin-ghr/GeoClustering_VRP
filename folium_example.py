'''
Folium is a Python library that makes it easy to create interactive maps.
It leverages the Leaflet JavaScript library to generate maps with various layers, markers, and popups.
You can use Folium to visualize data on maps, create interactive web-based maps, and more.
In this example, I'll walk you through how to use Folium to create a simple map and add markers to it.
'''

import folium

# Create a base map
m = folium.Map(location=[48.8588443, 2.2943506], zoom_start=15)

# Add markers to the map
# You can add multiple markers with different locations and popups
folium.Marker([48.8588443, 2.2943506], popup='Eiffel Tower').add_to(m)
folium.Marker([48.8606298, 2.3378339], popup='Louvre Museum').add_to(m)
folium.Marker([48.8529379, 2.3447785], popup='Notre-Dame Cathedral').add_to(m)

# Save the map as an HTML file
m.save('map.html')

'''
In this example:
1. We import the folium library.
2. We create a base map (m) using folium.Map(). You can specify the initial location and zoom level using the location and zoom_start parameters.
3.We add markers to the map using folium.Marker(). Each marker is associated with a specific location (specified as latitude and longitude) and a popup message.
4. Finally, we save the map to an HTML file using m.save('map.html').

When you run this code, it will generate an HTML file (in the current directory) called "map.html."
Open this file in a web browser, and you'll see an interactive map with markers at the specified locations.
You can click on the markers to view the pop-up messages.

Folium provides various features for customizing maps, including adding circles, polygons, popups, and tooltips.
You can explore the Folium documentation for more advanced usage and customization options: https://python-visualization.github.io/folium/.
'''

