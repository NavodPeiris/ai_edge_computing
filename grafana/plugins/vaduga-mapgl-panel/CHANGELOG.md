# Changelog

## 1.6.1
* fix cluster labels edit in thresholds to comply with braking < Tooltip > component since Grafana 10.3.0
* fix tooltip css classes for pointerEvents:all 
* ncu u: bump deck.gl & luma.gl versions

## 1.6.0
* cluster legend-filter
* fullscreen and compass widgets
* migrate to deck.gl 9 (with WebGPU support in the near future) 

## 1.5.0 
* point circle and text label dimensions
* svg icon rules collapsible + resource picker 
* bugfix: cluster hull polygon onHover doesn't lag on large datasets

## 1.4.0
* alerting states from built-in Grafana annotations query. State colors for nodes and clusters
* see-through convex hull polygon for cluster area 

## 1.3.0
* cluster max zoom menu select to control clusterization
* convex hull polygon shows cluster boundaries, cluster expansion zoom on click
* restore Grafana >=9.2.5 support
* fix performance issues that occured on large datasets because of composite cluster+circles+icons+text layer with sublayers constantly recalculating. Now that IconGeoJsonLayer (circles+icons+text) is separated from IconClusterLayer, deck.gl has less to render 
## 1.2.0

* svg icons for nodes
* text labels with collision filter.
* bug fixes: allow lineWidth custom size, no min/max

## 1.1.0

* Multi-source, multi-target support
* Switch path direction by declaring dashboard variable 'locRole'.
* Data-links: icon in tooltip to sets values for 'target' and 'source' dashboard variables. 
This lets you show charts dynamically in other panels.
* Comment icons for intermediate locations from text and color inlined in coordinates (ex.: [37.560447,55.550818, 0, "comment", "green"])
* Aggregation nodes and offset for overlapping lines.
* stat1/stat2 switch to disable offset and show secondary metric
* edge labels in stat2 mode , aggregation nodes labels.

## 1.0.2

* New: support for parent path as an array of coordinates or location names
* New: aggregation nodes and offset for overlapping lines in parent path.
* Parent line style improvements:
  -Extended path to root as a separate dotted line.    
* Bug fixes: 
 - isolate config options for different layers;  


## 1.0.1

- Multi layers support
- PolygonsLayer, Path (LineStrings) layer from frames datasource 
- Static GeoJson layer with FeatureCollection support from GeoJson file (url)
- Advanced thresholds processor for metrics. Set specific color for any set of parameters describing group of features.
- Points show toggle

## 1.0.0 

Initial release.
Repository has a demo provisioned dashboard with mock datasource 
