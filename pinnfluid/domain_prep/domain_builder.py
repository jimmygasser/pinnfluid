#!/usr/bin/env python3
"""
domain_builder.py — All-in-one domain creation tool.

Interactive web UI that combines:
  1. Terrain selection (Swiss DEM via swisstopo, or flat terrain)
  2. Structure placement (click on map or enter coordinates)
  3. Domain finalization (z-shift, STL export)

Usage:
  python3 scripts/domain_prep/domain_builder.py --port 8778
  python3 scripts/domain_prep/domain_builder.py --port 8778 --dem-root dem --out-root data_preparation/singlestructures
"""

import argparse
import glob
import http.server
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path

import numpy as np

try:
    from pyproj import Transformer
except ImportError:
    Transformer = None

try:
    import rasterio
    from rasterio.merge import merge as rasterio_merge
    from rasterio.windows import from_bounds
except ImportError:
    rasterio = None

try:
    import trimesh
except ImportError:
    trimesh = None

ROOT = Path(__file__).resolve().parent.parent.parent
# The DEM/structure helper scripts (dem_prep, place_structure, finalize_domain)
# are invoked as subprocesses and live next to this file.
_DOMAIN_PREP_DIR = Path(__file__).resolve().parent

# Available structures (display_name -> stl filename)
STRUCTURES = {
    # Wind turbines (largewt/smallwt) are excluded from this beta — the meshes
    # are heavy and turbine cases are the weakest part of the current models.
    "Flat panel": "flatpanel.stl",
    "Helioplant": "helioplant.stl",
    "Inclined panel": "advancedpanel.stl",
    "Flower panel": "flowerpanel.stl",
    "Inclined table": "tableinclined.stl",
    "Steeper inclined panel": "inclinedpanel.stl",
    "Concentrator": "concentrator.stl",
    "Flat table": "tableflat.stl",
}

# Grid structures (no wind turbines — grids of WT don't exist)
GRID_STRUCTURES = {
    "Flat panel": "flatpanel.stl",
    "Helioplant": "helioplant.stl",
    "Inclined panel": "advancedpanel.stl",
    "Flower panel": "flowerpanel.stl",
    "Inclined table": "tableinclined.stl",
    "Steeper inclined panel": "inclinedpanel.stl",
    "Concentrator": "concentrator.stl",
    "Flat table": "tableflat.stl",
}

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head>
<title>Domain Builder — pinn_terr_struc</title>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<style>
  * { box-sizing:border-box; }
  body { margin:0; padding:0; font-family:'Segoe UI',-apple-system,sans-serif; }
  #sidebar {
    position:absolute; top:0; right:0; z-index:1000;
    width:380px; height:100vh; background:#fff;
    box-shadow:-2px 0 16px rgba(0,0,0,0.15);
    display:flex; flex-direction:column; overflow-y:auto;
  }
  #sidebar-header {
    background:linear-gradient(135deg,#1565C0,#1976D2);
    color:#fff; padding:18px 20px; flex-shrink:0;
  }
  #sidebar-header h2 { margin:0 0 4px 0; font-size:17px; }
  #sidebar-header p { margin:0; font-size:11px; opacity:0.85; }
  .section { padding:14px 20px; border-bottom:1px solid #eee; }
  .section-title {
    font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.5px; color:#888; margin:0 0 8px 0;
  }
  label { display:block; font-size:12px; color:#444; margin-bottom:3px; font-weight:500; }
  input[type=text], input[type=number], select {
    width:100%; padding:7px 9px; border:1px solid #ddd; border-radius:4px;
    font-size:12px; margin-bottom:8px; background:#fafafa;
  }
  input:focus, select:focus { outline:none; border-color:#1976D2; background:#fff; }
  .row { display:flex; gap:8px; }
  .row > div { flex:1; }
  .btn {
    padding:9px 14px; cursor:pointer; border:none; border-radius:5px;
    font-size:12px; font-weight:600; width:100%; margin-bottom:6px;
  }
  .btn-primary { background:#1976D2; color:white; }
  .btn-primary:hover { background:#1565C0; }
  .btn-primary:disabled { background:#ccc; cursor:not-allowed; }
  .btn-secondary { background:#f5f5f5; color:#555; border:1px solid #ddd; }
  .btn-secondary:hover { background:#eee; }
  .btn-danger { background:#e53935; color:white; }
  .btn-danger:hover { background:#c62828; }
  .btn-success { background:#43A047; color:white; }
  .btn-success:hover { background:#388E3C; }
  .btn-success:disabled { background:#ccc; cursor:not-allowed; }
  .checkbox-row { display:flex; align-items:center; gap:6px; margin-bottom:8px; }
  .checkbox-row input { margin:0; }
  .checkbox-row label { margin:0; font-size:12px; }
  #struct-list { max-height:180px; overflow-y:auto; font-size:11px; }
  #struct-list .struct-item {
    padding:4px 8px; background:#f8f8f8; border-radius:3px;
    margin-bottom:3px; display:flex; justify-content:space-between; align-items:center;
  }
  #struct-list .struct-item .remove { color:#e53935; cursor:pointer; font-weight:bold; }
  #sel-info { font-size:11px; line-height:1.5; color:#555; min-height:18px; }
  #sel-info b { color:#333; }
  .help { font-size:10px; color:#999; margin-top:2px; }
  #map { position:absolute; top:0; left:0; right:380px; height:100vh; }
  #status {
    position:absolute; bottom:12px; left:12px; z-index:1000;
    background:rgba(255,255,255,0.95); padding:10px 14px; border-radius:6px;
    box-shadow:0 2px 8px rgba(0,0,0,0.15);
    font-family:Consolas,monospace; font-size:11px;
    max-width:calc(100vw - 420px);
  }
  #status.ok { border-left:3px solid #4CAF50; }
  #status.err { border-left:3px solid #e53935; }
  #status.busy { border-left:3px solid #FF9800; }
</style>
</head><body>
<div id="map"></div>

<div id="sidebar">
  <div id="sidebar-header">
    <h2>Domain Builder</h2>
    <p>Terrain + structure domain creation for pinn_terr_struc</p>
  </div>

  <!-- DOMAIN SETTINGS -->
  <div class="section">
    <div class="section-title">1. Domain settings</div>
    <label>Domain name</label>
    <input type="text" id="domainName" value="" placeholder="e.g. 51_jura_ridge_turbine"/>
    <div class="row">
      <div>
        <label>Domain size (m)</label>
        <select id="domainSize">
          <option value="500">500 x 500</option>
          <option value="1000" selected>1000 x 1000</option>
          <option value="2000">2000 x 2000</option>
          <option value="3000">3000 x 3000</option>
        </select>
      </div>
      <div>
        <label>Wind from (deg)</label>
        <input type="number" id="windFrom" value="270" min="0" max="360" step="1"/>
        <div class="help">Meteo: 0=N, 90=E, 270=W</div>
      </div>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="customZoffset"/>
      <label for="customZoffset">Custom total height above terrain</label>
      <input type="number" id="zOffsetVal" value="300" min="50" max="1000" step="50" style="width:70px; display:none; margin-left:6px;"/>
    </div>
    <div id="zOffsetDefault" class="help" style="margin:-4px 0 8px 0;">Auto: 200-500m based on domain size</div>
    <div class="row">
      <div>
        <label>Uref (m/s)</label>
        <input type="number" id="uref" value="10" min="1" max="30" step="0.5"/>
      </div>
      <div>
        <label>Zref (m)</label>
        <input type="number" id="zref" value="20" min="5" max="100" step="1"/>
      </div>
      <div>
        <label>z0 (m)</label>
        <input type="number" id="z0" value="0.1" min="0.001" max="2" step="0.01"/>
      </div>
    </div>
  </div>

  <!-- TERRAIN -->
  <div class="section">
    <div class="section-title">2. Terrain</div>
    <div class="checkbox-row">
      <input type="checkbox" id="flatTerrain"/>
      <label for="flatTerrain">Flat terrain (no DEM download)</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="customDem"/>
      <label for="customDem">Import custom DEM (max 3x3 km, projected GeoTIFF)</label>
    </div>
    <div id="customDemRow" style="display:none; margin:0 0 8px 0;">
      <input type="file" id="customDemFile" accept=".tif,.tiff,image/tiff" style="font-size:11px; width:100%; margin-bottom:6px;"/>
      <button class="btn btn-primary" id="btnUploadDEM" onclick="uploadCustomDEM()" disabled>
        Upload custom DEM
      </button>
      <div class="help">CRS must be projected (LV95, UTM, ...) in metres. North-up.</div>
    </div>
    <p style="font-size:11px; color:#666; margin:0 0 6px 0;" id="terrainHelp">
      <b>Click</b> on the map to select terrain center, or <b>draw rectangle</b>.
    </p>
    <div id="sel-info"></div>
    <button class="btn btn-primary" id="btnDownloadDEM" onclick="downloadDEM()" disabled>
      Download DEM tiles
    </button>
    <button class="btn btn-secondary" id="btnResetTerrain" onclick="resetTerrain()" style="display:none; margin-top:4px;">
      Reset terrain selection
    </button>
    <div id="demStatus" style="font-size:11px; color:#43A047; margin-top:4px;"></div>
  </div>

  <!-- SINGLE STRUCTURES -->
  <div class="section">
    <div class="section-title">3. Single structures (optional)</div>
    <div class="checkbox-row">
      <input type="checkbox" id="enableSingle"/>
      <label for="enableSingle">Place individual structures</label>
    </div>
    <div id="singleOptions" style="display:none;">
      <div class="row">
        <div style="flex:2">
          <label>Structure type</label>
          <select id="structType">
            """ + "".join(f'<option value="{v}">{k}</option>' for k, v in STRUCTURES.items()) + r"""
          </select>
        </div>
        <div style="flex:1">
          <label>Yaw (deg)</label>
          <input type="number" id="structYaw" value="0" step="1"/>
          <div class="help">0 = facing wind</div>
        </div>
      </div>
      <p style="font-size:11px; color:#666; margin:4px 0;">
        <b>Click map</b> to place structure, or enter CRS coords:
      </p>
      <div class="row">
        <div><input type="text" id="structE" placeholder="E (LV95)"/></div>
        <div><input type="text" id="structN" placeholder="N (LV95)"/></div>
        <div><button class="btn btn-secondary" onclick="addStructManual()" style="margin-top:0">Add</button></div>
      </div>
      <div id="struct-list"></div>
      <button class="btn btn-secondary" onclick="clearStructures()" style="margin-top:4px">Clear all structures</button>
      <div class="checkbox-row" style="margin-top:8px;">
        <input type="checkbox" id="flattenGround" checked/>
        <label for="flattenGround">Flatten ground around structures</label>
      </div>
      <div class="help">Levels terrain at structure footprint (+2m margin).</div>
    </div>
  </div>

  <!-- STRUCTURE GRID -->
  <div class="section">
    <div class="section-title">4. Structure grid (optional)</div>
    <div class="checkbox-row">
      <input type="checkbox" id="enableGrid"/>
      <label for="enableGrid">Place a regular grid of structures</label>
    </div>
    <div id="gridOptions" style="display:none;">
      <div class="row">
        <div>
          <label>Structure type</label>
          <select id="gridStructType">
            """ + "".join(f'<option value="{v}">{k}</option>' for k, v in GRID_STRUCTURES.items()) + r"""
          </select>
        </div>
      </div>
      <div class="row">
        <div>
          <label>Rows (wind dir)</label>
          <select id="gridRows">""" + "".join(f'<option value="{i}">{i}</option>' for i in range(1,11)) + r"""</select>
          <div class="help">Along wind (front→back)</div>
        </div>
        <div>
          <label>Columns (cross)</label>
          <select id="gridCols">""" + "".join(f'<option value="{i}">{i}</option>' for i in range(1,11)) + r"""</select>
          <div class="help">Perpendicular to wind</div>
        </div>
      </div>
      <div class="row">
        <div>
          <label>Row spacing (m)</label>
          <input type="number" id="gridSpacingX" value="3" min="1" max="20" step="0.5"/>
          <div class="help">Gap between structures (edge-to-edge)</div>
        </div>
        <div>
          <label>Col spacing (m)</label>
          <input type="number" id="gridSpacingY" value="3" min="1" max="20" step="0.5"/>
          <div class="help">Gap between structures (edge-to-edge)</div>
        </div>
      </div>
      <div class="row">
        <div>
          <label>Struct yaw (deg)</label>
          <input type="number" id="gridStructYaw" value="0" step="1"/>
          <div class="help">Each structure rotation</div>
        </div>
        <div>
          <label>Grid yaw (deg)</label>
          <input type="number" id="gridYaw" value="0" step="1"/>
          <div class="help">Whole grid rotation</div>
        </div>
      </div>
      <p style="font-size:11px; color:#666; margin:6px 0;">
        <b>Click map</b> to place grid center, or enter CRS coords:
      </p>
      <div class="row">
        <div><input type="text" id="gridE" placeholder="E (LV95)"/></div>
        <div><input type="text" id="gridN" placeholder="N (LV95)"/></div>
        <div><button class="btn btn-secondary" onclick="addGridManual()" style="margin-top:0">Set</button></div>
      </div>
      <div id="grid-info" style="font-size:11px; color:#555; margin-top:4px;"></div>
      <div class="checkbox-row" style="margin-top:8px;">
        <input type="checkbox" id="flattenGrid" checked/>
        <label for="flattenGrid">Flatten ground under entire grid</label>
      </div>
    </div>
  </div>

  <!-- BUILD -->
  <div class="section" style="border-bottom:none;">
    <div class="section-title">5. Build domain</div>
    <button class="btn btn-success" id="btnBuild" onclick="buildDomain()" disabled>
      Build domain (DEM + structures → STL)
    </button>
    <div class="help" style="margin-top:6px;">
      Creates ground.stl + structure.stl in data_preparation/, shifted to z=0.
    </div>
  </div>
</div>

<div id="status">Ready. Click on the map to select terrain.</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script>
var map = L.map('map').setView([46.8, 8.22], 9);
L.tileLayer(
  'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg',
  { attribution:'&copy; swisstopo', maxZoom:20 }
).addTo(map);
L.tileLayer(
  'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissalti3d-reliefschattierung/default/current/3857/{z}/{x}/{y}.png',
  { opacity:0.35, maxZoom:20 }
).addTo(map);

var drawnTerrain = new L.FeatureGroup(); map.addLayer(drawnTerrain);
var structMarkers = new L.FeatureGroup(); map.addLayer(structMarkers);

var drawCtl = new L.Control.Draw({
  draw:{ polygon:false, polyline:false, circle:false, circlemarker:false, marker:false,
          rectangle:{ shapeOptions:{ color:'#1976D2', weight:2, fillOpacity:0.1 } } },
  edit:{ featureGroup:drawnTerrain }
});
map.addControl(drawCtl);

var terrainBounds = null;
var terrainCenter = null;
var demReady = false;
var terrainLocked = false;  // locks terrain after DEM download
var structures = []; // {lat, lng, type, yaw, label}

function getBoxKm() {
  // Add 50% margin so rotated domain always fits inside the downloaded DEM
  return parseFloat(document.getElementById('domainSize').value) / 1000.0 * 1.5;
}

var previewGroup = L.layerGroup().addTo(map);  // all preview layers in one group

function drawDomainPreview(lat, lng) {
  previewGroup.clearLayers();

  var domSizeM = parseFloat(document.getElementById('domainSize').value);
  var windFrom = parseFloat(document.getElementById('windFrom').value);
  if (isNaN(windFrom)) windFrom = 270;
  var halfM = domSizeM / 2.0;
  var cosLat = Math.cos(lat * Math.PI / 180);

  // Helper: offset in metres to lat/lng delta
  function mToLat(m) { return m / 111320.0; }
  function mToLng(m) { return m / (111320.0 * cosLat); }

  // Rotation: dem_prep uses theta_pixel = -(wind_to - 90)
  // wind_to = (wind_from + 180) % 360
  // theta_pixel = -(wind_to - 90) = 90 - wind_to
  // This is the angle the DEM is actually rotated in pixel space (CW positive)
  // For the preview, we need the geographic rotation of the cropped domain:
  // The cropped domain's +x axis (after rotation) points in the wind_to direction
  // wind_to in geo: 0=N, 90=E, 180=S, 270=W
  var windTo = (windFrom + 180) % 360;
  // Geographic bearing of the domain's +x axis = windTo
  // Convert to math angle for rotation of corners:
  // Bearing 0(N)=90° math, 90(E)=0° math, 180(S)=270° math, 270(W)=180° math
  var xAxisRad = (90 - windTo) * Math.PI / 180;

  // Compute 4 corners of rotated square
  // Unrotated corners in metres: domain's local +x and +y axes
  // After rotation, +x points at bearing windTo
  var cornersLocal = [[-halfM,-halfM],[ halfM,-halfM],[ halfM, halfM],[-halfM, halfM]];
  var cornersLL = cornersLocal.map(function(c) {
    // c[0] = along domain +x, c[1] = along domain +y
    // Domain +x in geo: easting = cos(xAxisRad), northing = sin(xAxisRad)
    // Domain +y in geo: easting = -sin(xAxisRad), northing = cos(xAxisRad)
    var easting  = c[0] * Math.cos(xAxisRad) - c[1] * Math.sin(xAxisRad);
    var northing = c[0] * Math.sin(xAxisRad) + c[1] * Math.cos(xAxisRad);
    return [lat + mToLat(northing), lng + mToLng(easting)];
  });

  previewGroup.addLayer(L.polygon(cornersLL, {
    color:'#E65100', weight:2, fillOpacity:0.08
  }));

  // Wind arrow: wind blows FROM windFrom direction TOWARD center
  // windFrom bearing in math angle:
  var fromRad = (90 - windFrom) * Math.PI / 180;
  var arrowLenM = halfM * 0.7;
  // Arrow starts upwind of center, ends at center
  // "upwind" = in the direction windFrom points (where wind comes FROM)
  var startE =  arrowLenM * Math.cos(fromRad);
  var startN =  arrowLenM * Math.sin(fromRad);
  var endE   = -arrowLenM * 0.3 * Math.cos(fromRad);
  var endN   = -arrowLenM * 0.3 * Math.sin(fromRad);

  var startLL = [lat + mToLat(startN), lng + mToLng(startE)];
  var endLL   = [lat + mToLat(endN),   lng + mToLng(endE)];

  previewGroup.addLayer(L.polyline([startLL, endLL], {
    color:'#D32F2F', weight:3, opacity:0.8
  }));

  // Arrowhead at endLL pointing in wind direction (FROM start TO end)
  var headM = arrowLenM * 0.12;
  var windDirRad = Math.atan2(endN - startN, endE - startE); // direction of arrow
  var h1E = endE - headM * Math.cos(windDirRad - 0.4);
  var h1N = endN - headM * Math.sin(windDirRad - 0.4);
  var h2E = endE - headM * Math.cos(windDirRad + 0.4);
  var h2N = endN - headM * Math.sin(windDirRad + 0.4);

  previewGroup.addLayer(L.polygon([
    endLL,
    [lat + mToLat(h1N), lng + mToLng(h1E)],
    [lat + mToLat(h2N), lng + mToLng(h2E)]
  ], {
    color:'#D32F2F', weight:1, fillColor:'#D32F2F', fillOpacity:0.8
  }));
}

function clearPreview() {
  previewGroup.clearLayers();
}

// Update preview when wind direction or domain size changes
document.getElementById('windFrom').addEventListener('change', function() {
  if (terrainCenter) drawDomainPreview(terrainCenter.lat, terrainCenter.lng);
});
document.getElementById('domainSize').addEventListener('change', function() {
  if (terrainCenter && !terrainLocked) {
    // Redraw outer box too
    drawnTerrain.clearLayers();
    var lat=terrainCenter.lat, lng=terrainCenter.lng, half=getBoxKm()/2;
    var dLat=half/111.32, dLng=half/(111.32*Math.cos(lat*Math.PI/180));
    var b = L.latLngBounds([lat-dLat,lng-dLng],[lat+dLat,lng+dLng]);
    drawnTerrain.addLayer(L.rectangle(b, {color:'#1976D2',weight:1,fillOpacity:0.05,dashArray:'6 4'}));
    terrainBounds = b;
  }
  if (terrainCenter) drawDomainPreview(terrainCenter.lat, terrainCenter.lng);
});

// --- Map click: terrain selection OR structure/grid placement ---
map.on('click', function(e) {
  // If terrain is locked (DEM downloaded or flat), clicks place structures or grid
  if (terrainLocked || demReady) {
    // Grid mode: each click replaces the grid position
    if (document.getElementById('enableGrid').checked) {
      addGridAtLatLng(e.latlng.lat, e.latlng.lng);
      return;
    }
    if (document.getElementById('enableSingle').checked) {
      addStructAtLatLng(e.latlng.lat, e.latlng.lng);
    }
    return;
  }
  // Otherwise, select terrain
  if (document.getElementById('flatTerrain').checked) return;
  // Custom-DEM mode: terrain selection comes from the upload, not map clicks.
  if (document.getElementById('customDem').checked) return;
  drawnTerrain.clearLayers();
  var lat=e.latlng.lat, lng=e.latlng.lng, half=getBoxKm()/2;
  var dLat=half/111.32, dLng=half/(111.32*Math.cos(lat*Math.PI/180));
  var b = L.latLngBounds([lat-dLat,lng-dLng],[lat+dLat,lng+dLng]);
  drawnTerrain.addLayer(L.rectangle(b, {color:'#1976D2',weight:1,fillOpacity:0.05,dashArray:'6 4'}));
  terrainBounds = b;
  terrainCenter = {lat:lat, lng:lng};
  drawDomainPreview(lat, lng);
  showTerrainInfo(b);
  demReady = false;
  document.getElementById('demStatus').innerText = '';
});

map.on(L.Draw.Event.CREATED, function(e) {
  if (terrainLocked || demReady) return;
  if (document.getElementById('customDem').checked) return;
  drawnTerrain.clearLayers();
  drawnTerrain.addLayer(e.layer);
  terrainBounds = e.layer.getBounds();
  terrainCenter = terrainBounds.getCenter();
  showTerrainInfo(terrainBounds);
  demReady = false;
  document.getElementById('demStatus').innerText = '';
});

function showTerrainInfo(b) {
  var sw=b.getSouthWest(), ne=b.getNorthEast();
  var wKm=(ne.lng-sw.lng)*111.32*Math.cos((sw.lat+ne.lat)/2*Math.PI/180);
  var hKm=(ne.lat-sw.lat)*111.32;
  document.getElementById('sel-info').innerHTML=
    '<b>Center:</b> '+terrainCenter.lat.toFixed(5)+', '+terrainCenter.lng.toFixed(5)+'<br>'+
    '<b>Size:</b> ~'+wKm.toFixed(1)+' &times; '+hKm.toFixed(1)+' km';
  document.getElementById('btnDownloadDEM').disabled = false;
  updateBuildBtn();
}

// --- Flat terrain toggle ---
document.getElementById('flatTerrain').addEventListener('change', function() {
  var flat = this.checked;
  if (flat) {
    // Mutex with customDem.
    var cd = document.getElementById('customDem');
    if (cd && cd.checked) {
      cd.checked = false;
      cd.dispatchEvent(new Event('change'));
    }
    // Auto-select Lake Neuchâtel as flat terrain
    var lakeLat = 46.90, lakeLng = 6.85;
    terrainCenter = {lat:lakeLat, lng:lakeLng};
    var half = getBoxKm()/2;
    var dLat = half/111.32, dLng = half/(111.32*Math.cos(lakeLat*Math.PI/180));
    var b = L.latLngBounds([lakeLat-dLat,lakeLng-dLng],[lakeLat+dLat,lakeLng+dLng]);
    drawnTerrain.clearLayers();
    drawnTerrain.addLayer(L.rectangle(b, {color:'#1976D2',weight:1,fillOpacity:0.05,dashArray:'6 4'}));
    terrainBounds = b;
    drawDomainPreview(lakeLat, lakeLng);
    map.setView([lakeLat, lakeLng], 14);
    document.getElementById('sel-info').innerHTML = '<b>Flat terrain</b> (Lake Neuchatel)<br>Click "Download DEM" to get flat lake surface.';
    document.getElementById('btnDownloadDEM').disabled = false;
    document.getElementById('btnResetTerrain').style.display = 'block';
  } else {
    demReady = false;
    terrainLocked = false;
    document.getElementById('demStatus').innerText = '';
    document.getElementById('sel-info').innerHTML = '';
    document.getElementById('btnResetTerrain').style.display = 'none';
  }
  updateBuildBtn();
});

function resetTerrain() {
  // Custom-DEM teardown: drop overlay, restore Swiss tiles, clear file input.
  if (customDem) {
    _resetCustomDem();
    var cd = document.getElementById('customDem');
    if (cd) cd.checked = false;
    document.getElementById('customDemRow').style.display = 'none';
    var fInput = document.getElementById('customDemFile');
    if (fInput) fInput.value = '';
    document.getElementById('btnDownloadDEM').style.display = '';
    _showSwissTiles();
    var helpEl = document.getElementById('terrainHelp');
    if (helpEl) helpEl.innerHTML = '<b>Click</b> on the map to select terrain center, or <b>draw rectangle</b>.';
  }
  demReady = false;
  terrainLocked = false;
  terrainBounds = null;
  terrainCenter = null;
  drawnTerrain.clearLayers();
  clearPreview();
  structures = [];
  structMarkers.clearLayers();
  renderStructList();
  document.getElementById('demStatus').innerText = '';
  document.getElementById('sel-info').innerHTML = '';
  var btn = document.getElementById('btnDownloadDEM');
  btn.disabled = false;
  // Reset confirm-terrain button colour back to "needs action" (blue).
  btn.classList.remove('btn-success'); btn.classList.add('btn-primary');
  document.getElementById('btnResetTerrain').style.display = 'none';
  document.getElementById('flatTerrain').checked = false;
  updateBuildBtn();
  setStatus('Terrain reset. Click to select new area.', '');
}

// --- Structure placement ---
var redIcon = L.divIcon({
  className:'',
  html:'<div style="width:14px;height:14px;background:#e53935;border:2px solid #fff;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,0.4);"></div>',
  iconSize:[14,14], iconAnchor:[7,7]
});

function addStructAtLatLng(lat, lng) {
  var type = document.getElementById('structType').value;
  var yaw = parseFloat(document.getElementById('structYaw').value) || 0;
  var typeName = document.getElementById('structType').selectedOptions[0].text;
  var idx = structures.length + 1;
  var label = typeName + ' #' + idx;
  var crs = _latlngToCrs(lat, lng);
  var entry = {lat:lat, lng:lng, type:type, yaw:yaw, label:label};
  if (crs) { entry.crs_x = crs[0]; entry.crs_y = crs[1]; }
  structures.push(entry);
  var popup = label + (crs
    ? '<br>CRS=(' + crs[0].toFixed(1) + ', ' + crs[1].toFixed(1) + ')'
    : '<br>(' + lat.toFixed(5) + ', ' + lng.toFixed(5) + ')');
  var marker = L.marker([lat,lng], {icon:redIcon}).bindPopup(popup);
  structMarkers.addLayer(marker);
  renderStructList();
  updateBuildBtn();
  var locStr = crs
    ? 'CRS=(' + crs[0].toFixed(0) + ', ' + crs[1].toFixed(0) + ')'
    : '(' + lat.toFixed(5) + ', ' + lng.toFixed(5) + ')';
  setStatus('Placed: ' + label + ' at ' + locStr, 'ok');
}

function addStructManual() {
  var e = parseFloat(document.getElementById('structE').value);
  var n = parseFloat(document.getElementById('structN').value);
  if (isNaN(e) || isNaN(n)) { setStatus('Enter valid E/N coordinates','err'); return; }
  if (customDemActive()) {
    // Custom DEM: inputs are in DEM CRS — synthesize a fake latlng inside the
    // imageOverlay so the existing marker/preview pipeline still works.
    var ll = _crsToLatlng(e, n);
    if (!ll) { setStatus('Custom DEM not loaded','err'); return; }
    addStructAtLatLng(ll[0], ll[1]);
    return;
  }
  // Swiss path: inputs are LV95 → WGS84 for marker.
  fetch('/lv95_to_wgs84?e='+e+'&n='+n).then(r=>r.json()).then(function(d) {
    if (d.lat && d.lng) { addStructAtLatLng(d.lat, d.lng); }
    else { setStatus('Coordinate conversion failed','err'); }
  });
}

function clearStructures() {
  structures = [];
  structMarkers.clearLayers();
  renderStructList();
  updateBuildBtn();
}

function renderStructList() {
  var html = '';
  structures.forEach(function(s, i) {
    html += '<div class="struct-item">' +
      '<span>' + s.label + '<br><small>lat=' + s.lat.toFixed(5) + ' lng=' + s.lng.toFixed(5) + ' yaw=' + s.yaw + '&deg;</small></span>' +
      '<span class="remove" onclick="removeStruct('+i+')">&times;</span></div>';
  });
  document.getElementById('struct-list').innerHTML = html || '<div style="color:#999; font-size:11px;">No structures placed yet</div>';
}

function removeStruct(i) {
  structures.splice(i, 1);
  structMarkers.clearLayers();
  structures.forEach(function(s, idx) {
    s.label = s.label.split(' #')[0] + ' #' + (idx+1);
    structMarkers.addLayer(
      L.marker([s.lat,s.lng], {icon:redIcon})
        .bindPopup(s.label + '<br>(' + s.lat.toFixed(5) + ', ' + s.lng.toFixed(5) + ')')
    );
  });
  renderStructList();
  updateBuildBtn();
}

function updateBuildBtn() {
  var name = document.getElementById('domainName').value.trim();
  var ready = name && demReady;
  document.getElementById('btnBuild').disabled = !ready;
}
document.getElementById('domainName').addEventListener('input', updateBuildBtn);

// --- DEM Download ---
function downloadDEM() {
  if (!terrainBounds) return;
  var sw = terrainBounds.getSouthWest(), ne = terrainBounds.getNorthEast();
  var name = document.getElementById('domainName').value.trim();
  if (!name) { setStatus('Enter a domain name first','err'); return; }
  setStatus('Downloading DEM tiles...','busy');
  document.getElementById('btnDownloadDEM').disabled = true;
  fetch('/download_dem', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      south:sw.lat, west:sw.lng, north:ne.lat, east:ne.lng,
      domain_name:name, resolution:'2'
    })
  }).then(r=>r.json()).then(function(d) {
    if (d.success) {
      demReady = true;
      terrainLocked = true;
      document.getElementById('demStatus').innerText = 'DEM ready: ' + (d.dem_size||'') + ' — click map to place structures';
      document.getElementById('btnResetTerrain').style.display = 'block';
      // Confirm-terrain button → green to signal "ready, move on".
      var btn = document.getElementById('btnDownloadDEM');
      btn.classList.remove('btn-primary'); btn.classList.add('btn-success');
      setStatus('DEM downloaded. Now click on map to place structures.', 'ok');
      updateBuildBtn();
    } else {
      setStatus('DEM error: ' + d.error, 'err');
    }
    document.getElementById('btnDownloadDEM').disabled = false;
  }).catch(function(e) {
    setStatus('DEM error: '+e, 'err');
    document.getElementById('btnDownloadDEM').disabled = false;
  });
}

// --- Build Domain ---
function buildDomain() {
  var name = document.getElementById('domainName').value.trim();
  var domSize = parseInt(document.getElementById('domainSize').value);
  var windFrom = parseInt(document.getElementById('windFrom').value);
  var flat = document.getElementById('flatTerrain').checked;
  if (!name) { setStatus('Enter a domain name','err'); return; }

  setStatus('Building domain...','busy');
  document.getElementById('btnBuild').disabled = true;

  var structData = structures.map(function(s) {
    return {lat:s.lat, lng:s.lng, type:s.type, yaw:s.yaw,
            crs_x:s.crs_x, crs_y:s.crs_y};
  });

  var uref = parseFloat(document.getElementById('uref').value) || 10;
  var zref = parseFloat(document.getElementById('zref').value) || 20;
  var z0 = parseFloat(document.getElementById('z0').value) || 0.1;
  var flattenGround = document.getElementById('flattenGround').checked;

  fetch('/build', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      domain_name: name,
      domain_size: domSize,
      wind_from: windFrom,
      flat_terrain: flat,
      structures: structData,
      center: terrainCenter,
      uref: uref,
      zref: zref,
      z0: z0,
      flatten_ground: flattenGround,
      z_top_offset: document.getElementById('customZoffset').checked ?
        parseFloat(document.getElementById('zOffsetVal').value) : null,
      grid: document.getElementById('enableGrid').checked ? {
        type: document.getElementById('gridStructType').value,
        rows: parseInt(document.getElementById('gridRows').value),
        cols: parseInt(document.getElementById('gridCols').value),
        spacing_x: parseFloat(document.getElementById('gridSpacingX').value),
        spacing_y: parseFloat(document.getElementById('gridSpacingY').value),
        struct_yaw: parseFloat(document.getElementById('gridStructYaw').value) || 0,
        grid_yaw: parseFloat(document.getElementById('gridYaw').value) || 0,
        center: gridCenter,
        flatten: document.getElementById('flattenGrid').checked
      } : null
    })
  }).then(r=>r.json()).then(function(d) {
    if (d.success) {
      setStatus('Domain built: ' + d.output_dir, 'ok');
    } else {
      setStatus('Build error: ' + d.error, 'err');
    }
    document.getElementById('btnBuild').disabled = false;
  }).catch(function(e) {
    setStatus('Build error: '+e, 'err');
    document.getElementById('btnBuild').disabled = false;
  });
}

// --- Custom z offset toggle ---
document.getElementById('customZoffset').addEventListener('change', function() {
  document.getElementById('zOffsetVal').style.display = this.checked ? 'inline-block' : 'none';
  document.getElementById('zOffsetDefault').style.display = this.checked ? 'none' : 'inline';
});

// --- Single structure toggle ---
document.getElementById('enableSingle').addEventListener('change', function() {
  document.getElementById('singleOptions').style.display = this.checked ? 'block' : 'none';
  if (this.checked) {
    // Disable grid
    document.getElementById('enableGrid').checked = false;
    document.getElementById('gridOptions').style.display = 'none';
    gridCenter = null;
    gridPreviewGroup.clearLayers();
    document.getElementById('grid-info').innerHTML = '';
  }
  updateBuildBtn();
});

// --- Grid toggle ---
var gridCenter = null;
var gridMarker = null;
var gridPreviewGroup = L.layerGroup().addTo(map);

document.getElementById('enableGrid').addEventListener('change', function() {
  document.getElementById('gridOptions').style.display = this.checked ? 'block' : 'none';
  if (this.checked) {
    // Disable single structures and clear their markers
    document.getElementById('enableSingle').checked = false;
    document.getElementById('singleOptions').style.display = 'none';
    structures = [];
    structMarkers.clearLayers();
    renderStructList();
  }
  if (!this.checked) {
    gridCenter = null;
    gridPreviewGroup.clearLayers();
    document.getElementById('grid-info').innerHTML = '';
  }
  updateBuildBtn();
});

function addGridAtLatLng(lat, lng) {
  if (!document.getElementById('enableGrid').checked) return false;
  var gcrs = _latlngToCrs(lat, lng);
  gridCenter = {lat:lat, lng:lng};
  if (gcrs) { gridCenter.crs_x = gcrs[0]; gridCenter.crs_y = gcrs[1]; }
  gridPreviewGroup.clearLayers();

  var gridIcon = L.divIcon({
    className:'',
    html:'<div style="width:18px;height:18px;background:#FF6F00;border:2px solid #fff;border-radius:3px;box-shadow:0 1px 4px rgba(0,0,0,0.4);"></div>',
    iconSize:[18,18], iconAnchor:[9,9]
  });
  gridPreviewGroup.addLayer(L.marker([lat,lng], {icon:gridIcon}).bindPopup('Grid center'));

  var rows = parseInt(document.getElementById('gridRows').value);
  var cols = parseInt(document.getElementById('gridCols').value);
  var sx = parseFloat(document.getElementById('gridSpacingX').value);
  var sy = parseFloat(document.getElementById('gridSpacingY').value);
  // Approximate structure size (~5m) for edge-to-edge preview
  var structSize = 5.0;
  var totalX = (rows - 1) * (sx + structSize) + structSize;
  var totalY = (cols - 1) * (sy + structSize) + structSize;

  var cosLat = Math.cos(lat * Math.PI / 180);
  function mToLat(m) { return m / 111320.0; }
  function mToLng(m) { return m / (111320.0 * cosLat); }

  // Grid yaw
  var gridYawDeg = parseFloat(document.getElementById('gridYaw').value) || 0;
  // Wind direction determines the grid's +x (rows along wind)
  var windFrom = parseFloat(document.getElementById('windFrom').value);
  if (!Number.isFinite(windFrom)) windFrom = 270;
  var windTo = (windFrom + 180) % 360;
  var xAxisRad = (90 - windTo) * Math.PI / 180;
  var totalRad = xAxisRad + gridYawDeg * Math.PI / 180;

  // Draw grid footprint rectangle
  var halfX = totalX / 2, halfY = totalY / 2;
  var corners = [[-halfX,-halfY],[halfX,-halfY],[halfX,halfY],[-halfX,halfY]].map(function(c) {
    var rx = c[0] * Math.cos(totalRad) - c[1] * Math.sin(totalRad);
    var ry = c[0] * Math.sin(totalRad) + c[1] * Math.cos(totalRad);
    return [lat + mToLat(ry), lng + mToLng(rx)];
  });
  gridPreviewGroup.addLayer(L.polygon(corners, {
    color:'#FF6F00', weight:2, fillOpacity:0.12, dashArray:'4 3'
  }));

  document.getElementById('grid-info').innerHTML =
    '<b>Grid center:</b> ' + lat.toFixed(5) + ', ' + lng.toFixed(5) + '<br>' +
    '<b>Layout:</b> ' + rows + ' rows x ' + cols + ' cols<br>' +
    '<b>Footprint:</b> ~' + totalX.toFixed(0) + 'm x ' + totalY.toFixed(0) + 'm';
  updateBuildBtn();
  setStatus('Grid placed: ' + rows + 'x' + cols + ' at (' + lat.toFixed(5) + ', ' + lng.toFixed(5) + ')', 'ok');
  return true;
}

// Keep the footprint preview synchronized with every control that changes its
// size or orientation. Previously it was only redrawn when the centre was
// placed, so changing wind direction or grid yaw left a stale rectangle.
function refreshGridPreview() {
  if (!gridCenter || !document.getElementById('enableGrid').checked) return;
  addGridAtLatLng(gridCenter.lat, gridCenter.lng);
}
['gridRows', 'gridCols', 'gridSpacingX', 'gridSpacingY', 'gridYaw', 'windFrom']
  .forEach(function(id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', refreshGridPreview);
    el.addEventListener('change', refreshGridPreview);
  });

function addGridManual() {
  var e = parseFloat(document.getElementById('gridE').value);
  var n = parseFloat(document.getElementById('gridN').value);
  if (isNaN(e) || isNaN(n)) { setStatus('Enter valid E/N coordinates','err'); return; }
  if (customDemActive()) {
    var ll = _crsToLatlng(e, n);
    if (!ll) { setStatus('Custom DEM not loaded','err'); return; }
    addGridAtLatLng(ll[0], ll[1]);
    return;
  }
  fetch('/lv95_to_wgs84?e='+e+'&n='+n).then(r=>r.json()).then(function(d) {
    if (d.lat && d.lng) addGridAtLatLng(d.lat, d.lng);
    else setStatus('Coordinate conversion failed','err');
  });
}

// =========================================================================
// Custom DEM upload (alternative to swisstopo) — replaces the basemap with an
// uploaded GeoTIFF rendered as a hillshade PNG. Click handlers stay in lat/lng
// (via L.imageOverlay at synthetic Swiss-area bounds) and we translate to/from
// real DEM CRS coords through _latlngToCrs / _crsToLatlng.
// =========================================================================
var customDem = null;       // { boundsCrs:[W,S,E,N], crs, overlayBounds (LatLngBounds), imgLayer, w_m, h_m }
var swissBaseLayers = [];   // tile layers we hide while custom DEM is active

function customDemActive() { return customDem !== null && customDem.imgLayer; }

function _collectSwissTileLayers() {
  if (swissBaseLayers.length > 0) return;
  map.eachLayer(function(l) {
    if (l && l._url && typeof l._url === 'string' && l._url.indexOf('swisstopo') >= 0) {
      swissBaseLayers.push(l);
    }
  });
}

function _hideSwissTiles() {
  _collectSwissTileLayers();
  swissBaseLayers.forEach(function(l) { if (map.hasLayer(l)) map.removeLayer(l); });
}

function _showSwissTiles() {
  swissBaseLayers.forEach(function(l) { if (!map.hasLayer(l)) l.addTo(map); });
}

function _latlngToCrs(lat, lng) {
  if (!customDem) return null;
  var ob = customDem.overlayBounds;
  var bc = customDem.boundsCrs;  // [W, S, E, N] in DEM CRS
  var fx = (lng - ob.getWest())  / (ob.getEast()  - ob.getWest());
  var fy = (lat - ob.getSouth()) / (ob.getNorth() - ob.getSouth());
  var crsX = bc[0] + fx * (bc[2] - bc[0]);
  var crsY = bc[1] + fy * (bc[3] - bc[1]);
  return [crsX, crsY];
}

function _crsToLatlng(crsX, crsY) {
  if (!customDem) return null;
  var ob = customDem.overlayBounds;
  var bc = customDem.boundsCrs;
  var fx = (crsX - bc[0]) / (bc[2] - bc[0]);
  var fy = (crsY - bc[1]) / (bc[3] - bc[1]);
  var lng = ob.getWest()  + fx * (ob.getEast()  - ob.getWest());
  var lat = ob.getSouth() + fy * (ob.getNorth() - ob.getSouth());
  return [lat, lng];
}

document.getElementById('customDem').addEventListener('change', function() {
  var on = this.checked;
  document.getElementById('customDemRow').style.display = on ? 'block' : 'none';
  if (on) {
    // Mutex with flatTerrain. Also wipe any pending Swiss-style selection.
    document.getElementById('flatTerrain').checked = false;
    drawnTerrain.clearLayers();
    clearPreview();
    terrainBounds = null;
    terrainCenter = null;
    demReady = false;
    terrainLocked = false;
    var helpEl = document.getElementById('terrainHelp');
    if (helpEl) helpEl.innerHTML = '<b>Upload</b> a projected GeoTIFF (max 3x3 km), then click on it to place structures.';
    document.getElementById('btnDownloadDEM').style.display = 'none';
    document.getElementById('sel-info').innerHTML = '';
    _hideSwissTiles();
  } else {
    if (customDem) _resetCustomDem();
    var helpEl2 = document.getElementById('terrainHelp');
    if (helpEl2) helpEl2.innerHTML = '<b>Click</b> on the map to select terrain center, or <b>draw rectangle</b>.';
    document.getElementById('btnDownloadDEM').style.display = '';
    _showSwissTiles();
  }
});

document.getElementById('customDemFile').addEventListener('change', function() {
  var btn = document.getElementById('btnUploadDEM');
  btn.disabled = !(this.files && this.files.length > 0);
  // If a previous DEM is loaded, reset so the new one will replace it cleanly.
  if (customDem) _resetCustomDem();
});

function _readFileAsBase64(file) {
  return new Promise(function(resolve, reject) {
    var r = new FileReader();
    r.onload = function() {
      var s = r.result;  // "data:...;base64,XXXXX"
      var i = s.indexOf(',');
      resolve(i >= 0 ? s.substring(i + 1) : s);
    };
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function uploadCustomDEM() {
  var fileInput = document.getElementById('customDemFile');
  var name = document.getElementById('domainName').value.trim();
  if (!name) { setStatus('Enter a domain name first','err'); return; }
  if (!fileInput.files || fileInput.files.length === 0) {
    setStatus('Choose a GeoTIFF file first','err'); return;
  }
  var file = fileInput.files[0];
  var maxBytes = 20 * 1024 * 1024;
  if (file.size > maxBytes) {
    setStatus('File too large ('+(file.size/1e6).toFixed(1)+' MB > 20 MB)','err'); return;
  }
  setStatus('Uploading custom DEM ('+(file.size/1e6).toFixed(1)+' MB)...','busy');
  document.getElementById('btnUploadDEM').disabled = true;
  _readFileAsBase64(file).then(function(b64) {
    return fetch('/upload_dem', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({domain_name:name, data_base64:b64})
    });
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (!d.success) {
      setStatus('Upload error: ' + d.error, 'err');
      document.getElementById('btnUploadDEM').disabled = false;
      return;
    }
    _installCustomDem(d);
    setStatus('Custom DEM ready ('+d.dem_size+'). Click on it to place structures.','ok');
    document.getElementById('btnUploadDEM').disabled = false;
    document.getElementById('btnUploadDEM').classList.remove('btn-primary');
    document.getElementById('btnUploadDEM').classList.add('btn-success');
  }).catch(function(e) {
    setStatus('Upload error: '+e, 'err');
    document.getElementById('btnUploadDEM').disabled = false;
  });
}

function _installCustomDem(d) {
  // Build synthetic Swiss-area lat/lng bounds whose metric extent matches the
  // DEM (so Leaflet's metric helpers — domain-rect, wind arrow — render at
  // the correct scale). Clicks are translated to real CRS via _latlngToCrs.
  var fakeLat = 46.5, fakeLng = 7.5;
  var halfLat = (d.height_m / 2) / 111320.0;
  var halfLng = (d.width_m / 2) / (111320.0 * Math.cos(fakeLat * Math.PI / 180));
  var ob = L.latLngBounds([fakeLat - halfLat, fakeLng - halfLng],
                          [fakeLat + halfLat, fakeLng + halfLng]);
  var imgUrl = 'data:image/png;base64,' + d.png_base64;
  _hideSwissTiles();
  // Drop any prior overlay before replacing.
  if (customDem && customDem.imgLayer) map.removeLayer(customDem.imgLayer);
  var overlay = L.imageOverlay(imgUrl, ob, {opacity:1.0, interactive:false});
  overlay.addTo(map);
  customDem = {
    boundsCrs: d.bounds_crs,
    crs: d.crs,
    overlayBounds: ob,
    imgLayer: overlay,
    w_m: d.width_m,
    h_m: d.height_m,
  };
  // Lock terrain at the DEM centre — clicks now place structures, not terrain.
  terrainBounds = ob;
  terrainCenter = {lat: fakeLat, lng: fakeLng};
  demReady = true;
  terrainLocked = true;
  drawnTerrain.clearLayers();
  drawnTerrain.addLayer(L.rectangle(ob, {color:'#1976D2', weight:1, fillOpacity:0.0, dashArray:'6 4'}));
  drawDomainPreview(fakeLat, fakeLng);
  map.fitBounds(ob, {padding:[20,20]});
  document.getElementById('sel-info').innerHTML =
    '<b>Custom DEM:</b> ' + d.width_m.toFixed(0) + ' x ' + d.height_m.toFixed(0) + ' m'
    + '<br><b>CRS:</b> <code style="font-size:10px;">' + d.crs + '</code>';
  document.getElementById('btnResetTerrain').style.display = 'block';
  document.getElementById('demStatus').innerText = 'DEM ready: ' + d.dem_size + ' — click on it to place structures';
  // Update structure manual-coord placeholders to "X (CRS)" / "Y (CRS)".
  var sE = document.getElementById('structE'); if (sE) sE.placeholder = 'X (CRS)';
  var sN = document.getElementById('structN'); if (sN) sN.placeholder = 'Y (CRS)';
  var gE = document.getElementById('gridE');   if (gE) gE.placeholder = 'X (CRS)';
  var gN = document.getElementById('gridN');   if (gN) gN.placeholder = 'Y (CRS)';
  updateBuildBtn();
}

function _resetCustomDem() {
  if (customDem && customDem.imgLayer) map.removeLayer(customDem.imgLayer);
  customDem = null;
  drawnTerrain.clearLayers();
  clearPreview();
  structures = [];
  structMarkers.clearLayers();
  renderStructList();
  gridCenter = null;
  if (typeof gridPreviewGroup !== 'undefined') gridPreviewGroup.clearLayers();
  var gi = document.getElementById('grid-info'); if (gi) gi.innerHTML = '';
  terrainBounds = null;
  terrainCenter = null;
  demReady = false;
  terrainLocked = false;
  document.getElementById('sel-info').innerHTML = '';
  document.getElementById('demStatus').innerText = '';
  var u = document.getElementById('btnUploadDEM');
  u.classList.remove('btn-success'); u.classList.add('btn-primary');
  // Restore manual-coord placeholders.
  var sE = document.getElementById('structE'); if (sE) sE.placeholder = 'E (LV95)';
  var sN = document.getElementById('structN'); if (sN) sN.placeholder = 'N (LV95)';
  var gE = document.getElementById('gridE');   if (gE) gE.placeholder = 'E (LV95)';
  var gN = document.getElementById('gridN');   if (gN) gN.placeholder = 'N (LV95)';
  document.getElementById('btnResetTerrain').style.display = 'none';
  updateBuildBtn();
}

function setStatus(msg, cls) {
  var el=document.getElementById('status');
  el.innerText=msg; el.className=cls||'';
}
</script>
</body></html>"""

# ---------------------------------------------------------------------------
# Reuse DEM functions from dem_selector
# ---------------------------------------------------------------------------
STAC_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/"
    "ch.swisstopo.swissalti3d/items"
)

def query_stac_tiles(west, south, east, north, resolution="2"):
    import re
    all_tiles = []
    url = f"{STAC_URL}?bbox={west},{south},{east},{north}&limit=100"
    while url:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"[ERROR] STAC query failed: {e}", file=sys.stderr)
            break
        for feature in data.get("features", []):
            for asset_key, asset in feature.get("assets", {}).items():
                if f"_{resolution}_2056_5728.tif" in asset_key:
                    all_tiles.append({"name": asset_key, "href": asset["href"]})
                    break
        url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                url = link["href"]
                break

    # Deduplicate: same tile coordinates (EEEE-NNNN) appear with different years.
    # Keep only the latest year per location.
    best = {}  # key: "EEEE-NNNN" -> tile with highest year
    for t in all_tiles:
        m = re.search(r'swissalti3d_(\d{4})_(\d{4}-\d{4})_', t["name"])
        if m:
            year = int(m.group(1))
            coords = m.group(2)
            if coords not in best or year > best[coords]["year"]:
                best[coords] = {"year": year, **t}
        else:
            best[t["name"]] = {"year": 0, **t}

    tiles = [{"name": v["name"], "href": v["href"]} for v in best.values()]
    return tiles

def download_tiles(tiles, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    downloaded = failed = 0
    for i, tile in enumerate(tiles, 1):
        filepath = os.path.join(output_dir, tile["name"])
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            downloaded += 1; continue
        print(f"  [{i}/{len(tiles)}] Downloading {tile['name']} ...")
        try:
            urllib.request.urlretrieve(tile["href"], filepath)
            downloaded += 1
        except Exception as e:
            print(f"    FAILED: {e}"); failed += 1
    return downloaded, failed

def wgs84_to_lv95(west, south, east, north):
    tf = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    e_min, n_min = tf.transform(west, south)
    e_max, n_max = tf.transform(east, north)
    return e_min, n_min, e_max, n_max

def wgs84_point_to_lv95(lng, lat):
    tf = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    e, n = tf.transform(lng, lat)
    return e, n

def lv95_to_wgs84_point(e, n):
    tf = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lng, lat = tf.transform(e, n)
    return lng, lat

def merge_and_crop(output_dir, e_min, n_min, e_max, n_max, nodata=-9999.0):
    tif_files = sorted(glob.glob(os.path.join(output_dir, "swissalti3d_*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {output_dir}")
    srcs = [rasterio.open(f) for f in tif_files]
    mosaic, mosaic_transform = rasterio_merge(srcs, nodata=nodata)
    profile = srcs[0].profile.copy()
    for s in srcs: s.close()
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                   transform=mosaic_transform, count=1, nodata=nodata,
                   compress="lzw", tiled=True, dtype="float32")
    merged_path = os.path.join(output_dir, "dem_merged.tif")
    with rasterio.open(merged_path, "w", **profile) as dst:
        dst.write(mosaic[0].astype(np.float32), 1)
    cropped_path = os.path.join(output_dir, "dem_cropped.tif")
    with rasterio.open(merged_path) as src:
        b = src.bounds
        cl = max(e_min, b.left); cb = max(n_min, b.bottom)
        cr = min(e_max, b.right); ct = min(n_max, b.top)
        win = from_bounds(cl, cb, cr, ct, transform=src.transform)
        arr = src.read(1, window=win).astype(np.float32)
        crop_tf = src.window_transform(win)
        cp = src.profile.copy()
        cp.update(height=arr.shape[0], width=arr.shape[1], transform=crop_tf,
                  count=1, nodata=nodata, compress="lzw", tiled=True, dtype="float32")
        with rasterio.open(cropped_path, "w", **cp) as dst:
            dst.write(arr, 1)
    # cleanup
    for f in tif_files: os.remove(f)
    if os.path.exists(merged_path): os.remove(merged_path)
    w_m = cr - cl; h_m = ct - cb
    return cropped_path, w_m, h_m

def register_custom_dem(name, dem_bytes, dem_root, max_size_m=3000.0):
    """Register a user-uploaded GeoTIFF as `dem/<name>/dem_cropped.tif`.

    The DEM must be in a projected CRS in metres (any — LV95, UTM, ETRS89/LAEA,
    etc.). dem_prep.py is CRS-agnostic. We extract bounds + centroid in the
    DEM's own CRS, write selection.json (the `center_lv95` key is a misnomer —
    it actually carries the pivot in the DEM's CRS), and render a hillshade PNG
    for the frontend preview.

    Returns:
      {bounds_crs:[W,S,E,N], width_m, height_m, crs_str, png_base64}
    """
    import base64
    import io
    if rasterio is None:
        raise RuntimeError("rasterio not available — install rasterio to upload custom DEMs")
    output_dir = os.path.join(dem_root, name)
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, "dem_cropped.tif")
    # Stage to a temp path, validate, then promote on success.
    tmp_path = target + ".upload"
    with open(tmp_path, "wb") as f:
        f.write(dem_bytes)
    try:
        with rasterio.open(tmp_path) as src:
            if src.crs is None:
                raise ValueError("DEM has no CRS — must be a projected GeoTIFF (LV95, UTM, ...)")
            crs = src.crs
            # Reject only truly geographic CRSes (lat/lng in degrees).
            # swissalti3d tiles use LOCAL_CS["CH1903+ / LV95", UNIT["metre", 1]],
            # which rasterio reports as is_projected=False AND is_geographic=False
            # — that's fine, the units are metres.
            try:
                is_geographic = bool(crs.is_geographic)
            except Exception:
                is_geographic = False
            if is_geographic:
                raise ValueError(f"DEM CRS '{crs.to_string()}' is geographic (degrees) — must be in metres")
            try:
                unit = (crs.linear_units or "").lower()
            except Exception:
                unit = ""
            # Accept 'metre', 'meter', 'unknown' (LOCAL_CS often reports 'unknown'),
            # or empty. Reject only explicitly non-metric units (foot, link, ...).
            if unit and unit != "unknown" and "met" not in unit:
                raise ValueError(f"DEM CRS '{crs.to_string()}' uses '{unit}' — must be metres")
            b = src.bounds
            w_m = float(b.right - b.left)
            h_m = float(b.top - b.bottom)
            if w_m <= 0 or h_m <= 0:
                raise ValueError("DEM has zero or negative extent")
            if w_m > max_size_m + 1.0 or h_m > max_size_m + 1.0:
                raise ValueError(
                    f"DEM extent {w_m:.0f}x{h_m:.0f} m exceeds max {max_size_m:.0f} m. "
                    f"Crop in QGIS/GDAL before upload."
                )
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            crs_str = crs.to_string()
            cx = float((b.left + b.right) / 2)
            cy = float((b.top + b.bottom) / 2)
        # Promote tmp -> dem_cropped.tif
        if os.path.exists(target):
            os.remove(target)
        os.replace(tmp_path, target)
    except Exception:
        try: os.remove(tmp_path)
        except OSError: pass
        raise

    # Hillshade PNG preview — simple Lambertian with sun azimuth=315°, alt=45°.
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("DEM contains only NoData")
    z = np.where(finite, arr, np.nanmean(arr[finite]))
    # Pixel size in metres (use abs for safety).
    px = w_m / arr.shape[1]
    py = h_m / arr.shape[0]
    dz_dy, dz_dx = np.gradient(z, py, px)
    az_rad = math.radians(315.0)
    alt_rad = math.radians(45.0)
    aspect = np.arctan2(dz_dy, -dz_dx)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    shaded = (np.sin(alt_rad) * np.cos(slope)
              + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    shaded = np.clip(shaded, 0.0, 1.0)
    # Combine with normalised elevation for a hint of colour.
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    z_norm = (z - z_min) / max(z_max - z_min, 1e-6)
    # Simple terrain colormap: low=green, mid=brown, high=white.
    r = (0.30 + 0.55 * z_norm + 0.10 * (1 - z_norm)) * shaded
    g = (0.55 + 0.20 * z_norm + 0.20 * (1 - z_norm)) * shaded
    bch = (0.30 + 0.65 * z_norm) * shaded
    rgb = np.stack([r, g, bch], axis=-1)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    try:
        from PIL import Image  # type: ignore
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        raise RuntimeError(f"PNG encoding failed (Pillow needed): {e}")

    sel = {
        "center_lv95": {"E": round(cx, 3), "N": round(cy, 3)},  # historical key — value is in DEM CRS
        "domain_name": name,
        "custom_dem": True,
        "crs": crs_str,
        "bounds_crs": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
    }
    with open(os.path.join(output_dir, "selection.json"), "w") as f:
        json.dump(sel, f, indent=2)

    return {
        "bounds_crs": [float(b.left), float(b.bottom), float(b.right), float(b.top)],
        "width_m": w_m,
        "height_m": h_m,
        "crs": crs_str,
        "png_base64": png_b64,
    }


def create_flat_ground_stl(x_min, x_max, y_min, y_max, output_path):
    verts = np.array([[x_min,y_min,0],[x_max,y_min,0],[x_max,y_max,0],[x_min,y_max,0]], dtype=np.float64)
    faces = np.array([[0,1,2],[0,2,3]])
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.export(str(output_path))

# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------
def flatten_ground_at_structures(ground_stl_path, struct_bounds, struct_z_terrains, margin=2.0):
    """Flatten the ground STL within each structure's footprint + margin.

    Sets ground vertices to the exact z_terrain that place_structure sampled,
    so the structure base sits perfectly on the flattened ground.
    """
    if not struct_bounds or trimesh is None:
        return

    ground = trimesh.load_mesh(str(ground_stl_path))
    verts = ground.vertices.copy()
    modified = False

    for sb, z_terrain in zip(struct_bounds, struct_z_terrains):
        # Force square flattening region (use max extent in x/y)
        cx = (sb["min"][0] + sb["max"][0]) / 2
        cy = (sb["min"][1] + sb["max"][1]) / 2
        half_ext = max(sb["max"][0] - sb["min"][0], sb["max"][1] - sb["min"][1]) / 2 + margin
        x_min = cx - half_ext
        x_max = cx + half_ext
        y_min = cy - half_ext
        y_max = cy + half_ext

        mask = ((verts[:, 0] >= x_min) & (verts[:, 0] <= x_max) &
                (verts[:, 1] >= y_min) & (verts[:, 1] <= y_max))

        if mask.sum() > 0:
            verts[mask, 2] = z_terrain
            modified = True
            print(f"[BUILD] Flattened ground at [{x_min:.0f},{x_max:.0f}]x[{y_min:.0f},{y_max:.0f}] -> z={z_terrain:.2f}m ({mask.sum()} vertices)")

    if modified:
        ground.vertices = verts
        ground.export(str(ground_stl_path))


def build_domain(domain_name, domain_size, wind_from, flat_terrain,
                 struct_list, center_latlng, dem_root, out_root,
                 uref=10.0, zref=20.0, z0=0.1, flatten_ground=True,
                 grid=None, z_top_offset=None,
                 placement_max_slope_deg=45.0,
                 placement_require_all_structures=False,
                 placement_align_inclined_to_slope=False,
                 placement_slope_align_min_deg=20.0,
                 placement_slope_align_jitter_deg=15.0):
    """Run the full pipeline: DEM prep → place structures → finalize."""
    dem_dir = Path(dem_root) / domain_name
    out_dir = Path(out_root) / domain_name
    prep_dir = dem_dir / "prep"
    placed_dir = dem_dir / "placed"

    prep_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tri_dir = out_dir / "constant" / "triSurface"
    tri_dir.mkdir(parents=True, exist_ok=True)

    # --- All terrain (real DEM or lake flat) goes through the same path ---
    dem_tif = dem_dir / "dem_cropped.tif"
    if not dem_tif.exists():
        raise FileNotFoundError(f"DEM not found: {dem_tif}")

    # Read center from selection.json
    sel_json = dem_dir / "selection.json"
    pivot_args = []
    if sel_json.exists():
        sel = json.load(open(sel_json))
        c = sel.get("center_lv95", {})
        if "E" in c and "N" in c:
            pivot_args = ["--pivot-x", str(c["E"]), "--pivot-y", str(c["N"])]

    # 1) dem_prep
    print(f"[BUILD] Running dem_prep (wind_from={wind_from}, size={domain_size}m)...")
    cmd = [
        sys.executable, str(_DOMAIN_PREP_DIR / "dem_prep.py"),
        "--tifs", str(dem_tif),
        "--wind-from", str(wind_from),
        "--out-dir", str(prep_dir),
        "--width", str(domain_size),
        "--height", str(domain_size),
        "--stl",
    ] + pivot_args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dem_prep failed: {r.stderr}")
    print(r.stdout[-200:] if len(r.stdout) > 200 else r.stdout)

    # Rename terrain.stl -> ground.stl
    terrain_stl = prep_dir / "terrain.stl"
    ground_stl = prep_dir / "ground.stl"
    if terrain_stl.exists():
        terrain_stl.rename(ground_stl)

    # 2) Place structures (if any)
    struct_stl_path = None
    if struct_list:
        # Build positions.json. Custom-DEM mode supplies crs_x/crs_y directly
        # (its CRS may not be LV95). Swiss path still goes via lat/lng → LV95.
        positions = []
        for s in struct_list:
            if s.get("crs_x") is not None and s.get("crs_y") is not None:
                e, n = float(s["crs_x"]), float(s["crs_y"])
            else:
                e, n = wgs84_point_to_lv95(s["lng"], s["lat"])
            positions.append({"crs_x": e, "crs_y": n, "label": s.get("label", "s")})

        pos_file = dem_dir / "positions.json"
        with open(pos_file, "w") as f:
            json.dump(positions, f, indent=2)

        # All same structure type? Use first type for now
        stl_file = ROOT / "single_stl" / struct_list[0]["type"]
        yaw = struct_list[0].get("yaw", 0)

        print(f"[BUILD] Placing {len(struct_list)} structure(s)...")
        cmd = [
            sys.executable, str(_DOMAIN_PREP_DIR / "place_structure.py"),
            "--dem-dir", str(prep_dir),
            "--stl", str(stl_file),
            "--coords-file", str(pos_file),
            "--yaw-deg", str(yaw),
            "--out-dir", str(placed_dir),
            "--max-slope-deg", str(placement_max_slope_deg),
            "--min-boundary-clearance", "0",
        ]
        if placement_require_all_structures:
            cmd.append("--require-all-structures")
        if placement_align_inclined_to_slope:
            cmd.extend([
                "--align-inclined-to-slope",
                "--slope-align-min-deg", str(placement_slope_align_min_deg),
                "--slope-align-jitter-deg", str(placement_slope_align_jitter_deg),
            ])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"place_structure failed: {r.stderr}")
        print(r.stdout[-300:] if len(r.stdout) > 300 else r.stdout)

        struct_stl_path = placed_dir / "structures_placed.stl"

    # 2a-grid) Place grid of structures
    if grid is not None:
        grid_center = grid.get("center", {})
        gc_has_crs = (grid_center and grid_center.get("crs_x") is not None
                      and grid_center.get("crs_y") is not None)
        gc_has_ll = (grid_center and grid_center.get("lat") and grid_center.get("lng"))
        if gc_has_crs or gc_has_ll:
            grid_stl = ROOT / "single_stl" / grid["type"]
            if gc_has_crs:
                grid_e, grid_n = float(grid_center["crs_x"]), float(grid_center["crs_y"])
            else:
                grid_e, grid_n = wgs84_point_to_lv95(grid_center["lng"], grid_center["lat"])

            grid_str = f"{grid['rows']}x{grid['cols']}"
            print(f"[BUILD] Placing {grid_str} grid of structures...")

            placed_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(_DOMAIN_PREP_DIR / "place_structure.py"),
                "--dem-dir", str(prep_dir),
                "--stl", str(grid_stl),
                "--crs-xy", str(grid_e), str(grid_n),
                "--grid", grid_str,
                "--grid-spacing", str(grid["spacing_x"]), str(grid["spacing_y"]),
                "--grid-spacing-mode", "edge",
                "--grid-yaw-deg", str(grid.get("grid_yaw", 0)),
                "--yaw-deg", str(grid.get("struct_yaw", 0)),
                "--out-dir", str(placed_dir),
                "--max-slope-deg", str(placement_max_slope_deg),
                "--min-boundary-clearance", "0",
            ]
            if placement_require_all_structures:
                cmd.append("--require-all-structures")
            if placement_align_inclined_to_slope:
                cmd.extend([
                    "--align-inclined-to-slope",
                    "--slope-align-min-deg", str(placement_slope_align_min_deg),
                    "--slope-align-jitter-deg", str(placement_slope_align_jitter_deg),
                ])
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"place_structure (grid) failed: {r.stderr}")
            print(r.stdout[-300:] if len(r.stdout) > 300 else r.stdout)

            struct_stl_path = placed_dir / "structures_placed.stl"
            # Override struct_list for downstream processing
            n_grid = grid["rows"] * grid["cols"]
            struct_list = [{"type": grid["type"], "yaw": grid.get("struct_yaw", 0)}] * n_grid

    # 2b) Flatten ground around structures (before z-shift)
    if grid is not None:
        flatten_request = bool(grid.get("flatten", False))
    else:
        flatten_request = flatten_ground
    if flatten_request and struct_list and not flat_terrain:
        ground_stl = prep_dir / "ground.stl"
        domain_spec = placed_dir / "domain_spec.json"
        if ground_stl.exists() and domain_spec.exists():
            with open(domain_spec) as f:
                spec = json.load(f)
            orig_stl = ROOT / "single_stl" / struct_list[0]["type"]
            if orig_stl.exists():
                orig = trimesh.load_mesh(str(orig_stl))
                # Center the flatten region on the structure's BASE footprint, not
                # the full bbox. For asymmetric structures (e.g. wind turbines whose
                # blades swing far past the tower), bbox-centered flattening misses
                # the actual ground contact point.
                bnd = orig.bounds
                cx_full = (bnd[0][0] + bnd[1][0]) / 2.0
                cy_full = (bnd[0][1] + bnd[1][1]) / 2.0
                z_min = bnd[0][2]
                z_max = bnd[1][2]
                base_slab = min(0.05 * (z_max - z_min), 2.0)
                v = orig.vertices
                base_mask = v[:, 2] <= z_min + base_slab
                if base_mask.sum() > 0:
                    bx_min, bx_max = float(v[base_mask, 0].min()), float(v[base_mask, 0].max())
                    by_min, by_max = float(v[base_mask, 1].min()), float(v[base_mask, 1].max())
                    base_cx = (bx_min + bx_max) / 2.0
                    base_cy = (by_min + by_max) / 2.0
                    base_half_x = (bx_max - bx_min) / 2.0
                    base_half_y = (by_max - by_min) / 2.0
                else:
                    base_cx, base_cy = cx_full, cy_full
                    base_half_x = float(orig.extents[0]) / 2
                    base_half_y = float(orig.extents[1]) / 2
                # Offset of base centroid from bbox center, in unrotated STL frame.
                # place_structure.py centers the mesh on its bbox before applying yaw,
                # so this offset rotates with the structure's yaw.
                off_x = base_cx - cx_full
                off_y = base_cy - cy_full
                flat_bounds = []
                z_terrains = []
                for s_info in spec.get("structures", []):
                    lx = s_info["placement_local"]["x"]
                    ly = s_info["placement_local"]["y"]
                    zt = s_info["z_terrain"]
                    yaw_deg = float(s_info.get("yaw_deg", 0.0))
                    ang = math.radians(yaw_deg)
                    cos_a = math.cos(ang)
                    sin_a = math.sin(ang)
                    world_dx = off_x * cos_a - off_y * sin_a
                    world_dy = off_x * sin_a + off_y * cos_a
                    rot_half_x = abs(base_half_x * cos_a) + abs(base_half_y * sin_a)
                    rot_half_y = abs(base_half_x * sin_a) + abs(base_half_y * cos_a)
                    cx_world = lx + world_dx
                    cy_world = ly + world_dy
                    flat_bounds.append({
                        "min": [cx_world - rot_half_x, cy_world - rot_half_y, 0],
                        "max": [cx_world + rot_half_x, cy_world + rot_half_y, 0],
                    })
                    z_terrains.append(zt)
                flatten_ground_at_structures(str(ground_stl), flat_bounds, z_terrains, margin=2.0)

    # 3) Finalize (z-shift)
    print(f"[BUILD] Finalizing domain...")
    cmd = [
        sys.executable, str(_DOMAIN_PREP_DIR / "finalize_domain.py"),
        "--prep-dir", str(prep_dir),
        "--out-dir", str(out_dir),
    ]
    if struct_stl_path and struct_stl_path.exists():
        cmd.extend(["--structure-stls", str(struct_stl_path)])

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"finalize_domain failed: {r.stderr}")
    print(r.stdout[-200:] if len(r.stdout) > 200 else r.stdout)

    # Compute per-structure bounds from known placement positions + STL size
    # Load the original STL to get its extents, then offset per placement
    struct_bounds = []
    if struct_list:
        # Get original structure extents (before placement)
        stl_file = ROOT / "single_stl" / struct_list[0]["type"]
        if stl_file.exists():
            orig = trimesh.load_mesh(str(stl_file))
            orig_ext = orig.extents  # [dx, dy, dz]
            half_x = float(orig_ext[0]) / 2
            half_y = float(orig_ext[1]) / 2
            struct_h = float(orig_ext[2])

        # Read placed structure to get actual z values per position
        struct_stl_final = tri_dir / "structure.stl"
        placed_mesh = None
        if struct_stl_final.exists():
            placed_mesh = trimesh.load_mesh(str(struct_stl_final))

        # For each structure, compute bounds from the placed positions file
        placed_dir = Path(dem_root) / domain_name / "placed"
        domain_spec = placed_dir / "domain_spec.json"
        if domain_spec.exists():
            with open(domain_spec) as f:
                spec = json.load(f)
            for s_info in spec.get("structures", []):
                lx = s_info["placement_local"]["x"]
                ly = s_info["placement_local"]["y"]
                zt = s_info["z_terrain"]
                yaw_deg = float(s_info.get("yaw_deg", 0.0))
                # Apply z_offset from finalize (z_min shift)
                tf_path = tri_dir / "transform.json"
                z_off = 0.0
                if tf_path.exists():
                    with open(tf_path) as f:
                        tf_meta = json.load(f)
                    z_off = float(tf_meta.get("z_offset_applied", 0))
                zt_local = zt + z_off

                # Use the AABB measured from the actually placed/yawed mesh.
                # Rebuilding it from the source STL's unrotated extents makes
                # wind-rose plots falsely show structures rotating with wind.
                placed_bounds = s_info.get("placed_bounds")
                if (
                    isinstance(placed_bounds, list)
                    and len(placed_bounds) >= 2
                    and len(placed_bounds[0]) >= 3
                    and len(placed_bounds[1]) >= 3
                ):
                    xmin, ymin = float(placed_bounds[0][0]), float(placed_bounds[0][1])
                    xmax, ymax = float(placed_bounds[1][0]), float(placed_bounds[1][1])
                    zmin = float(placed_bounds[0][2]) + z_off
                    zmax = float(placed_bounds[1][2]) + z_off
                else:
                    angle = math.radians(yaw_deg)
                    rot_half_x = abs(half_x * math.cos(angle)) + abs(half_y * math.sin(angle))
                    rot_half_y = abs(half_x * math.sin(angle)) + abs(half_y * math.cos(angle))
                    xmin, xmax = lx - rot_half_x, lx + rot_half_x
                    ymin, ymax = ly - rot_half_y, ly + rot_half_y
                    zmin, zmax = zt_local, zt_local + struct_h

                angle = math.radians(yaw_deg)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                footprint_xy = []
                for dx, dy in (
                    (-half_x, -half_y),
                    (half_x, -half_y),
                    (half_x, half_y),
                    (-half_x, half_y),
                ):
                    footprint_xy.append([
                        lx + dx * cos_a - dy * sin_a,
                        ly + dx * sin_a + dy * cos_a,
                    ])
                struct_bounds.append({
                    "min": [xmin, ymin, zmin],
                    "max": [xmax, ymax, zmax],
                    "footprint_xy": footprint_xy,
                    "yaw_deg": yaw_deg,
                    "label": s_info.get("label", s_info.get("id", "")),
                })
        elif placed_mesh is not None:
            # Fallback: single structure, use full mesh bounds
            b = placed_mesh.bounds
            struct_bounds.append({
                "min": [float(b[0][0]), float(b[0][1]), float(b[0][2])],
                "max": [float(b[1][0]), float(b[1][1]), float(b[1][2])],
            })

    grid_info = None
    if grid is not None:
        grid_info = {
            "type": grid["type"],
            "rows": grid["rows"], "cols": grid["cols"],
            "spacing_x": grid["spacing_x"], "spacing_y": grid["spacing_y"],
            "struct_yaw": grid.get("struct_yaw", 0),
            "grid_yaw": grid.get("grid_yaw", 0),
        }

    meta = {"domain_name": domain_name, "flat_terrain": flat_terrain,
            "domain_size": [domain_size, domain_size],
            "wind_from": wind_from, "n_structures": len(struct_list),
            "ABL": {"Uref": uref, "Zref": zref, "z0": z0},
            "structure_bounds": struct_bounds,
            "grid": grid_info,
            "z_top_offset": z_top_offset}
    with open(tri_dir / "domain_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    return str(out_dir)


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    dem_root = "dem"
    out_root = "data_preparation/singlestructures"

    def log_message(self, fmt, *args): pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path.startswith("/lv95_to_wgs84"):
            # Parse query params
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            try:
                e = float(qs["e"][0])
                n = float(qs["n"][0])
                lng, lat = lv95_to_wgs84_point(e, n)
                self._json_response({"lat": lat, "lng": lng})
            except Exception as ex:
                self._json_response({"error": str(ex)})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        if self.path == "/download_dem":
            try:
                west, south, east, north = body["west"], body["south"], body["east"], body["north"]
                name = body.get("domain_name", "unnamed")
                res = body.get("resolution", "2")
                output_dir = os.path.join(self.dem_root, name)

                print(f"\n[DEM] {name}: W={west:.5f} S={south:.5f} E={east:.5f} N={north:.5f}")
                tiles = query_stac_tiles(west, south, east, north, resolution=res)
                if not tiles:
                    self._json_response({"success": False, "error": "No tiles found"})
                    return

                downloaded, failed = download_tiles(tiles, output_dir)
                if downloaded == 0:
                    self._json_response({"success": False, "error": "All downloads failed"})
                    return

                e_min, n_min, e_max, n_max = wgs84_to_lv95(west, south, east, north)
                _, w_m, h_m = merge_and_crop(output_dir, e_min, n_min, e_max, n_max)

                # Save selection center
                center_e = (e_min + e_max) / 2
                center_n = (n_min + n_max) / 2
                sel = {"center_lv95": {"E": round(center_e, 1), "N": round(center_n, 1)},
                       "domain_name": name}
                with open(os.path.join(output_dir, "selection.json"), "w") as f:
                    json.dump(sel, f, indent=2)

                self._json_response({"success": True, "dem_size": f"{w_m:.0f}x{h_m:.0f} m",
                                     "downloaded": downloaded})
            except Exception as e:
                print(f"[ERROR] {e}")
                self._json_response({"success": False, "error": str(e)})

        elif self.path == "/build":
            try:
                result_dir = build_domain(
                    domain_name=body["domain_name"],
                    domain_size=body["domain_size"],
                    wind_from=body["wind_from"],
                    flat_terrain=body.get("flat_terrain", False),
                    struct_list=body.get("structures", []),
                    center_latlng=body.get("center"),
                    dem_root=self.dem_root,
                    out_root=self.out_root,
                    uref=float(body.get("uref", 10)),
                    zref=float(body.get("zref", 20)),
                    z0=float(body.get("z0", 0.1)),
                    flatten_ground=bool(body.get("flatten_ground", True)),
                    grid=body.get("grid"),
                    z_top_offset=body.get("z_top_offset"),
                )
                self._json_response({"success": True, "output_dir": result_dir})
            except Exception as e:
                import traceback; traceback.print_exc()
                self._json_response({"success": False, "error": str(e)})
        else:
            self.send_error(404)

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    p = argparse.ArgumentParser(description="All-in-one domain builder")
    p.add_argument("--dem-root", default="dem")
    p.add_argument("--out-root", default="data_preparation/singlestructures")
    p.add_argument("--port", type=int, default=8778)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    Handler.dem_root = args.dem_root
    Handler.out_root = args.out_root

    server = http.server.HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"

    print(f"Domain Builder running at {url}")
    print(f"  DEM root: {args.dem_root}/")
    print(f"  Output:   {args.out_root}/")
    print(f"Press Ctrl+C to stop.\n")

    if not args.no_browser:
        def _open():
            if "microsoft" in os.uname().release.lower():
                os.system(f'cmd.exe /c start {url}')
            else:
                webbrowser.open(url)
        threading.Timer(0.5, _open).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
