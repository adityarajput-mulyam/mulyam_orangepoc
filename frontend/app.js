let map;
let baseTileLayer = null;
let drawnItems;
let currentPolygon = null;
let treeMarkersLayer = null;
let gapMarkersLayer = null;
let currentProvider = "google";

const TILE_URLS = {
  google: "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
  google_hybrid: "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
  esri: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
};

// Presets (Major Indian Citrus & Mandarin Belts)
const PRESETS = {
  nagpur: {
    lat: 21.2787,
    lon: 78.5866,
    zoom: 18,
    name: "Katol / Nagpur, Maharashtra (Nagpur Mandarin)",
    defaultPoly: [
      [78.5850, 21.2798],
      [78.5885, 21.2798],
      [78.5885, 21.2775],
      [78.5850, 21.2775],
      [78.5850, 21.2798]
    ]
  },
  warud: {
    lat: 21.4650,
    lon: 78.2670,
    zoom: 18,
    name: "Warud / Morshi, Vidarbha (Orange Groves)",
    defaultPoly: [
      [78.2650, 21.4665],
      [78.2690, 21.4665],
      [78.2690, 21.4635],
      [78.2650, 21.4635],
      [78.2650, 21.4665]
    ]
  },
  abohar: {
    lat: 30.1453,
    lon: 74.1993,
    zoom: 18,
    name: "Abohar, Punjab (Kinnow Citrus Belt)",
    defaultPoly: [
      [74.1970, 30.1465],
      [74.2015, 30.1465],
      [74.2015, 30.1438],
      [74.1970, 30.1438],
      [74.1970, 30.1465]
    ]
  },
  jhalawar: {
    lat: 24.5973,
    lon: 76.1612,
    zoom: 18,
    name: "Jhalawar, Rajasthan (Mandarin Orchards)",
    defaultPoly: [
      [76.1590, 24.5985],
      [76.1630, 24.5985],
      [76.1630, 24.5960],
      [76.1590, 24.5960],
      [76.1590, 24.5985]
    ]
  },
  coorg: {
    lat: 12.3375,
    lon: 75.8069,
    zoom: 18,
    name: "Coorg, Karnataka (Coorg Mandarin)",
    defaultPoly: [
      [75.8050, 12.3390],
      [75.8090, 12.3390],
      [75.8090, 12.3360],
      [75.8050, 12.3360],
      [75.8050, 12.3390]
    ]
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupEventListeners();
  currentPolygon = null;
  updateBoundaryStatus(false);
});

function setTileProvider(providerKey) {
  currentProvider = providerKey;
  const url = TILE_URLS[providerKey] || TILE_URLS.google;
  
  if (baseTileLayer) {
    map.removeLayer(baseTileLayer);
  }
  
  baseTileLayer = L.tileLayer(url, {
    attribution: providerKey.startsWith("google") ? "Imagery &copy; Google Maps" : "Tiles &copy; Esri",
    maxZoom: 20,
    subdomains: ['mt0', 'mt1', 'mt2', 'mt3']
  }).addTo(map);

  document.getElementById("zoomIndicator").innerText = `Zoom: 18 | Satellite: ${providerKey.toUpperCase()}`;
}

function initMap() {
  map = L.map("map", {
    center: [21.2787, 78.5866],
    zoom: 18,
    maxZoom: 20
  });

  setTileProvider("google");

  // Drawing tools layer
  drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  const drawControl = new L.Control.Draw({
    draw: {
      polygon: {
        allowIntersection: false,
        showArea: true,
        drawError: { color: "#e1e100", message: "<strong>Error:</strong> polygon cannot cross!" },
        shapeOptions: { color: "#ff7a18", weight: 3, fillOpacity: 0.2 }
      },
      rectangle: {
        shapeOptions: { color: "#ff7a18", weight: 3, fillOpacity: 0.2 }
      },
      circle: false,
      circlemarker: false,
      polyline: false,
      marker: false
    },
    edit: {
      featureGroup: drawnItems,
      remove: true
    }
  });
  map.addControl(drawControl);

  // Capture user drawn polygon
  map.on(L.Draw.Event.CREATED, (event) => {
    drawnItems.clearLayers();
    const layer = event.layer;
    drawnItems.addLayer(layer);
    
    // Extract [lon, lat] coordinates
    const latlngs = layer.getLatLngs()[0];
    currentPolygon = latlngs.map(pt => [pt.lng, pt.lat]);
    // Close polygon if open
    if (currentPolygon[0][0] !== currentPolygon[currentPolygon.length - 1][0] || 
        currentPolygon[0][1] !== currentPolygon[currentPolygon.length - 1][1]) {
      currentPolygon.push([currentPolygon[0][0], currentPolygon[0][1]]);
    }

    updateBoundaryStatus(true);
  });

  map.on(L.Draw.Event.DELETED, () => {
    currentPolygon = null;
    updateBoundaryStatus(false);
  });
}

function loadPreset(presetKey) {
  const preset = PRESETS[presetKey];
  if (!preset) return;

  map.setView([preset.lat, preset.lon], preset.zoom);
  document.getElementById("coordInput").value = `${preset.lat}, ${preset.lon}`;
  
  // Clear any existing drawings so user starts fresh
  drawnItems.clearLayers();
  currentPolygon = null;
  if (treeMarkersLayer) {
    map.removeLayer(treeMarkersLayer);
    treeMarkersLayer = null;
  }
  if (gapMarkersLayer) {
    map.removeLayer(gapMarkersLayer);
    gapMarkersLayer = null;
  }
  updateBoundaryStatus(false);
}

function updateBoundaryStatus(hasBoundary) {
  const chip = document.getElementById("boundaryStatus");
  if (hasBoundary) {
    chip.className = "status-chip chip-active";
    chip.innerText = "Farm boundary active. Ready to run analysis.";
  } else {
    chip.className = "status-chip chip-neutral";
    chip.innerText = "No boundary drawn. Outline your orchard plot using map tools.";
  }
}

let lastAnalysisData = null;

let overlayVisible = true;
let isEditMode = false;

function setupEventListeners() {
  const slider = document.getElementById("sensitivitySlider");
  const sliderLbl = document.getElementById("sensitivityLabel");

  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    if (val <= 0.3) sliderLbl.innerText = "Mature (Tighter)";
    else if (val >= 0.7) sliderLbl.innerText = "Young (Sensitive)";
    else sliderLbl.innerText = "Normal";
  });

  document.getElementById("providerSelect").addEventListener("change", (e) => {
    setTileProvider(e.target.value);
  });

  document.getElementById("presetSelect").addEventListener("change", (e) => {
    loadPreset(e.target.value);
  });

  document.getElementById("jumpCoordBtn").addEventListener("click", () => {
    const raw = document.getElementById("coordInput").value.trim();
    const parts = raw.split(",").map(p => parseFloat(p.trim()));
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      map.setView([parts[0], parts[1]], 18);
    } else {
      alert("Please enter valid GPS coordinates: 'latitude, longitude'");
    }
  });

  // Toggle Overlay (Blink) to inspect raw satellite imagery
  const toggleBtn = document.getElementById("toggleOverlayBtn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      overlayVisible = !overlayVisible;
      if (treeMarkersLayer) {
        if (overlayVisible) map.addLayer(treeMarkersLayer);
        else map.removeLayer(treeMarkersLayer);
      }
      if (gapMarkersLayer) {
        if (overlayVisible) map.addLayer(gapMarkersLayer);
        else map.removeLayer(gapMarkersLayer);
      }
      toggleBtn.innerText = overlayVisible ? "👁️ Hide Rings" : "👁️ Show Rings";
    });
  }

  // Edit Mode: Click dot to delete, click map to add
  const editBtn = document.getElementById("editModeBtn");
  if (editBtn) {
    editBtn.addEventListener("click", () => {
      isEditMode = !isEditMode;
      document.getElementById("editModeStatus").style.display = isEditMode ? "inline-block" : "none";
      editBtn.classList.toggle("btn-primary", isEditMode);
      editBtn.classList.toggle("btn-outline-danger", !isEditMode);
      editBtn.innerText = isEditMode ? "💾 Done Editing" : "✏️ Edit Markings";
    });
  }

  map.on("click", (e) => {
    if (!isEditMode || !lastAnalysisData) return;
    // Add a tree marker manually
    const newTree = {
      id: lastAnalysisData.trees.length + 1,
      gps: { lat: e.latlng.lat, lon: e.latlng.lng },
      canopy_diameter_m: 3.5,
      health: { health_grade: "Good", score: 85 }
    };
    lastAnalysisData.trees.push(newTree);
    addSingleTreeMarker(newTree);
    recalculateDashboard();
  });

  const attachReset = (id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", resetMarkings);
  };
  attachReset("resetBtn");
  attachReset("mapResetBtn");
  attachReset("resetDashBtn");

  document.getElementById("analyzeBtn").addEventListener("click", runAnalysis);

  document.getElementById("exportCsvBtn").addEventListener("click", () => {
    if (!lastAnalysisData) return;
    exportFarmCSV(lastAnalysisData);
  });
}

function addSingleTreeMarker(t) {
  if (!treeMarkersLayer) {
    treeMarkersLayer = L.layerGroup().addTo(map);
  }
  const circle = L.circleMarker([t.gps.lat, t.gps.lon], {
    radius: 3.5,
    color: "#00e676",
    fillColor: "#00e676",
    fillOpacity: 0.35,
    weight: 1.5
  });

  circle.on("click", (e) => {
    if (isEditMode) {
      L.DomEvent.stopPropagation(e);
      treeMarkersLayer.removeLayer(circle);
      lastAnalysisData.trees = lastAnalysisData.trees.filter(item => item.id !== t.id);
      recalculateDashboard();
    }
  });

  circle.bindPopup(`<b>Orange Tree #${t.id}</b><br>Health: ${t.health.health_grade} (${t.health.score}%)<br>Canopy: ${t.canopy_diameter_m}m<br>GPS: ${t.gps.lat.toFixed(5)}, ${t.gps.lon.toFixed(5)}<br><em>(In Edit Mode: click dot to delete)</em>`);
  treeMarkersLayer.addLayer(circle);
}

function recalculateDashboard() {
  if (!lastAnalysisData) return;
  const count = lastAnalysisData.trees.length;
  const area = lastAnalysisData.summary.farm_area_ha;
  const density = area > 0 ? (count / area).toFixed(1) : 0;
  const yieldTons = (count * 55 / 1000).toFixed(2);
  const rev = Math.round(yieldTons * 1000 * 35);

  document.getElementById("kpiTrees").innerText = count.toLocaleString();
  document.getElementById("kpiDensity").innerText = `Density: ${density} trees/ha`;
  document.getElementById("valYield").innerText = `${yieldTons} tons`;
  document.getElementById("valRevenue").innerText = `₹${rev.toLocaleString('en-IN')}`;
}

function resetMarkings() {
  // Clear map detection layers
  if (treeMarkersLayer) {
    map.removeLayer(treeMarkersLayer);
    treeMarkersLayer = null;
  }
  if (gapMarkersLayer) {
    map.removeLayer(gapMarkersLayer);
    gapMarkersLayer = null;
  }

  lastAnalysisData = null;
  document.getElementById("exportCsvBtn").style.display = "none";
  const rDash = document.getElementById("resetDashBtn");
  if (rDash) rDash.style.display = "none";
  const sBox = document.getElementById("sanityBox");
  if (sBox) sBox.style.display = "none";

  // Reset KPI cards
  document.getElementById("kpiTrees").innerText = "0";
  document.getElementById("kpiDensity").innerText = "Density: 0 trees/ha";
  document.getElementById("kpiArea").innerText = "0.00 ha";
  document.getElementById("kpiGaps").innerText = "0 spots";
  document.getElementById("kpiHealth").innerText = "--%";
  document.getElementById("kpiHealthLabel").innerText = "Pending Analysis";
  document.getElementById("valYield").innerText = "0.00 tons";
  document.getElementById("valRevenue").innerText = "₹0.00";
  document.getElementById("analysisTimestamp").innerText = "Markings cleared";

  // Reset preview
  const preview = document.getElementById("previewContainer");
  preview.innerHTML = `
    <div class="empty-preview">
      <div class="empty-icon">🛰️</div>
      <p>Draw a plot boundary on the map and click <strong>Run Tree Detection</strong> to generate an analysis.</p>
    </div>
  `;

  // Clear seasonal table
  document.getElementById("seasonalTableBody").innerHTML = `
    <tr><td><strong>Early Harvest (Oct - Dec)</strong></td><td><span class="badge" style="margin:0">25%</span></td><td>0.00 tons</td><td><strong style="color:#2ea043">₹0.00</strong></td></tr>
    <tr><td><strong>Peak Harvest (Jan - Mar)</strong></td><td><span class="badge" style="margin:0">55%</span></td><td>0.00 tons</td><td><strong style="color:#2ea043">₹0.00</strong></td></tr>
    <tr><td><strong>Late Season (Apr - May)</strong></td><td><span class="badge" style="margin:0">20%</span></td><td>0.00 tons</td><td><strong style="color:#2ea043">₹0.00</strong></td></tr>
  `;
}

function exportFarmCSV(data) {
  const summary = data.summary;
  const trees = data.trees;
  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Tree_ID,Latitude,Longitude,Health_Grade,Health_Score_Pct,Canopy_Diameter_m,Confidence\n";

  trees.forEach(t => {
    const row = [
      t.id,
      t.gps.lat.toFixed(6),
      t.gps.lon.toFixed(6),
      t.health.health_grade,
      t.health.score,
      t.canopy_diameter_m,
      t.confidence
    ].join(",");
    csvContent += row + "\n";
  });
  
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `citrus_farm_trees_report_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

async function runAnalysis() {
  if (!currentPolygon || currentPolygon.length < 3) {
    alert("Please draw or select a farm plot boundary on the map first.");
    return;
  }

  const analyzeBtn = document.getElementById("analyzeBtn");
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing Satellite Imagery...';

  const sens = parseFloat(document.getElementById("sensitivitySlider").value) || 0.5;

  try {
    const response = await fetch("/api/analyze-plot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        coordinates: currentPolygon,
        conf_threshold: 0.12,
        farm_name: "Citrus Orchard Analysis",
        market_price_per_kg: 35.0,
        zoom_level: 18,
        provider: currentProvider,
        sensitivity: sens
      })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Server analysis error");
    }

    const data = await response.json();
    lastAnalysisData = data;
    document.getElementById("exportCsvBtn").style.display = "inline-flex";
    const rDash = document.getElementById("resetDashBtn");
    if (rDash) rDash.style.display = "inline-flex";
    renderResults(data);

  } catch (error) {
    console.error("Analysis Error:", error);
    alert("Analysis Failed: " + error.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span> Run Tree Detection & Analytics';
  }
}

function renderResults(data) {
  const summary = data.summary;
  const economics = data.economics;

  // Update KPIs
  document.getElementById("kpiTrees").innerText = summary.total_orange_trees.toLocaleString();
  document.getElementById("kpiDensity").innerText = `Density: ${summary.plantation_density_trees_per_ha} trees/ha`;
  document.getElementById("kpiArea").innerText = `${summary.farm_area_ha} ha`;
  document.getElementById("kpiGaps").innerText = `${summary.detected_gaps_count} spots`;
  document.getElementById("kpiHealth").innerText = `${summary.average_health_score}%`;
  document.getElementById("kpiHealthLabel").innerText = summary.average_health_score >= 80 ? "Healthy / High Vigor" : "Moderate Vigor";

  // Orchard Density Classification Badge
  const dBadge = document.getElementById("densityBadge");
  if (dBadge) {
    const d = summary.plantation_density_trees_per_ha;
    if (d >= 320) {
      dBadge.innerText = "High Density";
      dBadge.className = "metric-pill";
    } else if (d >= 180) {
      dBadge.innerText = "Commercial Grid";
      dBadge.className = "metric-pill";
    } else if (d > 0) {
      dBadge.innerText = "Spaced / Fallow";
      dBadge.className = "metric-pill neutral";
    } else {
      dBadge.innerText = "No Trees";
      dBadge.className = "metric-pill danger";
    }
  }

  // Economics
  document.getElementById("valYield").innerText = `${economics.total_yield_tons} tons`;
  document.getElementById("valRevenue").innerText = `₹${economics.total_production_value.toLocaleString('en-IN')}`;

  // Timestamp
  document.getElementById("analysisTimestamp").innerText = `Analyzed: ${new Date().toLocaleTimeString()}`;

  // Seasonal Table
  const tbody = document.getElementById("seasonalTableBody");
  tbody.innerHTML = "";
  for (const [season, details] of Object.entries(economics.seasonal_breakdown)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${season}</strong></td>
      <td><span class="badge" style="margin:0">${details.percentage}</span></td>
      <td>${details.yield_tons} tons</td>
      <td><strong style="color:#2ea043">₹${details.est_revenue.toLocaleString('en-IN')}</strong></td>
    `;
    tbody.appendChild(tr);
  }

  // Detection Preview Image
  const preview = document.getElementById("previewContainer");
  preview.innerHTML = `<img src="${data.image_overlay_base64}" alt="Citrus Tree Detection Overlay" />`;

  // Plot Tree Markers on Leaflet Map
  if (treeMarkersLayer) map.removeLayer(treeMarkersLayer);
  if (gapMarkersLayer) map.removeLayer(gapMarkersLayer);

  treeMarkersLayer = L.layerGroup();
  gapMarkersLayer = L.layerGroup();

  data.trees.forEach(t => {
    addSingleTreeMarker(t);
  });

  data.gaps.forEach((g, idx) => {
    // Distinct bare-soil gap marker
    const marker = L.circleMarker([g.gps.lat, g.gps.lon], {
      radius: 4,
      color: "#ff1744",
      fillColor: "transparent",
      fillOpacity: 0,
      weight: 2,
      dashArray: "2, 2"
    }).bindPopup(`<b>Verified Missing Tree Gap #${idx+1}</b><br>Bare soil confirmed in planting row`);
    gapMarkersLayer.addLayer(marker);
  });

  map.addLayer(treeMarkersLayer);
  map.addLayer(gapMarkersLayer);
}
