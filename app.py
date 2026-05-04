import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
import json
import os
import tempfile
import zipfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
import io
import re
from pathlib import Path
import streamlit.components.v1 as components

import google.generativeai as genai

genai.configure(api_key=st.secrets["api_key"])
def get_working_model():
    models = genai.list_models()
    usable = [
        m.name for m in models
        if "generateContent" in m.supported_generation_methods
    ]
    def score(name):
        name = name.lower()
        if "pro" in name: return 3
        if "flash" in name: return 2
        return 1
    usable = sorted(usable, key=score, reverse=True)
    last_error = None
    for model_name in usable:
        try:
            model = genai.GenerativeModel(model_name)
            model.generate_content("test")
            return model
        except Exception as e:
            last_error = e
            continue
    raise Exception(f"No usable model found. Last error: {last_error}")

model = get_working_model()

st.set_page_config(
    page_title="AI Cartographer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  :root {
    --cream:    #f7f3ec;
    --ivory:    #fdfaf5;
    --parchment:#ede8df;
    --border:   #d6cfc4;
    --border2:  #c4bdb3;

    --forest:   #2d5a3d;
    --forest2:  #3d7a55;
    --terra:    #b85c38;
    --terra2:   #d4704a;
    --ink:      #1e1e1e;
    --ink2:     #2c2c2c;
    --muted:    #7a7068;
    --muted2:   #9a9088;

    --surface:  #ffffff;
    --shadow:   rgba(45,90,61,0.10);
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--cream) !important;
    color: var(--ink) !important;
  }

  [data-testid="stSidebar"],
  [data-testid="collapsedControl"],
  section[data-testid="stSidebar"] { display: none !important; }

  div[data-testid="stAppViewContainer"] > .main > div {
    max-width: 1360px;
    margin: 0 auto;
    padding: 0 2rem;
  }

  .main .block-container {
    padding-top: 0 !important;
    padding-right: 1rem;
    margin: 0 auto;
  }

  .masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 0 1rem;
    border-bottom: 2px solid var(--ink2);
    margin-bottom: 1.6rem;
    position: relative;
  }
  .masthead::after {
    content: "";
    position: absolute;
    bottom: 3px;
    left: 0; right: 0;
    height: 1px;
    background: var(--border);
  }
  .masthead-left {
    display: flex;
    align-items: baseline;
    gap: 1rem;
  }
  .masthead-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.01em;
    line-height: 1;
  }
  .masthead-title em {
    font-style: italic;
    color: var(--forest);
  }
  .masthead-rule {
    width: 1px;
    height: 22px;
    background: var(--border2);
    display: inline-block;
  }
  .masthead-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 400;
    color: var(--muted);
    letter-spacing: 0.02em;
  }
  .masthead-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--forest);
    background: rgba(45,90,61,0.08);
    border: 1px solid rgba(45,90,61,0.2);
    border-radius: 3px;
    padding: 3px 8px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sec-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: var(--ivory) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: var(--forest) !important;
    box-shadow: 0 0 0 3px rgba(45,90,61,0.10) !important;
    background: var(--surface) !important;
  }

  div[data-baseweb="select"] > div {
    background: var(--ivory) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  div[data-baseweb="select"] > div:focus-within {
    border-color: var(--forest) !important;
    box-shadow: 0 0 0 3px rgba(45,90,61,0.10) !important;
  }

  [data-testid="stFileUploader"] {
    background: var(--ivory) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 8px !important;
    transition: border-color 0.2s !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: var(--forest2) !important;
    background: var(--surface) !important;
  }
  [data-testid="stFileUploader"] * { color: var(--ink) !important; }

  label, .stSelectbox label, .stTextInput label,
  .stTextArea label, .stFileUploader label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-family: 'DM Mono', monospace !important;
  }

  .stButton > button {
    background: var(--forest) !important;
    color: var(--ivory) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 12px rgba(45,90,61,0.22) !important;
    transition: all 0.2s !important;
    position: relative !important;
    overflow: hidden !important;
  }
  .stButton > button:hover {
    background: var(--forest2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(45,90,61,0.30) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }

  div[data-testid="column"] .stButton > button {
    width: 100%;
    padding: 0.45rem 0 !important;
    font-size: 0.8rem !important;
    border-radius: 5px !important;
    background: var(--parchment) !important;
    color: var(--ink2) !important;
    box-shadow: none !important;
    border: 1.5px solid var(--border) !important;
  }
  div[data-testid="column"] .stButton > button:hover {
    background: var(--border) !important;
    transform: none !important;
    box-shadow: none !important;
  }

  .stDownloadButton > button {
    background: var(--ivory) !important;
    color: var(--forest) !important;
    border: 1.5px solid var(--forest) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.6rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
  }
  .stDownloadButton > button:hover {
    background: var(--forest) !important;
    color: var(--ivory) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(45,90,61,0.20) !important;
  }

  .info-box  { background:#eef5f0;border:1.5px solid #a8cbb5;border-radius:6px;padding:.75rem 1rem;font-size:.82rem;color:#1e4030;margin:.4rem 0; }
  .warn-box  { background:#fdf6ec;border:1.5px solid #e8c98a;border-radius:6px;padding:.7rem 1rem;font-size:.81rem;color:#6b4a1a;margin:.4rem 0; }
  .err-box   { background:#fdf0ee;border:1.5px solid #e8a89a;border-radius:6px;padding:.7rem 1rem;font-size:.81rem;color:#7a2a1e;margin:.4rem 0; }

  .panel-wrap {
    background: var(--ivory);
    border: 1.5px solid var(--border);
    border-radius: 10px;
    padding: 1.4rem 1.3rem 1.6rem;
    box-shadow: 0 2px 16px var(--shadow);
  }

  .map-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 24px var(--shadow);
    margin-bottom: 0.6rem;
  }
  .map-header {
    background: var(--forest);
    padding: 0.9rem 1.3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .map-header-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ivory);
    letter-spacing: -0.01em;
  }
  .map-header-desc {
    color: rgba(253,250,245,0.72);
    font-size: 0.76rem;
    margin-top: 2px;
    font-family: 'DM Sans', sans-serif;
  }
  .map-header-icon {
    font-size: 1.3rem;
    opacity: 0.85;
  }

  .legend-preview {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 0.6rem 1.2rem;
    background: var(--parchment);
    border-top: 1px solid var(--border);
  }
  .legend-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 9px;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 500;
    background: var(--ivory);
    border: 1px solid var(--border);
    color: var(--ink2);
    font-family: 'DM Sans', sans-serif;
  }
  .legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .map-placeholder-header {
    background: var(--forest);
    padding: 0.9rem 1.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .map-placeholder-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--ivory);
    font-style: italic;
  }

  hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
  .stSpinner > div { border-top-color: var(--forest) !important; }
  .stProgress > div > div > div {
    background: linear-gradient(90deg, var(--forest), var(--terra)) !important;
  }
  [data-testid="column"] { padding: 0 0.3rem !important; }

  .stTextArea textarea {
    min-height: 60px !important;
    resize: vertical !important;
  }

  .topo-accent {
    height: 3px;
    background: repeating-linear-gradient(
      90deg,
      var(--forest) 0px, var(--forest) 2px,
      transparent 2px, transparent 8px,
      var(--terra) 8px, var(--terra) 10px,
      transparent 10px, transparent 18px
    );
    border-radius: 2px;
    margin-bottom: 1.2rem;
    opacity: 0.5;
  }
</style>
""", unsafe_allow_html=True)

BASEMAPS = {
    "Dark Matter (Carto)":  "CartoDB dark_matter",
    "Positron (Carto)":     "CartoDB positron",
    "OpenStreetMap":        "OpenStreetMap",
    "Satellite (ESRI)":     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Terrain (ESRI)":       "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
    "Topo (ESRI)":          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    "Ocean (ESRI)":         "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
    "NatGeo (ESRI)":        "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
    "Light (Carto)":        "CartoDB positron",
}

VECTOR_TYPES = ["geojson", "json", "kml", "zip"]
RASTER_TYPES = ["tif", "tiff", "geotiff"]


def parse_color(s):
    if not s or not s.strip():
        return None
    s = s.strip().lower()
    if s in mcolors.CSS4_COLORS:
        return mcolors.to_hex(mcolors.CSS4_COLORS[s])
    try:
        return mcolors.to_hex(s)
    except Exception:
        return None


def auto_colors(n: int) -> list:
    palette = ["#2d5a3d","#b85c38","#5b8fa8","#8b6f47","#6a8e5f",
               "#c4704a","#3d7a8a","#a07040","#4a7a5a","#d4885a"]
    if n <= len(palette):
        return palette[:n]
    cmap = plt.get_cmap("tab10")
    return [mcolors.to_hex(cmap(i / n)) for i in range(n)]


def read_vector(file_bytes: bytes, filename: str) -> gpd.GeoDataFrame:
    ext = Path(filename).suffix.lower().lstrip(".")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, filename)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        if ext == "zip":
            with zipfile.ZipFile(tmp_path, "r") as z:
                z.extractall(tmpdir)
            shps = list(Path(tmpdir).rglob("*.shp"))
            if not shps:
                raise ValueError("No .shp found in ZIP.")
            tmp_path = str(shps[0])
        elif ext == "kml":
            try:
                import fiona
                fiona.drvsupport.supported_drivers["KML"] = "rw"
                fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
            except Exception:
                pass
        gdf = gpd.read_file(tmp_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf


def load_raster_raw(file_bytes: bytes, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        with rasterio.open(tmp) as src:
            bounds_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            scale = max(1, src.width // 1024, src.height // 1024)
            data = src.read(
                1,
                out_shape=(src.height // scale, src.width // scale),
                resampling=rasterio.enums.Resampling.average,
            )
            nodata = src.nodata
            dtype  = str(src.dtypes[0])
    finally:
        os.unlink(tmp)

    masked = np.ma.masked_equal(data, nodata) if nodata is not None else np.ma.array(data)
    masked = np.ma.masked_invalid(masked)
    flat   = masked.compressed()
    vmin   = float(flat.min()) if flat.size > 0 else 0.0
    vmax   = float(flat.max()) if flat.size > 0 else 1.0
    unique_vals = np.unique(flat).tolist() if flat.size > 0 and len(np.unique(flat)) <= 50 else []

    return {
        "bounds":      bounds_4326,
        "data":        masked,
        "vmin":        vmin,
        "vmax":        vmax,
        "dtype":       dtype,
        "unique_vals": unique_vals,
        "n_unique":    int(len(np.unique(flat))) if flat.size > 0 else 0,
    }


def render_raster_continuous(raw: dict, colormap_name: str = "viridis") -> str:
    masked = raw["data"]
    vmin, vmax = raw["vmin"], raw["vmax"]
    if vmin == vmax:
        vmax = vmin + 1
    norm = (masked - vmin) / (vmax - vmin)
    try:
        cmap = plt.get_cmap(colormap_name)
    except ValueError:
        cmap = plt.get_cmap("viridis")
    rgba = cmap(norm)
    mask = masked.mask if np.ma.is_masked(masked) else np.zeros(masked.shape, bool)
    rgba[..., 3] = np.where(mask, 0.0, 0.80)
    return _rgba_to_data_uri(rgba, masked.shape)


def render_raster_classified(raw: dict, class_colors: list) -> str:
    masked      = raw["data"]
    unique_vals = raw["unique_vals"]
    rgba        = np.zeros((*masked.shape, 4), dtype=float)

    if class_colors and "value" in class_colors[0]:
        for c in class_colors:
            try:
                rgb = mcolors.to_rgba(c["color"])
            except Exception:
                rgb = (0.5, 0.5, 0.5, 1.0)
            mask_v = (masked == float(c["value"]))
            rgba[mask_v] = [rgb[0], rgb[1], rgb[2], 0.85]
    else:
        colors_list = [c["color"] for c in class_colors] if class_colors else auto_colors(max(len(unique_vals), 1))
        for i, uv in enumerate(sorted(unique_vals)):
            hex_c = colors_list[i % len(colors_list)]
            try:
                rgb = mcolors.to_rgba(hex_c)
            except Exception:
                rgb = (0.5, 0.5, 0.5, 1.0)
            rgba[masked == uv] = [rgb[0], rgb[1], rgb[2], 0.85]

    if np.ma.is_masked(masked):
        rgba[masked.mask] = [0, 0, 0, 0]

    return _rgba_to_data_uri(rgba, masked.shape)


def _rgba_to_data_uri(rgba: np.ndarray, shape: tuple) -> str:
    h, w = shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(rgba, aspect="auto")
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def get_centroid_and_zoom(all_bounds):
    if not all_bounds:
        return (20.0, 0.0), 2
    min_lon = min(b[0] for b in all_bounds)
    min_lat = min(b[1] for b in all_bounds)
    max_lon = max(b[2] for b in all_bounds)
    max_lat = max(b[3] for b in all_bounds)
    center  = ((min_lat + max_lat) / 2, (min_lon + max_lon) / 2)
    span    = max(max_lat - min_lat, max_lon - min_lon)
    if   span < 0.01: zoom = 15
    elif span < 0.1:  zoom = 12
    elif span < 1:    zoom = 10
    elif span < 5:    zoom = 8
    elif span < 20:   zoom = 6
    elif span < 60:   zoom = 4
    else:             zoom = 3
    return center, zoom


def ask_ai_for_map_config(prompt, title, description, basemap,
                           layer_type, legend_items, filenames, raster_metadata) -> dict:
    system = """You are a professional GIS cartographer. Return ONLY valid JSON (no markdown fences).

Schema:
{
  "title": "finalized title",
  "description": "finalized description",
  "basemap": "one of the available basemap names",
  "layers": [
    {
      "filename": "exact filename",
      "label": "display name",
      "type": "vector|raster",
      "style": {
        "color": "#hex",
        "fillColor": "#hex",
        "weight": 1.5,
        "fillOpacity": 0.6,
        "raster_mode": "continuous|classified",
        "colormap": "viridis|plasma|RdYlGn|Blues|Reds|YlOrRd|hot|terrain|coolwarm|Spectral",
        "class_colors": [{"value": 1, "color": "#hex"}, ...]
      },
      "tooltip_fields": []
    }
  ],
  "legend": [{"label": "text", "color": "#hex"}],
  "map_notes": "optional notes"
}

RASTER RENDERING RULES (read carefully):
- If the user describes CLASSES, CATEGORIES, LAND-USE TYPES, or assigns specific colors to pixel values
  → raster_mode = "classified", populate class_colors with {"value": <pixel_value>, "color": "#hex"}
  → Use unique_vals from metadata to know what pixel values exist
  → If exact values unknown, use rank-based: [{"color":"#hex"}, ...] in sorted-value order
- If the user describes a GRADIENT, SPECTRUM, or CONTINUOUS data (NDVI, elevation, temperature, etc.)
  → raster_mode = "continuous", set colormap name only
- When n_unique <= 15 and user is ambiguous → prefer "classified"
- When n_unique > 50 → prefer "continuous"
"""

    meta_str = ""
    if raster_metadata:
        meta_str = "\nRaster metadata:\n"
        for fn, m in raster_metadata.items():
            meta_str += (f"  {fn}: dtype={m['dtype']}, range=[{m['vmin']:.3f},{m['vmax']:.3f}], "
                         f"n_unique={m['n_unique']}, unique_vals={m['unique_vals'][:20]}\n")

    user_msg = f"""Title: {title or 'Untitled'}
Description: {description or 'none'}
Basemap: {basemap}
Layer type: {layer_type}
Files: {', '.join(filenames) or 'none'}
{meta_str}
Legend items: {json.dumps(legend_items, indent=2)}

USER PROMPT:
{prompt or 'No prompt provided.'}

Available basemaps: {', '.join(BASEMAPS.keys())}"""

    full_prompt = f"{system}\n\n{user_msg}"
    response = model.generate_content(full_prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def build_folium_map(config, vector_files, raster_raw) -> folium.Map:
    basemap_key = config.get("basemap", "Positron (Carto)")
    if basemap_key not in BASEMAPS:
        for k in BASEMAPS:
            if basemap_key.lower() in k.lower():
                basemap_key = k
                break
        else:
            basemap_key = "Positron (Carto)"

    tile_source = BASEMAPS[basemap_key]
    all_bounds = []
    for lc in config.get("layers", []):
        fn = lc.get("filename", "")
        if lc["type"] == "vector" and fn in vector_files:
            b = vector_files[fn].total_bounds
            all_bounds.append((b[0], b[1], b[2], b[3]))
        elif lc["type"] == "raster" and fn in raster_raw:
            all_bounds.append(raster_raw[fn]["bounds"])

    if not all_bounds:
        center, zoom = (20, 0), 2
    else:
        center, zoom = get_centroid_and_zoom(all_bounds)

    if tile_source.startswith("http"):
        m = folium.Map(location=center, zoom_start=zoom, tiles=tile_source, attr="ESRI")
    else:
        try:
            m = folium.Map(location=center, zoom_start=zoom, tiles=tile_source)
        except Exception:
            m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    for lc in config.get("layers", []):
        fn    = lc.get("filename", "")
        label = lc.get("label", fn)
        style = lc.get("style", {})
        ltype = lc.get("type", "vector")

        if ltype == "vector" and fn in vector_files:
            gdf    = vector_files[fn]
            if len(gdf) == 0:
                continue
            t_flds = [f for f in lc.get("tooltip_fields", []) if f in gdf.columns][:3]
            fc     = style.get("fillColor", "#2d5a3d")
            sc     = style.get("color", "#fff")
            fo     = style.get("fillOpacity", 0.6)
            w      = style.get("weight", 1.5)

            geom_t = gdf.geometry.geom_type.iloc[0]
            if "Point" in geom_t:
                for _, row in gdf.iterrows():
                    tip = "<br>".join(f"<b>{f}</b>: {row[f]}" for f in t_flds if f in row) or label
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=6, color=sc, fill=True,
                        fill_color=fc, fill_opacity=fo, weight=w,
                        tooltip=folium.Tooltip(tip),
                    ).add_to(m)
            else:
                def _sfn(feature, _fc=fc, _sc=sc, _fo=fo, _w=w):
                    return {"fillColor": _fc, "color": _sc, "weight": _w, "fillOpacity": _fo}
                folium.GeoJson(
                    gdf.__geo_interface__, name=label, style_function=_sfn,
                    tooltip=folium.GeoJsonTooltip(fields=t_flds, aliases=t_flds) if t_flds else None,
                ).add_to(m)

        elif ltype == "raster" and fn in raster_raw:
            raw         = raster_raw[fn]
            raster_mode = style.get("raster_mode", "continuous")
            if raster_mode == "classified":
                img_uri = render_raster_classified(raw, style.get("class_colors", []))
            else:
                img_uri = render_raster_continuous(raw, style.get("colormap", "viridis"))

            b = raw["bounds"]
            folium.raster_layers.ImageOverlay(
                image=img_uri,
                bounds=[[b[1], b[0]], [b[3], b[2]]],
                opacity=0.85,
                name=label,
            ).add_to(m)

    leg = config.get("legend", [])
    if leg:
        m.get_root().html.add_child(folium.Element(_build_legend_html(leg, config.get("title", ""))))

    folium.LayerControl(position="topright").add_to(m)
    return m


def _build_legend_html(items, title="") -> str:
    rows = "".join(
        '<div style="display:flex;align-items:center;gap:8px;margin:5px 0;white-space:nowrap;">'
        '<div style="width:11px;height:11px;'
        'background:' + i.get("color", "#2d5a3d") + ';'
        'border-radius:2px;flex-shrink:0;border:1px solid rgba(0,0,0,0.15);"></div>'
        '<span style="font-size:11.5px;color:#1e1e1e;font-family:Georgia,serif;font-weight:400;">'
        + i.get("label", "") +
        '</span></div>'
        for i in items
    )
    return (
        '<div id="geoai-legend" style="'
        'position:fixed;bottom:18px;right:12px;z-index:9999;'
        'background:rgba(253,250,245,0.97);'
        'border:1px solid rgba(0,0,0,0.12);'
        'border-radius:6px;'
        'padding:10px 14px 9px;'
        'display:inline-block;width:fit-content;'
        'font-family:Georgia,serif;'
        'box-shadow:0 2px 12px rgba(0,0,0,0.12);'
        'max-width:200px;'
        '">'
        '<div style="font-size:9px;text-transform:uppercase;letter-spacing:0.16em;'
        'font-weight:700;margin-bottom:8px;color:#2d5a3d;font-family:Georgia,serif;">'
        + (title or "Legend") +
        "</div>"
        + rows +
        "</div>"
    )


def export_map_html(folium_map: folium.Map, title: str, description: str) -> str:
    raw_html = folium_map.get_root().render()
    m = re.search(r'id=["\']([^"\']*(?:map|folium)[^"\']*)["\']', raw_html)
    map_div_id = m.group(1) if m else None

    has_desc  = bool(description and description.strip())
    hdr_h     = "72px" if has_desc else "48px"

    id_css = ""
    if map_div_id:
        id_css = f"""
        #{map_div_id} {{
            position: absolute !important;
            inset: 0 !important;
            width: 100% !important;
            height: 100% !important;
        }}"""

    desc_block = f'<p class="hd">{description}</p>' if has_desc else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{title or 'Interactive Map'}</title>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;1,400&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *,*::before,*::after{{margin:0;padding:0;box-sizing:border-box;}}
    html,body{{width:100%;height:100%;overflow:hidden;background:#f7f3ec;font-family:'DM Sans',sans-serif;}}
    #hdr{{
      position:fixed;top:0;left:0;right:0;
      height:{hdr_h};z-index:10000;
      background:#2d5a3d;
      border-bottom:2px solid rgba(253,250,245,0.15);
      padding:0 24px;
      display:flex;flex-direction:column;justify-content:center;
      box-shadow:0 2px 12px rgba(0,0,0,0.18);
    }}
    .ht{{
      font-family:'Playfair Display',Georgia,serif;
      font-size:clamp(1.4rem, 3.2vw, 2.2rem);
      font-weight:600;color:#fdfaf5;
      letter-spacing:-0.01em;line-height:1.1;
    }}
    .hd{{
      font-size:clamp(0.9rem, 1.6vw, 1.1rem);
      color:rgba(253,250,245,0.70);
      margin-top:3px;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
      font-family:'DM Sans',sans-serif;
    }}
    #map-wrap{{
      position:fixed;
      top:{hdr_h};left:0;right:0;bottom:0;
      overflow:hidden;
    }}
    {id_css}
    .leaflet-container{{width:100%!important;height:100%!important;}}
  </style>
</head>
<body>
  <div id="hdr">
    <div class="ht">{title or 'Interactive Map'}</div>
    {desc_block}
  </div>
  <div id="map-wrap">
    {raw_html}
  </div>
</body>
</html>"""



st.markdown("""
<div class="masthead">
  <div class="masthead-left">
    <span class="masthead-title">AI <em>Cartographer</em></span>
    <span class="masthead-rule"></span>
    <span class="masthead-sub">Describe any map - the AI builds it</span>
  </div>
  <span class="masthead-badge">Geospatial Studio</span>
</div>
""", unsafe_allow_html=True)

left_col, right_col = st.columns([1.5, 2.2], gap="medium")

with left_col:
    st.markdown('<div class="sec-label">Basemap</div>', unsafe_allow_html=True)
    basemap_choice = st.selectbox("Basemap", list(BASEMAPS.keys()), label_visibility="collapsed")

    st.markdown('<div class="sec-label" style="margin-top:1rem;">Layer Type</div>', unsafe_allow_html=True)
    layer_type = st.selectbox("Layer Type",
                              ["Vector", "Raster", "Mixed (Vector + Raster)"],
                              label_visibility="collapsed")
    if "Vector" in layer_type:
        allowed_ext = VECTOR_TYPES
    elif "Raster" in layer_type:
        allowed_ext = RASTER_TYPES
    else:
        allowed_ext = VECTOR_TYPES + RASTER_TYPES

    st.markdown('<div class="sec-label" style="margin-top:1rem;">Upload Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload", accept_multiple_files=True,
                                      type=allowed_ext, label_visibility="collapsed")

    st.markdown('<div class="sec-label" style="margin-top:1rem;">Legend Items</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:.73rem;color:var(--muted2);margin-bottom:.6rem;font-family:\'DM Sans\',sans-serif;">Label + color per item. Leave blank for auto-assignment.</div>',
                unsafe_allow_html=True)

    if "legend_count" not in st.session_state:
        st.session_state.legend_count = 2

    c1, c2 = st.columns(2)
    if c1.button("+ Add"):
        st.session_state.legend_count += 1
    if c2.button("− Remove"):
        st.session_state.legend_count = max(1, st.session_state.legend_count - 1)

    legend_entries = []
    for i in range(st.session_state.legend_count):
        a, b = st.columns([2, 1])
        label = a.text_input(f"L{i}", placeholder="Item name", label_visibility="collapsed", key=f"l{i}")
        color = b.color_picker(f"C{i}", label_visibility="collapsed", key=f"c{i}")
        if label:
            legend_entries.append({"label": label, "color": color})

    st.markdown('<div class="sec-label" style="margin-top:1rem;">Prompt</div>', unsafe_allow_html=True)
    ai_prompt = st.text_area(
        "AI Prompt",
        placeholder=( "Describe title, purpose, layers, colors, classes, study area …\n\n" ),
        height=130,
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    create_clicked = st.button("Generate Map →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


with right_col:
    def draw_map():
        cfg             = st.session_state.get("map_config")
        folium_map_html = st.session_state.get("folium_map_html")

        if folium_map_html is None:
            st.markdown("""
            <div class="map-card">
              <div class="map-placeholder-header">
                <span class="map-placeholder-title">World Overview</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
            st_folium(m, height=560, width="100%", returned_objects=[])
            return

      
        ftitle = st.session_state.get("map_title") or cfg.get("title", "Map")
        fdesc  = st.session_state.get("map_description") or cfg.get("description", "")
        legend = cfg.get("legend", [])

        pills = "".join(
            f'<span class="legend-pill">'
            f'<span class="legend-swatch" style="background:{it.get("color","#999")};"></span>'
            f'{it.get("label","")}</span>'
            for it in legend
        )
        lprev = f'<div class="legend-preview">{pills}</div>' if pills else ""

        st.markdown(f"""
        <div class="map-card">
          <div class="map-header">
            <div>
              <div class="map-header-title">{ftitle}</div>
              {'<div class="map-header-desc">' + fdesc + '</div>' if fdesc else ''}
            </div>
            <div class="map-header-icon">🗺️</div>
          </div>
          {lprev}
        </div>
        """, unsafe_allow_html=True)

        components.html(folium_map_html, height=560)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        html_out = st.session_state.get("map_html_export", "")
        safe_t   = re.sub(r"[^\w\-]", "_", ftitle or "map").lower()
        st.download_button(
            label="↓  Download Interactive Map (.html)",
            data=html_out.encode("utf-8"),
            file_name=f"{safe_t}_interactive.html",
            mime="text/html",
            use_container_width=True,
        )
        st.markdown(
            '<div style="text-align:center;color:var(--muted2);font-size:0.7rem;margin-top:0.4rem;'
            'font-family:\'DM Mono\',monospace;letter-spacing:0.05em;">'
            'Opens as a full-page interactive map in your browser</div>',
            unsafe_allow_html=True,
        )

  
    if create_clicked:

        errs = []
        if not uploaded_files:
            errs.append("Upload at least one file.")
        if not ai_prompt:
            errs.append("Enter a prompt.")
        for e in errs:
            st.markdown(f'<div class="err-box">⚠ {e}</div>', unsafe_allow_html=True)
        if errs:
            st.stop()

        vector_files    = {}
        raster_raw      = {}
        filenames       = []
        raster_metadata = {}

        prog = st.progress(0, text="Reading files…")

        for i, uf in enumerate(uploaded_files):
            fname  = uf.name
            ext    = Path(fname).suffix.lower().lstrip(".")
            fbytes = uf.read()
            filenames.append(fname)
            prog.progress((i + 1) / (len(uploaded_files) + 4), text=f"Loading {fname}…")

            if ext in VECTOR_TYPES:
                try:
                    vector_files[fname] = read_vector(fbytes, fname)
                except Exception as ex:
                    st.markdown(f'<div class="warn-box">⚠ Vector read failed: {ex}</div>', unsafe_allow_html=True)
            elif ext in RASTER_TYPES:
                try:
                    raw = load_raster_raw(fbytes, fname)
                    raster_raw[fname] = raw
                    raster_metadata[fname] = {k: raw[k] for k in ("n_unique","unique_vals","vmin","vmax","dtype")}
                except Exception as ex:
                    st.markdown(f'<div class="warn-box">⚠ Raster read failed: {ex}</div>', unsafe_allow_html=True)

        ac = auto_colors(max(len(legend_entries), 1))
        resolved_legend = [
            {"label": it["label"], "color": parse_color(it.get("color","")) or ac[idx % len(ac)]}
            for idx, it in enumerate(legend_entries)
        ]

        prog.progress(0.65, text="Consulting AI cartographer…")

        try:
            config = ask_ai_for_map_config(
                prompt=ai_prompt,
                title="",
                description="",
                basemap=basemap_choice,
                layer_type=layer_type,
                legend_items=resolved_legend,
                filenames=filenames,
                raster_metadata=raster_metadata,
            )
        except Exception as ex:
            st.markdown(f'<div class="warn-box">⚠ AI config failed: {ex}. Using defaults.</div>', unsafe_allow_html=True)
            config = {
                "title": "Untitled", "description": "", "basemap": basemap_choice,
                "layers": [], "legend": resolved_legend, "map_notes": "",
            }

        if not config.get("legend"):
            config["legend"] = resolved_legend

        prog.progress(0.88, text="Rendering map…")
        folium_map = build_folium_map(config, vector_files, raster_raw)

        st.session_state["folium_map_html"]  = folium_map._repr_html_()
        st.session_state["map_html_export"]  = export_map_html(
            folium_map,
            config.get("title", "Untitled Map"),
            config.get("description", "")
        )
        st.session_state["map_config"]       = config
        st.session_state["map_title"]        = config.get("title", "Untitled Map")
        st.session_state["map_description"]  = config.get("description", "")

        prog.progress(1.0, text="Done!")
        prog.empty()

        st.rerun()

    draw_map()
