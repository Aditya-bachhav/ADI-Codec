# FilterX · ADI v34 Math Mesh UI

Web UI and Flask API that wrap the "ADI v34" geometric codec. Upload any bitmap, tune the vertex count, and download the mesh-rendered artwork in your preferred format.

## Features
- Drag-and-drop uploader with live preview of both original and processed images.
- Slider to change the number of math vertices (500–5000).
- Resolution cap control to keep processing fast on very large files.
- Download button supporting PNG, JPEG, WebP, or BMP outputs.
- Telemetry panel exposing vertex/triangle counts, canvas size, runtime, and scale.

## Getting Started
1. **Create & activate a virtual environment (recommended)**
   ```powershell
   cd c:/Users/adity/OneDrive/Desktop/adi_prototype/FilterX
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run the dev server**
   ```powershell
   flask --app app run --debug
   ```
4. Visit <http://127.0.0.1:5000> and start meshing images.

## Project Structure
```
FilterX/
├─ app.py              # Flask entry point + API
├─ mesh_filter.py      # Math mesh core logic reused by the API
├─ templates/
│   └─ index.html      # Single-page UI
├─ static/
│   ├─ styles.css      # Custom styling
│   └─ script.js       # Front-end interactions + fetch logic
└─ requirements.txt    # Python dependencies
```

## Customization Tips
- Tweak `DEFAULT_NUM_POINTS`, `MAX_NUM_POINTS`, or `MAX_DIMENSION` inside `mesh_filter.py` to change the permissible processing envelope.
- Extend `ALLOWED_EXPORTS` in `app.py` if you need additional formats supported by Pillow.
- The UI is plain HTML/CSS/JS, so you can integrate it into another framework or restyle it without touching the backend logic.

## Notes
- OpenCV (headless build) is used strictly for feature detection; SciPy powers Delaunay triangulation, while Pillow handles all drawing.
- Because the rendering step scales extremely large images down to `maxDimension` for processing, the telemetry panel reports the upscaled canvas dimensions so you always know the final export size.
