# Film Batch Sorter

A local macOS web app that scans a folder of sequentially-named film scan images, detects batch-marker frames (images with a printed 3-digit number on a label), and copies each batch into a numbered subfolder.

## How it works

Each roll of film is introduced by a **marker image** — a photograph of a small rectangular label bearing a printed 3-digit number (e.g. `042`). The app detects these markers using Claude's vision API and sorts all subsequent images into a folder named `Film042/` until the next marker appears.

### Supported formats
- JPEG (`.jpg` / `.jpeg`)
- Nikon RAW (`.nef`)

### Rotation handling
Marker images may be photographed upside-down. Each image is sent to the API at both 0° and 180° so the correct orientation is always found regardless of EXIF metadata.

### Speed
API calls are parallelised (6 at a time by default). For a typical batch of 200 images the scan phase takes around 2–3 minutes.

---

## Requirements

- macOS
- Python 3 (the system Python on macOS is fine)
- An [Anthropic API key](https://console.anthropic.com/settings/keys)

Python packages are installed automatically on first launch:
- `flask`
- `anthropic`
- `pillow`
- `rawpy`

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/film-batch-sorter.git
cd film-batch-sorter
chmod +x launch.command
```

Then double-click **`launch.command`** in Finder, or run:

```bash
./launch.command
```

The app opens at **http://localhost:5174** in your browser.

> **First run note:** macOS may show a security warning. Right-click `launch.command` → **Open** → **Open**.

---

## Usage

1. Paste your Anthropic API key (saved locally in `~/.film_batch_sorter.json`)
2. Select the **input folder** containing your incoming images
3. Select the **output folder** where `Film###/` subfolders will be created
4. Click **Sort batches ▶**

The log streams live as images are scanned. Original files are never modified — the app only copies.

---

## Configuration

`PARALLEL_WORKERS = 6` near the top of `film_batch_sorter.py` controls how many API calls run simultaneously. Raise it for faster throughput if your API tier allows; lower it if you hit rate-limit errors.

---

## Output structure

```
output_folder/
├── Film023/
│   ├── dcs-001.jpg   ← marker image
│   ├── dcs-002.jpg
│   └── …
├── Film024/
│   ├── dcs-017.jpg   ← marker image
│   └── …
└── …
```
