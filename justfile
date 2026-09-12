# Webcomics print pipeline
# Usage: just process "<comic folder name>"

python := ".venv/bin/python"

default:
    @just --list

# create the venv and install dependencies (run once)
setup:
    uv venv .venv
    uv pip install --python .venv/bin/python playwright pillow pikepdf numpy
    .venv/bin/playwright install chromium

# list vertical images and report the ones too big for the upscaler
scan comic:
    {{python}} process.py "{{comic}}" scan

# upscale images to their exact canvas size with Real-ESRGAN (local) -> upscaled/<comic>/
upscale comic:
    {{python}} process.py "{{comic}}" upscale

# upscale via iloveimg.com instead (mangles comic lettering; add --headed in process.py if needed)
upscale-iloveimg comic:
    {{python}} process.py "{{comic}}" upscale --engine iloveimg

# place upscaled images on template.xcf -> xcf/<comic>/ + pdf/<comic>/
compose comic:
    {{python}} process.py "{{comic}}" compose

# combine all page PDFs into pdf/<comic>.pdf
merge comic:
    {{python}} process.py "{{comic}}" merge

# full pipeline: upscale + compose + merge
process comic:
    {{python}} process.py "{{comic}}" all
