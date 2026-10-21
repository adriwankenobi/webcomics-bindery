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

# combine all page PDFs into pdf/<comic>.pdf (builds the comparison PDF first)
merge comic:
    {{python}} process.py "{{comic}}" merge

# original | final side by side, one page per book page -> pdf/<comic> - comparison.pdf
compare comic:
    {{python}} process.py "{{comic}}" compare

# full pipeline: upscale + compose + merge; --relettering = re-lettered book (pauses for the transcription, resumes when filled)
process comic *flags:
    {{python}} process.py "{{comic}}" all {{flags}}

# torch 2.2.2 = the last release with Intel-Mac wheels; it needs numpy 1.x
# one-time setup for `just untext`: .venv-untext + the LaMa and CRAFT models in tools/
setup-untext:
    [ -x .venv-untext/bin/python ] || uv venv --python 3.12 .venv-untext
    uv pip install --python .venv-untext/bin/python "torch==2.2.2" "torchvision==0.17.2" "numpy<2" pillow "opencv-python-headless<4.11" easyocr
    .venv-untext/bin/python untext.py --setup

# remove the text from ONE image -> <name>-untexted + <name>-untexted-mask.png beside it (an edited mask is reused; delete it to detect again)
[no-cd]
untext image:
    "{{justfile_directory()}}/.venv-untext/bin/python" "{{justfile_directory()}}/untext.py" "{{image}}"

# unit tests for the pipeline glue (tests/; untext's run in .venv-untext once it exists)
test:
    {{python}} -m unittest discover tests
    if [ -x .venv-untext/bin/python ]; then .venv-untext/bin/python -m unittest tests.test_untext; fi
