# Comic font (user-supplied — REQUIRED for re-lettering)

Comic lettering fonts are licensed material, so **no font files are
tracked in this repo**. Before running the re-lettering stage you must
supply your chosen font here:

1. **Copy the font files into this folder** with these exact names:

   ```
   relettering/fonts/regular.ttf       the dialogue face
   relettering/fonts/bolditalic.ttf    the emphasis face (*word* in transcripts)
   ```

   Any comic-lettering font works (e.g. CC WildWords, Komika). Pick one
   whose character set covers your comic's language — verify accents /
   special characters render before committing to a full book (the fit
   measures glyphs with PIL, so a missing glyph shows up as wrong widths
   or tofu in the final render).

2. **Install the same font system-wide** (macOS: `~/Library/Fonts`) so
   GIMP can render it, and tell the GIMP pass its family names in
   `relettering/fonts/fonts.json` (untracked, next to this file):

   ```json
   {
    "gimp_regular": "CC WildWords Lower",
    "gimp_bolditalic": "CC WildWords Lower Bold Italic"
   }
   ```

   These are the names GIMP shows in its font picker — they usually differ
   from the file names. To list what GIMP sees:
   `gimp -i -b "(gimp-fonts-refresh)" -b "(gimp-version)"` and check the
   font picker, or query `pdb.gimp_fonts_get_list("")` in the GIMP console.

Notes:
- `reletter_fit.py` refuses to run until both `.ttf` files are present.
- Cap-height and line metrics are **measured from the font at run time** —
  never assume ratios; different faces vary a lot (e.g. WildWords' cap
  height is ~1.0 em, Komika's ~0.72 em).
- If you change fonts mid-project, re-run the fit for every page (layouts
  and the book-wide size depend on the font's metrics).
