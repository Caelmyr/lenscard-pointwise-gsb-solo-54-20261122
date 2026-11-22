# LensCard

LensCard is a local print desk for preparing a single lenticular photograph from several views. It keeps the original camera files, working layout, retained proofs, and most recent approved print as distinct records.

## Run

Run `./start.sh` with Docker and Compose, then open http://127.0.0.1:19957/. The service binds to the loopback interface. It keeps the card library in a Compose named volume. `docker compose stop` stops the service and preserves the volume. Set `LENSCARD_PORT` and use a separate Compose project name for an isolated copy.

For a native start, install the locked dependencies in a Python 3.11 virtual environment and run `python -B -m lenscard`. `LENSCARD_DATA`, `HOST`, and `PORT` change the data directory and listener. The default native data directory is `runtime-data`.

## Prepare a card

1. In the library, open **Harbour at blue hour** for a three-angle example or **Terrace in the rain** for a four-angle example. These are marked synthetic illustrations. To use your own photographs, create a card and import each view with its camera angle. PNG, JPEG, single-page TIFF and WebP are supported. Active originals must share the same oriented dimensions.
2. In **Compose**, enter the lens pitch, phase, drift, viewing cone, paper dimensions and printer DPI. Change referencing, common coverage and crop independently of the original images. Save draft settings, then build a print preview. Use **Observed angle** to test the synthetic angular model and **Band map** to inspect native view occupancy. Back-mounted print geometry is reflected as part of the export.
3. For two flip messages on one blank area, set **Flip captions**: both texts, the read angle for each, and the shared blank area. **Check flip captions** previews the two lens bands, the angle span each message reads over, the quiet span between them and any one-band retreat each side could take. Nothing is moved automatically, and the captions are drawn only in the derived print raster and its previews, never into the camera originals.
4. Keep the displayed proof for approval, then explicitly approve it from **Records**. Download the actual-size PDF, original device PNG or evidence ZIP from **Approved print**. Working edits leave the approved print pinned to the previous snapshot. Print at actual size with fit-to-page disabled.

## Check the inputs and printed card

| Operation | Entry and observable result |
| --- | --- |
| Source file receipt and readout | **Source info** shows original-byte digest, orientation, encoding and observed pixel support. The original is downloadable. |
| Pixel sampling | Click an original in the stage, then use **Original pixel probe** to read a covered neighborhood. |
| Capture exposure | **Exposure and clipping** maps high and low levels and shows linear-code distributions. |
| Profiles and edge energy | **Native line profile** and **Edge energy** show measured intensities, gradients and unsupported locations. |
| Texture and grain | **Texture frequency** or **Approximate grain** helps distinguish recurring patterns and quiet-area variation. |
| Registration and comparison | **Shared framing** reports aligned coverage. **Registered view comparison** presents image divider and signed differences in comparable locations. |
| Stock simulation | **Paper simulation** compares source and approximate soft-proof appearance. **Density step response** reads explicitly marked patches. |
| Pitch fitting | **Pitch target** builds a true-size pattern. **Fit marked lens lines** fits one global physical grid from indexed print locations and reports residuals. |
| Direct optical isolation | **View isolation target** is printable. **Measure angle photographs** compares each intended band to neighbouring bands using local white and dark reference patches. |
| Flip caption check | **Check flip captions** reports the read span of each message, the quiet span between them, the largest simultaneous mix and a one-band retreat per side. **Observed angle** shows which message is visible at the slider angle. |
| Readback geometry | **Rectify camera photograph** uses the four marked corners of the paper to produce a normalized copy. |
| Camera operating checklist | **Plan angle stops** prepares a PDF and CSV of predicted viewing intervals. **Review camera timing** compares captured times to the planned holds. |
| Physical print verification | **Inspect device pixels**, **Physical ruler proof**, **Check measured print size**, and **Measure scanned stripe pitch** read retained sampling or physical measurements. |
| Version check | **Record print lineage**, **Compare two raster versions**, and **Scan device band coverage** inspect saved device files and the source snapshots. |
| Portable materials | Download the **View set ZIP** to import original views as a new draft. Check an approved ZIP or inspection ZIP without overwriting the library. |

A layout has up to eight angle views. Device output is limited to eight megapixels and 5,000 pixels on the longest side. Imports are limited to 22 MiB per image and 14 megapixels per source. An angle response is a simple global lens model. It cannot predict a bowed lens, partial obstruction or the true colour response of a particular printer. Use the real test chart, verified ruler and camera measurements when checking manufactured work. Sample geometry, plots and photographs are generated locally. The example ruler settings and viewing angles are in the metrology controls for those cards.
