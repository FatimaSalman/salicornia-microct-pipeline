
import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

# slicer_render_C0.py
# =========================================================================
# Run INSIDE 3D Slicer:  View > Python Console, then paste:
#
#   exec(open('$SALICORNIA_DATA/slicer_render_C0.py').read())
#
# Builds Figure 5c from the corrected C-0 mask: loads the label map, forces
# the correct 2.5 um voxel spacing, converts it to a segmentation, matches the
# colour and smoothing of the published panel, hides the orientation furniture,
# points the camera the same way, and writes a high-resolution PNG.
# =========================================================================

import os
import vtk
import ctk
import slicer

# ------------------------- settings -------------------------------------
NRRD   = os.path.join(DATA, "Salicornia_5/C-0_cc_reference_mask.nrrd")
OUT    = os.path.join(DATA, "Figure5c_C0_new.png")
SPACING_MM = 0.0025            # 2.5 um expressed in mm
COLOR      = (0.75, 0.42, 0.33)  # salmon, as in the published panel
SMOOTHING  = "0.5"
AZIMUTH    = 0                 # rotate the camera around the stem axis, degrees
ELEVATION  = 0                 # tilt, degrees
MAGNIFY    = 4                 # output magnification
# ------------------------------------------------------------------------

slicer.mrmlScene.Clear(0)

# 1. load as a label map ---------------------------------------------------
lm = slicer.util.loadVolume(NRRD, properties={"labelmap": True})
print("loaded:", lm.GetName())

# 2. force the correct spacing BEFORE any conversion -----------------------
lm.SetSpacing(SPACING_MM, SPACING_MM, SPACING_MM)
lm.SetOrigin(0.0, 0.0, 0.0)
print("spacing set to:", lm.GetSpacing())

import numpy as np
arr = slicer.util.arrayFromVolume(lm)
vox = (arr > 0).sum()
print(f"non-zero voxels: {vox:,}  ->  {vox * (2.50000064 ** 3) / 1e9:.4f} mm3  (expect 0.2782)")

# 3. convert to a segmentation --------------------------------------------
seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "C0")
seg.CreateDefaultDisplayNodes()
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(lm, seg)
seg.GetSegmentation().SetConversionParameter("Smoothing factor", SMOOTHING)
seg.CreateClosedSurfaceRepresentation()

segmentation = seg.GetSegmentation()
sid = segmentation.GetNthSegmentID(0)
segment = segmentation.GetSegment(sid)
segment.SetName("central cylinder")
segment.SetColor(*COLOR)
print("segment:", segment.GetName())

# 4. hide the label map in the 2D views, keep only the 3D surface ----------
slicer.util.setSliceViewerLayers(background=None, label=None)
segDisp = seg.GetDisplayNode()
segDisp.SetVisibility2D(False)
segDisp.SetVisibility3D(True)
segDisp.SetOpacity3D(1.0)

# 5. one big 3D view --------------------------------------------------------
lm_node = slicer.app.layoutManager()
lm_node.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
slicer.app.processEvents()

view3d = lm_node.threeDWidget(0).threeDView()
viewNode = view3d.mrmlViewNode()

# hide axis labels, bounding box and the orientation cube
viewNode.SetAxisLabelsVisible(False)
viewNode.SetBoxVisible(False)
try:
    viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeNone)
except Exception:
    pass
# Keep the scale bar, it is present in the published panel.
# IMPORTANT: Slicer only draws the 3D ruler in PARALLEL projection, so that is
# switched on below, just before the camera is positioned.
try:
    viewNode.SetRulerType(slicer.vtkMRMLAbstractViewNode.RulerTypeThick)
    viewNode.SetRulerColor(slicer.vtkMRMLAbstractViewNode.RulerColorWhite)
except Exception as e:
    print("ruler setup skipped:", e)

# hide the slice planes inside the 3D view
for name in ("Red", "Green", "Yellow"):
    sn = slicer.app.layoutManager().sliceWidget(name).mrmlSliceNode()
    sn.SetSliceVisible(False)

# 6. camera: stem vertical, viewed from the front --------------------------
view3d.resetFocalPoint()
view3d.lookFromViewAxis(ctk.ctkAxesWidget.Anterior)
cam = slicer.util.getNode("vtkMRMLCameraNode1")
# parallel projection is REQUIRED for the scale bar to be drawn, and it also
# removes perspective foreshortening, which makes the panel easier to compare
# with the other three reconstructions
cam.GetCamera().SetParallelProjection(True)
if AZIMUTH:
    cam.GetCamera().Azimuth(AZIMUTH)
if ELEVATION:
    cam.GetCamera().Elevation(ELEVATION)
    cam.GetCamera().OrthogonalizeViewUp()
view3d.resetCamera()
slicer.app.processEvents()
view3d.forceRender()

# 7. high-resolution screenshot --------------------------------------------
rw = view3d.renderWindow()
renderer = rw.GetRenderers().GetFirstRenderer()
large = vtk.vtkRenderLargeImage()
large.SetInput(renderer)
large.SetMagnification(MAGNIFY)
large.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName(OUT)
writer.SetInputConnection(large.GetOutputPort())
writer.Write()
print("wrote:", OUT)
print("If the background looks tiled, set MAGNIFY = 1 and enlarge the Slicer window instead.")
