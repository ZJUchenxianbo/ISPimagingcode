"""Trace Figure 6.2's vector paths for comparison, not raw-data recovery.

This helper is specific to PSWF_project1_Maxwell (3).pdf, PDF page 17.
The main reproduction script does not depend on it. Requires pypdf and numpy.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("pdf", type=Path)
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figure62_reproduction")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
reader = PdfReader(args.pdf)
form = reader.pages[16]["/Resources"]["/XObject"]["/Im3"].get_object()
ops = ContentStream(form, reader).operations
curves = {}
for i, (operands, operator) in enumerate(ops):
    if operator != b"RG" or len(operands) != 3:
        continue
    color = np.array([float(v) for v in operands])
    J = 3 if np.allclose(color, [.1215686275, .4666666667, .7058823529]) else (
        6 if np.allclose(color, [1, .4980392157, .0549019608]) else None)
    if J is None:
        continue
    points = []
    for values, op in ops[i + 1:]:
        if op in (b"S", b"RG", b"Q"):
            break
        if op in (b"m", b"l"):
            points.append([float(v) for v in values])
    if len(points) > 10:
        curves[J] = np.array(points)
assert set(curves) == {3, 6}
# Clip rectangle of the axes, and the vector grid lines labelled 2e-3 and 1e-2.
left, width = 69.271875, 376.398888
y_low, y_high = 59.957238, 245.930630
slope = (np.log(.01) - np.log(.002)) / (y_high - y_low)
rho = (curves[3][:, 0] - left) / width
assert np.allclose(curves[3][:, 0], curves[6][:, 0])
means = {J: np.exp(np.log(.002) + slope * (points[:, 1] - y_low))
         for J, points in curves.items()}
np.savetxt(args.out / "original_vector_curves.csv",
           np.column_stack([rho, means[3], means[6]]), delimiter=",",
           header="rho,mean_J3,mean_J6", comments="")
(args.out / "reference_provenance.json").write_text(json.dumps({
    "source": str(args.pdf.resolve()), "pdf_page": 17, "figure": "6.2",
    "method": "extract vector curve paths and invert the logarithmic axis transform",
    "not_raw_experiment_data": True,
    "calibration": {"x_left": left, "x_width": width,
                    "log_axis_points": [[y_low, .002], [y_high, .01]]},
    "points_per_curve": len(rho),
}, indent=2), encoding="utf-8")
print("Traced reference curves:", len(rho), "points per design")
