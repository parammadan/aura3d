"""Convert FANUC's official M-710iC/50 xacro (Apache-2.0) into a MuJoCo-loadable URDF.

Source: github.com/FANUC-CORPORATION/fanuc_description (commit recorded in assets/robots/fanuc_m710ic_50/SOURCE).
MuJoCo cannot read the .dae visual meshes, so the collision STLs are used for both roles.
Kinematics, joint limits and link inertials come from FANUC's file unchanged.
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

import xacro

PKG = "fanuc_m710_description"
MODEL = "m710ic_50"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="external/fanuc_description")
    ap.add_argument("--out", default="assets/robots/fanuc_m710ic_50")
    a = ap.parse_args()
    src, out = Path(a.src).resolve(), Path(a.out)
    pkg = src / PKG
    (out / "meshes").mkdir(parents=True, exist_ok=True)

    # xacro's $(find pkg) needs ROS; substitute the package path instead.
    tmp = out / "_entry.xacro"
    entry = (pkg / "robot" / f"{MODEL}.urdf.xacro").read_text().replace(f"$(find {PKG})", str(pkg))
    tmp.write_text(entry)
    macro = pkg / "urdf" / f"{MODEL}_urdf_macro.xacro"
    patched = out / "_macro.xacro"
    patched.write_text(macro.read_text().replace(f"$(find {PKG})", str(pkg)))
    tmp.write_text(entry.replace(str(macro), str(patched.resolve())))
    urdf = xacro.process_file(str(tmp)).toprettyxml(indent="  ")
    tmp.unlink()
    patched.unlink()

    for stl in (pkg / "meshes" / MODEL / "collision").glob("*.stl"):
        shutil.copy(stl, out / "meshes" / stl.name)
    # package://.../visual/x.dae and .../collision/x.stl -> meshes/x.stl
    urdf = re.sub(r'filename="[^"]*/(\w+)\.(dae|stl)"', r'filename="\1.stl"', urdf)
    (out / f"{MODEL}.urdf").write_text(urdf)
    shutil.copy(src / "LICENSES" / "Apache-2.0.txt", out / "LICENSE")
    sha = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "SOURCE").write_text(
        f"https://github.com/FANUC-CORPORATION/fanuc_description@{sha}\n{PKG}/robot/{MODEL}.urdf.xacro\n"
        "Modified: xacro expanded; visual .dae replaced by collision .stl; mesh paths flattened.\n"
    )
    print("wrote", out / f"{MODEL}.urdf")


if __name__ == "__main__":
    main()
