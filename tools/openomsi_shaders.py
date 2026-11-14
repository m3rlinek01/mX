#!/usr/bin/env python3
"""Write openOMSI's WGSL modules, assembled as its renderer assembles them for Metal, Vulkan
and Direct3D 12 (not OpenGL), into one directory, one `.wgsl` file per module.

    python3 tools/openomsi_shaders.py <openOMSI checkout> <out dir>

`cargo test -p mx --test openomsi_shaders` then compiles each through mX (set
MX_OPENOMSI_SHADERS=<out dir>). The assembly follows crates/omsi-render/src/shaders.rs and
shaders/sources.rs; when those change, change this script with them.
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1]) / "crates"
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
src = root / "omsi-render" / "src"


def r(p):
    return (src / p).read_text()


SCENE = "".join(r(f"shaders/scene/{f.name}") for f in sorted((src / "shaders/scene").glob("*.wgsl")))
ENHANCED = "".join(r(f"shaders/enhanced/{f.name}") for f in sorted((src / "shaders/enhanced").glob("*.wgsl")))
FOG_LAMP_LIGHTS = """struct PointLight {
    pos: vec4<f32>,
    color: vec4<f32>,
    dir: vec4<f32>,
    extra: vec4<f32>,
};
@group(0) @binding(3) var<storage, read> lights: array<PointLight>;
@group(0) @binding(4) var<storage, read> grid: array<u32>;
const CELL_CAP: u32 = 32u;
"""


def scene(rt_gbuf, amd_vulkan, ms):
    s = "\n".join([r("colour.wgsl"), SCENE, r("road_snow.wgsl"), r("enhanced_common.wgsl"), r("puddle_common.wgsl"), r("beam_shadow_none.wgsl"), r("lamp_air.wgsl"), ENHANCED])
    if rt_gbuf:
        s = s.replace("//RT ", "")
    s = s.replace("//VK ", "") if amd_vulkan else s.replace("//DX ", "")
    if ms:
        s = s.replace("//MS ", "")
    return s


def precip(own):
    return "\n".join([r("precip_common.wgsl"), own, r("enhanced_common.wgsl"), FOG_LAMP_LIGHTS, r("beam_shadow_none.wgsl"), r("lamp_air.wgsl")])


common = r("enhanced_common.wgsl")
fsr = r("fsr1.wgsl")
modules = {
    "scene": scene(False, False, False),
    "scene_rt_gbuf": scene(True, False, False),
    "scene_msaa_shading": scene(False, False, True),
    "scene_amd_vulkan": scene(False, True, False),
    "temporal_motion": scene(False, False, False) + "\n" + r("temporal/motion.wgsl"),
    "sky": "\n".join([r("colour.wgsl"), r("sky.wgsl")]),
    "sky_enhanced": "\n".join([r("colour.wgsl"), r("sky.wgsl"), common, r("sky_enhanced.wgsl")]),
    "snow": precip(r("snow.wgsl")),
    "rain": precip(r("rain.wgsl")),
    "fog_lamps": "\n".join([r("fog_lamps.wgsl"), common, FOG_LAMP_LIGHTS, r("beam_shadow.wgsl"), r("lamp_air.wgsl")]),
    "corona": "\n".join([r("corona.wgsl"), common, FOG_LAMP_LIGHTS, r("beam_shadow_none.wgsl"), r("lamp_air.wgsl")]),
    "puddles": "\n".join([r("colour.wgsl"), r("puddle_common.wgsl"), r("puddle_reflection.wgsl")]),
    "rt_lighting": "\n".join([r("rt_common.wgsl"), r("rt.wgsl")]),
    "rt_reflect": "\n".join([r("rt_common.wgsl"), common[: common.index("@group(0) @binding(11)")], r("rt_reflect.wgsl")]),
    "rt_avg": r("rt_avg.wgsl"),
    "rt_uv": r("rt_uv.wgsl"),
    "fsr1_srgb": fsr,
    "fsr1_linear": fsr.replace("const SRGB_TARGET: bool = true;", "const SRGB_TARGET: bool = false;"),
    "fsr1_srgb_raw": fsr.replace("const RAW_INPUT: bool = false;", "const RAW_INPUT: bool = true;"),
    "mip": r("mip.wgsl"),
    "upscale": r("upscale.wgsl"),
    "overlay": r("overlay.wgsl"),
    "xr_ui": r("xr_ui.wgsl"),
    "ssao": r("ssao.wgsl"),
    "post": r("post.wgsl"),
    "lamp_sight": r("lamp_sight.wgsl"),
    "ui": (root / "omsi-ui/src/ui.wgsl").read_text(),
}
xr = root / "omsi-app/src/xr_mirror.wgsl"
if xr.exists():
    modules["xr_mirror"] = xr.read_text()
for name, text in modules.items():
    (out / f"{name}.wgsl").write_text(text)
print(f"{len(modules)} modules in {out}")
