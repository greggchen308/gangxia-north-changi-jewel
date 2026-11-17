"""
Jewel Changi Airport - Rain Vortex interior, fully procedural (Blender 5.x, Cycles CPU).

Everything in the frame is modelled here: toroidal diagrid glass dome with oculus,
the Rain Vortex waterfall (layered translucent shells + volumetric mist + spray),
basin and viewing deck, stepped forest terraces scattered with a small library of
procedural tropical plants, the people-mover on its guideway, tiny people, haze.

Run (bpy as a Python module):
    python jewel_rain_vortex.py -- --scale 0.25 --samples 48          # quick preview
    python jewel_rain_vortex.py -- --scale 0.5  --samples 96          # mid-res check
    python jewel_rain_vortex.py -- --scale 1.0  --samples 256         # final 1440x2560
Options (or env vars JEWEL_SCALE / JEWEL_SAMPLES / JEWEL_OUT / JEWEL_THREADS):
    --scale   resolution scale of 1440x2560     (default 0.25)
    --samples Cycles max samples                (default 48)
    --out     output PNG path                   (default ./out/jewel_rain_vortex[_preview].png)
    --threads render threads                    (default 2)
Units are metres. World origin = centre of the basin at deck level; camera looks along +Y.
"""
import bpy
import math
import os
import random
import sys
import time
from math import cos, sin, pi, radians
from mathutils import Vector, Matrix, noise
from bpy_extras.object_utils import world_to_camera_view

# ----------------------------------------------------------------------------- args
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {
        "scale": float(os.environ.get("JEWEL_SCALE", 0.25)),
        "samples": int(os.environ.get("JEWEL_SAMPLES", 48)),
        "out": os.environ.get("JEWEL_OUT", ""),
        "threads": int(os.environ.get("JEWEL_THREADS", 2)),
    }
    i = 0
    while i < len(argv):
        k = argv[i].lstrip("-")
        if k in opts and i + 1 < len(argv):
            opts[k] = type(opts[k])(argv[i + 1]) if opts[k] != "" else argv[i + 1]
            i += 2
        else:
            i += 1
    if not opts["out"]:
        here = os.path.dirname(os.path.abspath(__file__))
        name = "jewel_rain_vortex.png" if opts["scale"] >= 0.999 else "jewel_rain_vortex_preview.png"
        opts["out"] = os.path.join(here, "out", name)
    return opts


ARGS = parse_args()
RNG = random.Random(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
ROOT = scene.collection


def new_coll(name, parent=None, exclude=False):
    c = bpy.data.collections.new(name)
    (parent or ROOT).children.link(c)
    if exclude:
        bpy.context.view_layer.layer_collection.children[name].exclude = True
    return c


# ----------------------------------------------------------------------------- geometry helpers
class Geo:
    """Accumulates verts / faces / per-face material index / per-corner UVs, then builds a mesh."""

    def __init__(self):
        self.v, self.f, self.m, self.uv = [], [], [], []

    def add(self, verts, faces, mat=0, uvs=None, M=None):
        o = len(self.v)
        if M is not None:
            verts = [M @ Vector(p) for p in verts]
        self.v.extend(Vector(p) for p in verts)
        for f in faces:
            self.f.append(tuple(i + o for i in f))
            self.m.append(mat)
            self.uv.append([uvs[i] for i in f] if uvs else [(0.0, 0.0)] * len(f))

    def build(self, name, mats, coll, smooth=True):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for p in self.v], [], self.f)
        uvl = me.uv_layers.new(name="UVMap")
        flat = [c for face in self.uv for corner in face for c in corner]
        uvl.data.foreach_set("uv", flat)
        me.polygons.foreach_set("material_index", self.m)
        me.polygons.foreach_set("use_smooth", [smooth] * len(self.f))
        for m in mats:
            me.materials.append(m)
        me.update()
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        return ob


def tube(pts, rads, sides=6):
    """Tube along a polyline with parallel-transport frames. Returns verts, faces, uvs."""
    verts, faces, uvs = [], [], []
    n = len(pts)
    prev = None
    for i, p in enumerate(pts):
        if i == 0:
            t = (pts[1] - pts[0]).normalized()
        elif i == n - 1:
            t = (pts[-1] - pts[-2]).normalized()
        else:
            t = (pts[i + 1] - pts[i - 1]).normalized()
        if prev is None:
            a = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
            nrm = t.cross(a).normalized()
        else:
            nrm = (prev - t * prev.dot(t)).normalized()
        prev = nrm
        b = t.cross(nrm)
        for s in range(sides):
            a_ = 2 * pi * s / sides
            verts.append(p + rads[i] * (cos(a_) * nrm + sin(a_) * b))
            uvs.append((s / sides, i / (n - 1)))
    for i in range(n - 1):
        for s in range(sides):
            s2 = (s + 1) % sides
            faces.append((i * sides + s, i * sides + s2, (i + 1) * sides + s2, (i + 1) * sides + s))
    return verts, faces, uvs


def revolve(profile, segs, close=False, phase=0.0):
    """Surface of revolution of [(r, z), ...] around Z. UV: u around, v along profile."""
    verts, faces, uvs = [], [], []
    n = len(profile)
    for j in range(segs + 1):
        a = 2 * pi * (j / segs + phase)
        for i, (r, z) in enumerate(profile):
            verts.append((r * cos(a), r * sin(a), z))
            uvs.append((j / segs, i / max(1, n - 1)))
    for j in range(segs):
        for i in range(n - 1):
            a0 = j * n + i
            b0 = (j + 1) * n + i
            faces.append((a0, b0, b0 + 1, a0 + 1))
    return verts, faces, uvs


def frame_matrix(origin, ydir, up=Vector((0, 0, 1))):
    """Matrix whose local +Y points along ydir and local +Z is as close to `up` as possible."""
    y = ydir.normalized()
    x = y.cross(up)
    if x.length < 1e-5:
        x = y.cross(Vector((1, 0, 0)))
    x.normalize()
    z = x.cross(y).normalized()
    M = Matrix((x, y, z)).transposed().to_4x4()
    M.translation = origin
    return M


def blade(L, W, nseg=4, droop=0.25, fold=0.15, shape="lance"):
    """Single leaf blade along +Y, face normal +Z. Returns verts, faces, uvs."""
    verts, faces, uvs = [], [], []
    for i in range(nseg + 1):
        t = i / nseg
        y = L * t
        z = -droop * L * t * t
        if shape == "oblong":     # banana / heliconia
            w = W * max(0.0, 1 - (2 * t - 1) ** 6) ** 0.5
        elif shape == "heart":    # alocasia / taro
            w = W * (math.sin(pi * min(1.0, t * 1.15 + 0.08)) ** 0.7)
        else:                      # lanceolate
            w = W * math.sin(pi * t) ** 0.85
        verts += [(-w, y, z + fold * w), (0, y, z), (w, y, z + fold * w)]
        uvs += [(0.0, t), (0.5, t), (1.0, t)]
    for i in range(nseg):
        a, b = 3 * i, 3 * (i + 1)
        faces.append((a, a + 1, b + 1, b))
        faces.append((a + 1, a + 2, b + 2, b + 1))
    return verts, faces, uvs


# ----------------------------------------------------------------------------- material helpers
def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    return m, nt, out


def N(nt, kind, **inputs):
    n = nt.nodes.new(kind)
    for k, v in inputs.items():
        if k.startswith("_"):
            setattr(n, k[1:], v)
        else:
            n.inputs[k].default_value = v
    return n


def L(nt, a, b):
    nt.links.new(a, b)


def math_node(nt, op, a=None, b=None, c=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for idx, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[idx].default_value = v
        else:
            nt.links.new(v, n.inputs[idx])
    return n.outputs[0]


def ramp(nt, src, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], stops[0][1]
    els[1].position, els[1].color = stops[-1][0], stops[-1][1]
    for p, c in stops[1:-1]:
        e = els.new(p)
        e.color = c
    L(nt, src, r.inputs[0])
    return r


def principled(name, color, rough=0.5, metal=0.0, spec=0.5, emit=None, emit_strength=0.0):
    m, nt, out = node_mat(name)
    b = N(nt, "ShaderNodeBsdfPrincipled")
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    L(nt, b.outputs[0], out.inputs[0])
    return m


def noisy_principled(name, c1, c2, scale=2.0, rough=0.7, bump=0.0, coord="Object"):
    m, nt, out = node_mat(name)
    tc = N(nt, "ShaderNodeTexCoord")
    nz = N(nt, "ShaderNodeTexNoise", Scale=scale, Detail=6.0, Roughness=0.6)
    L(nt, tc.outputs[coord], nz.inputs["Vector"])
    r = ramp(nt, nz.outputs["Fac"], [(0.3, (*c1, 1)), (0.7, (*c2, 1))])
    b = N(nt, "ShaderNodeBsdfPrincipled", Roughness=rough)
    b.inputs["Specular IOR Level"].default_value = 0.35
    L(nt, r.outputs[0], b.inputs["Base Color"])
    if bump > 0:
        bp = N(nt, "ShaderNodeBump", Strength=bump, Distance=0.05)
        L(nt, nz.outputs["Fac"], bp.inputs["Height"])
        L(nt, bp.outputs[0], b.inputs["Normal"])
    L(nt, b.outputs[0], out.inputs[0])
    return m


def leaf_material(name, greens, translucency=0.35, gloss=0.45, hue_jitter=0.035):
    """Leaf: per-instance colour from a ramp of greens, per-leaf (island) jitter, translucent."""
    m, nt, out = node_mat(name)
    oi = N(nt, "ShaderNodeObjectInfo")
    geo = N(nt, "ShaderNodeNewGeometry")
    isl = geo.outputs["Random Per Island"]
    # colour pick: mostly per plant instance, a little per leaf
    rnd = math_node(nt, "MULTIPLY_ADD", oi.outputs["Random"], 0.75,
                    math_node(nt, "MULTIPLY", isl, 0.25))
    cr = ramp(nt, rnd, [(i / (len(greens) - 1), (*g, 1)) for i, g in enumerate(greens)])
    hsv = N(nt, "ShaderNodeHueSaturation", Saturation=1.0)
    L(nt, math_node(nt, "MULTIPLY_ADD", isl, hue_jitter * 2, 0.5 - hue_jitter), hsv.inputs["Hue"])
    L(nt, math_node(nt, "MULTIPLY_ADD", isl, 0.45, 0.78), hsv.inputs["Value"])
    L(nt, cr.outputs[0], hsv.inputs["Color"])
    # paler midrib from blade UV (u = 0.5 on the midrib)
    tc = N(nt, "ShaderNodeTexCoord")
    sep = N(nt, "ShaderNodeSeparateXYZ")
    L(nt, tc.outputs["UV"], sep.inputs[0])
    d = math_node(nt, "ABSOLUTE", math_node(nt, "SUBTRACT", sep.outputs[0], 0.5))
    near_rib = math_node(nt, "LESS_THAN", d, 0.03)
    ribmix = N(nt, "ShaderNodeMix", _data_type="RGBA", _blend_type="MULTIPLY")
    L(nt, near_rib, ribmix.inputs[0])
    L(nt, hsv.outputs[0], ribmix.inputs[6])
    ribmix.inputs[7].default_value = (1.35, 1.4, 1.05, 1)
    b = N(nt, "ShaderNodeBsdfPrincipled", Roughness=gloss)
    b.inputs["Specular IOR Level"].default_value = 0.55
    L(nt, ribmix.outputs[2], b.inputs["Base Color"])
    tr = N(nt, "ShaderNodeBsdfTranslucent")
    tint = N(nt, "ShaderNodeMix", _data_type="RGBA", _blend_type="MULTIPLY", Factor=1.0)
    L(nt, ribmix.outputs[2], tint.inputs[6])
    tint.inputs[7].default_value = (1.25, 1.55, 0.55, 1)
    L(nt, tint.outputs[2], tr.inputs["Color"])
    mx = N(nt, "ShaderNodeMixShader", Fac=translucency)
    L(nt, b.outputs[0], mx.inputs[1])
    L(nt, tr.outputs[0], mx.inputs[2])
    L(nt, mx.outputs[0], out.inputs[0])
    return m


# ----------------------------------------------------------------------------- materials
MAT = {}


def build_materials():
    MAT["rib"] = principled("rib_white_steel", (0.86, 0.87, 0.86), rough=0.35, spec=0.5,
                           emit=(1.0, 0.99, 0.97), emit_strength=0.3)
    # dome glass: almost fully transmissive, slight green tint + faint reflection
    m, nt, out = node_mat("dome_glass")
    tr = N(nt, "ShaderNodeBsdfTransparent", Color=(0.93, 0.96, 0.95, 1))
    gl = N(nt, "ShaderNodeBsdfGlossy", Roughness=0.08)
    lw = N(nt, "ShaderNodeLayerWeight", Blend=0.12)
    mx = N(nt, "ShaderNodeMixShader")
    L(nt, lw.outputs["Fresnel"], mx.inputs[0])
    L(nt, tr.outputs[0], mx.inputs[1])
    L(nt, gl.outputs[0], mx.inputs[2])
    L(nt, mx.outputs[0], out.inputs[0])
    MAT["glass"] = m

    # oculus ring: white with soft glow from light bouncing in the funnel
    MAT["oculus"] = principled("oculus_ring", (0.85, 0.86, 0.86), rough=0.3,
                               emit=(1.0, 0.98, 0.95), emit_strength=0.6)
    MAT["oculus_in"] = principled("oculus_inner", (0.95, 0.95, 0.95), rough=0.2,
                                  emit=(1.0, 0.98, 0.96), emit_strength=4.0)
    MAT["concrete"] = noisy_principled("concrete", (0.48, 0.47, 0.45), (0.62, 0.61, 0.58),
                                       scale=1.5, rough=0.8, bump=0.1)
    MAT["deck"] = noisy_principled("deck_stone", (0.20, 0.19, 0.18), (0.30, 0.29, 0.27), scale=3.0, rough=0.6)
    MAT["soil"] = noisy_principled("ground_cover", (0.025, 0.04, 0.015), (0.07, 0.10, 0.03),
                                   scale=1.2, rough=0.9, bump=0.3)
    MAT["wall"] = noisy_principled("terrace_wall", (0.22, 0.22, 0.2), (0.38, 0.37, 0.34),
                                   scale=0.8, rough=0.85, bump=0.2)
    MAT["bark"] = noisy_principled("bark", (0.10, 0.085, 0.065), (0.24, 0.21, 0.17),
                                   scale=8.0, rough=0.85, bump=0.4)
    MAT["palm_trunk"] = noisy_principled("palm_trunk", (0.22, 0.2, 0.17), (0.40, 0.37, 0.32),
                                         scale=5.0, rough=0.8, bump=0.3)
    MAT["stem"] = principled("stem", (0.12, 0.2, 0.05), rough=0.5)
    MAT["leaf_dark"] = leaf_material("leaf_dark", [(0.020, 0.055, 0.016), (0.035, 0.085, 0.022),
                                                   (0.05, 0.11, 0.03)], translucency=0.3)
    MAT["leaf_mid"] = leaf_material("leaf_mid", [(0.035, 0.09, 0.02), (0.06, 0.14, 0.03),
                                                 (0.10, 0.19, 0.035)], translucency=0.35)
    MAT["leaf_light"] = leaf_material("leaf_light", [(0.09, 0.19, 0.03), (0.15, 0.27, 0.04),
                                                     (0.22, 0.34, 0.05)], translucency=0.42)
    MAT["palm"] = leaf_material("palm_leaf", [(0.04, 0.09, 0.025), (0.06, 0.13, 0.035),
                                             (0.09, 0.17, 0.045)], translucency=0.35, gloss=0.38)
    MAT["banana"] = leaf_material("banana_leaf", [(0.06, 0.15, 0.04), (0.10, 0.21, 0.05),
                                                 (0.13, 0.25, 0.06)], translucency=0.4, gloss=0.35)

    # basin: dark smoked acrylic with water film
    MAT["bowl"] = principled("bowl_dark", (0.008, 0.009, 0.01), rough=0.18, spec=0.5)
    MAT["steel"] = principled("steel", (0.6, 0.6, 0.6), rough=0.3, metal=1.0)
    m, nt, out = node_mat("rail_glass")
    tr = N(nt, "ShaderNodeBsdfTransparent", Color=(0.85, 0.9, 0.9, 1))
    gl = N(nt, "ShaderNodeBsdfGlossy", Roughness=0.05)
    mx = N(nt, "ShaderNodeMixShader", Fac=0.12)
    L(nt, tr.outputs[0], mx.inputs[1]); L(nt, gl.outputs[0], mx.inputs[2]); L(nt, mx.outputs[0], out.inputs[0])
    MAT["railglass"] = m

    # train: white body, dark window band from object-space height, dark skirt
    m, nt, out = node_mat("train_body")
    tc = N(nt, "ShaderNodeTexCoord")
    sep = N(nt, "ShaderNodeSeparateXYZ")
    L(nt, tc.outputs["Object"], sep.inputs[0])
    zb = math_node(nt, "MULTIPLY", math_node(nt, "GREATER_THAN", sep.outputs[2], 1.25),
                   math_node(nt, "LESS_THAN", sep.outputs[2], 2.55))
    xb = math_node(nt, "LESS_THAN", math_node(nt, "ABSOLUTE", sep.outputs[0]), 5.35)
    band = math_node(nt, "MULTIPLY", zb, xb)
    skirt = math_node(nt, "LESS_THAN", sep.outputs[2], 0.35)
    white = N(nt, "ShaderNodeBsdfPrincipled", Roughness=0.22)
    white.inputs["Base Color"].default_value = (0.86, 0.87, 0.88, 1)
    white.inputs["Coat Weight"].default_value = 0.5
    dark = N(nt, "ShaderNodeBsdfPrincipled", Roughness=0.06)
    dark.inputs["Base Color"].default_value = (0.02, 0.025, 0.03, 1)
    dark.inputs["Specular IOR Level"].default_value = 0.9
    grey = N(nt, "ShaderNodeBsdfPrincipled", Roughness=0.5)
    grey.inputs["Base Color"].default_value = (0.12, 0.12, 0.13, 1)
    m1 = N(nt, "ShaderNodeMixShader")
    L(nt, band, m1.inputs[0]); L(nt, white.outputs[0], m1.inputs[1]); L(nt, dark.outputs[0], m1.inputs[2])
    m2 = N(nt, "ShaderNodeMixShader")
    L(nt, skirt, m2.inputs[0]); L(nt, m1.outputs[0], m2.inputs[1]); L(nt, grey.outputs[0], m2.inputs[2])
    L(nt, m2.outputs[0], out.inputs[0])
    MAT["train"] = m
    MAT["dark"] = principled("dark_rubber", (0.03, 0.03, 0.03), rough=0.6)

    # people: per-instance clothing colour
    m, nt, out = node_mat("people")
    oi = N(nt, "ShaderNodeObjectInfo")
    r = ramp(nt, oi.outputs["Random"], [(0.0, (0.75, 0.74, 0.7, 1)), (0.2, (0.08, 0.09, 0.12, 1)),
                                        (0.4, (0.55, 0.15, 0.12, 1)), (0.55, (0.85, 0.8, 0.62, 1)),
                                        (0.7, (0.15, 0.25, 0.42, 1)), (0.85, (0.9, 0.9, 0.88, 1)),
                                        (1.0, (0.3, 0.32, 0.3, 1))])
    r.color_ramp.interpolation = "CONSTANT"
    b = N(nt, "ShaderNodeBsdfPrincipled", Roughness=0.7)
    L(nt, r.outputs[0], b.inputs["Base Color"])
    L(nt, b.outputs[0], out.inputs[0])
    MAT["people"] = m
    MAT["skin"] = principled("skin", (0.45, 0.3, 0.22), rough=0.6)
    MAT["hair"] = principled("hair", (0.03, 0.025, 0.02), rough=0.6)

    MAT["water_shell"] = []
    for k in range(4):
        MAT["water_shell"].append(water_shell_material(k))
    MAT["mist"] = mist_material("mist_core", 0.02)
    MAT["spray"] = spray_material()
    MAT["foam"] = foam_material()


def water_shell_material(k):
    """Translucent falling-water sheet: vertically stretched noise streaks -> alpha,
    brighter / more continuous near the oculus, more opaque at grazing angles."""
    m, nt, out = node_mat(f"water_shell_{k}")
    tc = N(nt, "ShaderNodeTexCoord")
    sep = N(nt, "ShaderNodeSeparateXYZ")
    L(nt, tc.outputs["Object"], sep.inputs[0])
    hz = math_node(nt, "DIVIDE", sep.outputs[2], 40.0, clamp=True)  # 0 basin .. 1 oculus
    # fine streaks: stretch noise along Z
    mp = N(nt, "ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (7.0, 7.0, 0.11)
    mp.inputs["Location"].default_value = (k * 13.1, k * 7.7, k * 3.3)
    L(nt, tc.outputs["Object"], mp.inputs["Vector"])
    nz = N(nt, "ShaderNodeTexNoise", Scale=1.0, Detail=5.0, Roughness=0.62, Distortion=0.15)
    L(nt, mp.outputs[0], nz.inputs["Vector"])
    # coarse clumps
    mp2 = N(nt, "ShaderNodeMapping")
    mp2.inputs["Scale"].default_value = (1.1, 1.1, 0.045)
    mp2.inputs["Location"].default_value = (k * 5.0, 0, 0)
    L(nt, tc.outputs["Object"], mp2.inputs["Vector"])
    nz2 = N(nt, "ShaderNodeTexNoise", Scale=1.0, Detail=3.0, Roughness=0.5)
    L(nt, mp2.outputs[0], nz2.inputs["Vector"])
    comb = math_node(nt, "ADD", math_node(nt, "MULTIPLY", nz.outputs["Fac"], 0.65),
                     math_node(nt, "MULTIPLY", nz2.outputs["Fac"], 0.35))
    a = ramp(nt, comb, [(0.47, (0, 0, 0, 1)), (0.66, (1, 1, 1, 1))])
    # continuity near the top: add hz^3
    top = math_node(nt, "POWER", hz, 3.0)
    alpha = math_node(nt, "MULTIPLY_ADD", a.outputs[0], math_node(nt, "MULTIPLY_ADD", top, 0.9, 1.0),
                      math_node(nt, "MULTIPLY", top, 0.06))
    lw = N(nt, "ShaderNodeLayerWeight", Blend=0.5)
    graze = math_node(nt, "MULTIPLY_ADD", lw.outputs["Facing"], 0.9, 0.35)
    alpha = math_node(nt, "MULTIPLY", alpha, graze)
    alpha = math_node(nt, "MULTIPLY", alpha, [0.5, 0.38, 0.3, 0.25][k], clamp=True)
    # shading
    tr = N(nt, "ShaderNodeBsdfTransparent")
    dif = N(nt, "ShaderNodeBsdfDiffuse", Color=(0.9, 0.92, 0.93, 1))
    tl = N(nt, "ShaderNodeBsdfTranslucent", Color=(0.92, 0.95, 0.96, 1))
    m1 = N(nt, "ShaderNodeMixShader", Fac=0.55)
    L(nt, dif.outputs[0], m1.inputs[1]); L(nt, tl.outputs[0], m1.inputs[2])
    em = N(nt, "ShaderNodeEmission", Color=(1.0, 0.985, 0.96, 1))
    es = math_node(nt, "MULTIPLY_ADD", math_node(nt, "POWER", hz, 3.0), 1.0, 0.14)
    L(nt, es, em.inputs["Strength"])
    ad = N(nt, "ShaderNodeAddShader")
    L(nt, m1.outputs[0], ad.inputs[0]); L(nt, em.outputs[0], ad.inputs[1])
    mx = N(nt, "ShaderNodeMixShader")
    L(nt, alpha, mx.inputs[0]); L(nt, tr.outputs[0], mx.inputs[1]); L(nt, ad.outputs[0], mx.inputs[2])
    L(nt, mx.outputs[0], out.inputs[0])
    return m


def mist_material(name, density):
    """Homogeneous (cheap) faint mist inside the falling column."""
    m, nt, out = node_mat(name)
    v = N(nt, "ShaderNodeVolumePrincipled", Density=density, Anisotropy=0.35)
    v.inputs["Color"].default_value = (0.95, 0.97, 0.98, 1)
    L(nt, v.outputs[0], out.inputs["Volume"])
    return m


def spray_material():
    m, nt, out = node_mat("spray")
    tc = N(nt, "ShaderNodeTexCoord")
    # generated coords of a scaled sphere: distance from centre in normalized object space
    sep = N(nt, "ShaderNodeSeparateXYZ")
    L(nt, tc.outputs["Object"], sep.inputs[0])
    ln = N(nt, "ShaderNodeVectorMath", _operation="LENGTH")
    L(nt, tc.outputs["Object"], ln.inputs[0])
    fall = math_node(nt, "SUBTRACT", 1.0, ln.outputs["Value"], clamp=True)
    fall = math_node(nt, "POWER", fall, 1.6)
    nz = N(nt, "ShaderNodeTexNoise", Scale=2.5, Detail=4.0, Roughness=0.6)
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    d = math_node(nt, "MULTIPLY", fall, math_node(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 1.4, -0.25))
    d = math_node(nt, "MULTIPLY", d, 0.07, clamp=False)
    d = math_node(nt, "MAXIMUM", d, 0.0)
    v = N(nt, "ShaderNodeVolumePrincipled", Anisotropy=0.3)
    v.inputs["Color"].default_value = (0.95, 0.96, 0.97, 1)
    L(nt, d, v.inputs["Density"])
    L(nt, v.outputs[0], out.inputs["Volume"])
    return m


def foam_material():
    m, nt, out = node_mat("foam")
    tc = N(nt, "ShaderNodeTexCoord")
    nz = N(nt, "ShaderNodeTexNoise", Scale=1.8, Detail=8.0, Roughness=0.65)
    L(nt, tc.outputs["Object"], nz.inputs["Vector"])
    r = ramp(nt, nz.outputs["Fac"], [(0.35, (0.05, 0.06, 0.065, 1)), (0.65, (0.85, 0.88, 0.9, 1))])
    b = N(nt, "ShaderNodeBsdfPrincipled", Roughness=0.35)
    L(nt, r.outputs[0], b.inputs["Base Color"])
    L(nt, b.outputs[0], out.inputs[0])
    return m


# ----------------------------------------------------------------------------- architecture
OCULUS_R = 7.5      # outer radius of oculus ring (dome starts here)
OCULUS_Z = 41.0


def z_dome(r):
    return OCULUS_Z + 27.0 * (1 - math.exp(-(r - OCULUS_R) / 14.0)) - 0.004 * max(0.0, r - 55.0) ** 2


def build_dome(coll):
    """Triangulated toroidal gridshell: rings with alternating half-step offsets give a
    diagrid of near-equilateral triangles; ring count doubles twice going outward so
    panels stay ~2-4 m. Glass = faces, ribs = Wireframe modifier on a copy."""
    rings = []
    r, n, phase = OCULUS_R, 40, 0.0
    switches = [(15.0, 80), (34.0, 160)]
    while r < 128.0:
        rings.append((r, z_dome(r), n, phase))
        edge = 2 * pi * r / n
        slope = (z_dome(r + 0.01) - z_dome(r)) / 0.01
        r += edge * 0.866 / math.sqrt(1 + slope * slope)
        if switches and r >= switches[0][0]:
            n = switches.pop(0)[1]
            phase = phase * 2.0
        else:
            phase += 0.5
    verts, faces, starts = [], [], []
    for (rr, zz, nn, ph) in rings:
        starts.append(len(verts))
        for i in range(nn):
            a = 2 * pi * (i + ph) / nn
            verts.append((rr * cos(a), rr * sin(a), zz))
    for k in range(len(rings) - 1):
        na, nb = rings[k][2], rings[k + 1][2]
        A, B = starts[k], starts[k + 1]
        if nb == na:
            for i in range(na):
                i1 = (i + 1) % na
                faces.append((A + i, A + i1, B + i))
                faces.append((B + i, A + i1, B + i1))
        else:
            for i in range(na):
                i1 = (i + 1) % na
                b0, b1, b2 = B + 2 * i, B + 2 * i + 1, B + (2 * i + 2) % nb
                faces.append((A + i, b1, b0))
                faces.append((A + i, A + i1, b1))
                faces.append((A + i1, b2, b1))
    g = Geo()
    g.add(verts, faces)
    glass = g.build("dome_glass", [MAT["glass"]], coll, smooth=False)
    ribs = bpy.data.objects.new("dome_ribs", glass.data.copy())
    coll.objects.link(ribs)
    ribs.data.materials.clear()
    ribs.data.materials.append(MAT["rib"])
    wf = ribs.modifiers.new("ribs", "WIREFRAME")
    wf.thickness = 0.30
    wf.use_even_offset = True
    wf.use_replace = True
    wf.offset = 0.0
    # ribs are shaded white but kept out of shadow rays a little (glass roof stays bright)
    # major radial "spoke" beams that sweep from the oculus (like the real tree-like supports)
    g2 = Geo()
    for i in range(10):
        a = 2 * pi * (i + 0.25) / 10
        pts, rad = [], []
        for j in range(30):
            rr = OCULUS_R + 0.3 + j * 1.6
            aa = a + 0.006 * j * j * 0.5
            pts.append(Vector((rr * cos(aa), rr * sin(aa), z_dome(rr) - 0.45)))
            rad.append(0.38)
        v, f, uv = tube(pts, rad, 6)
        g2.add(v, f, 0, uv)
    g2.build("dome_spokes", [MAT["rib"]], coll)
    return glass, ribs


def build_oculus(coll):
    # thick white lip ring of the oculus, rounded section
    prof = []
    for j in range(17):
        t = j / 16 * 2 * pi
        prof.append((6.75 + 0.8 * cos(t), OCULUS_Z - 0.9 + 1.35 * sin(t)))
    v, f, uv = revolve(prof, 160)
    g = Geo(); g.add(v, f, 0, uv)
    g.build("oculus_ring", [MAT["oculus"]], coll)
    # inner funnel cylinder up into the roof (glowing acrylic)
    prof = [(5.95, OCULUS_Z - 0.4), (6.0, OCULUS_Z + 2.0), (6.6, OCULUS_Z + 5.5), (8.0, OCULUS_Z + 7.5)]
    v, f, uv = revolve(prof, 160)
    g = Geo(); g.add(v, f, 0, uv)
    ob = g.build("oculus_funnel", [MAT["oculus_in"]], coll)
    ob.visible_shadow = False


def basin_profile():
    return [(4.6, -7.0), (5.5, -5.5), (7.5, -3.6), (10.0, -1.8), (12.6, -0.45), (13.2, -0.1), (13.4, 0.0)]


def build_basin(coll):
    v, f, uv = revolve(basin_profile(), 160)
    g = Geo(); g.add(v, f, 0, uv)
    g.build("basin_bowl", [MAT["bowl"]], coll)
    # foam ring where water hits
    v, f, uv = revolve([(3.0, -5.9), (5.0, -5.85), (6.2, -5.0)], 96)
    g = Geo(); g.add(v, f, 0, uv)
    g.build("basin_foam", [MAT["foam"]], coll)
    # deck (z=0) out to first terrace
    v, f, uv = revolve([(13.4, 0.0), (21.0, 0.0)], 160)
    g = Geo(); g.add(v, f, 0, uv)
    g.build("deck", [MAT["deck"]], coll, smooth=False)
    # glass balustrade + steel top rail
    v, f, uv = revolve([(13.55, 0.0), (13.55, 1.05)], 160)
    g = Geo(); g.add(v, f, 0, uv)
    g.build("rail_glass", [MAT["railglass"]], coll, smooth=False)
    prof = [(13.55 + 0.05 * cos(t * 2 * pi / 8), 1.08 + 0.05 * sin(t * 2 * pi / 8)) for t in range(9)]
    v, f, uv = revolve(prof, 160)
    g = Geo(); g.add(v, f, 0, uv)
    g.build("rail_top", [MAT["steel"]], coll)


def r_water(z):
    """Waterfall column radius vs height (basin z=0, oculus z=40)."""
    t = max(0.0, min(1.0, z / 40.0))
    return 3.55 + 2.05 * t ** 1.7 + 0.9 * math.exp(-(z + 4.5) / 3.0)


def build_waterfall(coll):
    zs = [-6.0 + i * 0.5 for i in range(int((40.6 + 6.0) / 0.5) + 1)]
    scales = [1.0, 0.955, 1.04, 0.88]
    for k, s in enumerate(scales):
        prof = [(r_water(z) * s, z) for z in zs]
        v, f, uv = revolve(prof, 192, phase=k * 0.17)
        g = Geo(); g.add(v, f, 0, uv)
        ob = g.build(f"vortex_shell_{k}", [MAT["water_shell"][k]], coll)
        ob.visible_shadow = False      # thin water sheets: don't darken the basin
        ob.visible_diffuse = k < 2     # only outer sheets show up in bounce light
    # mist core volume (closed)
    prof = [(0.0, -5.0)] + [(r_water(z) * 0.82, z) for z in zs[2:]] + [(0.0, 40.6)]
    v, f, uv = revolve(prof, 64)
    g = Geo(); g.add(v, f, 0, uv)
    ob = g.build("vortex_mist", [MAT["mist"]], coll)
    # spray cloud at the basin: unit sphere scaled
    v, f, uv = revolve([(sin(pi * i / 16) * 1.0, -cos(pi * i / 16) * 1.0) for i in range(17)], 48)
    g = Geo(); g.add(v, f, 0, uv)
    ob = g.build("vortex_spray", [MAT["spray"]], coll)
    ob.scale = (10.5, 10.5, 7.0)
    ob.location = (0, 0, -3.0)


# terraces ---------------------------------------------------------------------
TER_R = [21.0, 26.0, 32.0, 39.0, 47.0, 56.0, 66.0, 77.0, 89.0, 102.0, 116.0, 131.0]
TER_H = [1.2, 3.0, 5.5, 8.5, 12.5, 17.0, 22.0, 27.5, 33.5, 39.5, 45.0]
CAM_THETA = -pi / 2   # camera sits on the -Y side


def ter_gain(theta):
    """Terraces are lower on the camera side so the basin stays visible."""
    d = (1 - cos(theta - CAM_THETA)) / 2      # 0 near side .. 1 far side
    return 0.42 + 0.58 * d ** 0.8


def ter_radius(k, theta):
    return TER_R[k] + 1.6 * sin(3 * theta + k * 1.3) + 1.1 * noise.noise(Vector((cos(theta) * 2, sin(theta) * 2, k)))


def ter_height(k, theta):
    return TER_H[k] * ter_gain(theta)


def ground_z(x, y):
    r = math.hypot(x, y)
    th = math.atan2(y, x)
    if r < TER_R[0]:
        return 0.0 if r > 13.4 else None
    z = 0.0
    for k in range(len(TER_H)):
        if r >= ter_radius(k, th):
            z = ter_height(k, th)
    return z


def build_terraces(coll):
    segs = 288
    g = Geo()
    for k in range(len(TER_H)):
        ring_in, ring_out, wall_lo = [], [], []
        for j in range(segs + 1):
            th = 2 * pi * j / segs
            r0 = ter_radius(k, th)
            r1 = ter_radius(k + 1, th) if k + 1 < len(TER_H) else TER_R[-1]
            h = ter_height(k, th)
            hlo = ter_height(k - 1, th) if k > 0 else 0.0
            c, s = cos(th), sin(th)
            ring_in.append((r0 * c, r0 * s, h))
            ring_out.append(((r1 + 0.3) * c, (r1 + 0.3) * s, h))
            wall_lo.append(((r0 - 0.15) * c, (r0 - 0.15) * s, hlo - 0.2))
        base = len(g.v)
        verts = ring_in + ring_out + wall_lo
        faces_top, faces_wall = [], []
        n = segs + 1
        for j in range(segs):
            faces_top.append((j, j + 1, n + j + 1, n + j))
            faces_wall.append((2 * n + j, 2 * n + j + 1, j + 1, j))
        g.add(verts, faces_top, 0)
        # walls reuse the same vertex block: add faces directly
        for f in faces_wall:
            g.f.append(tuple(i + base for i in f)); g.m.append(1); g.uv.append([(0, 0)] * 4)
    g.build("terraces", [MAT["soil"], MAT["wall"]], coll, smooth=False)
    # outer perimeter wall up to the dome edge (mostly hidden by foliage / haze)
    prof = [(130.0, 0.0), (130.0, z_dome(128.0) + 1)]
    v, f, uv = revolve(prof, 160)
    g = Geo(); g.add(v, f, 0, uv)
    g.build("perimeter", [MAT["wall"]], coll, smooth=False)


# guideway + train -------------------------------------------------------------
GW_Y0 = 8.0
GW_TOP = 18.4


def gw_y(x):
    return GW_Y0 + 0.0028 * x * x


def build_guideway(coll):
    g = Geo()
    xs = [-110 + i * 2.0 for i in range(111)]
    prof = [(-2.1, 0.0), (2.1, 0.0), (2.1, -0.5), (1.4, -1.9), (-1.4, -1.9), (-2.1, -0.5)]
    verts, faces = [], []
    for i, x in enumerate(xs):
        y = gw_y(x)
        dx = 1.0; dy = 0.0056 * x
        t = Vector((dx, dy, 0)).normalized()
        nrm = Vector((-t.y, t.x, 0))
        for (u, w) in prof:
            p = Vector((x, y, GW_TOP)) + nrm * u + Vector((0, 0, w))
            verts.append(p)
    np_ = len(prof)
    for i in range(len(xs) - 1):
        for j in range(np_):
            j2 = (j + 1) % np_
            faces.append((i * np_ + j, i * np_ + j2, (i + 1) * np_ + j2, (i + 1) * np_ + j))
    g.add(verts, faces, 0)
    # low parapet / running rails
    for side in (-1.9, 1.9):
        pts = []
        for x in xs:
            t = Vector((1.0, 0.0056 * x, 0)).normalized()
            nrm = Vector((-t.y, t.x, 0))
            pts.append(Vector((x, gw_y(x), GW_TOP + 0.25)) + nrm * side)
        v, f, uv = tube(pts, [0.18] * len(pts), 4)
        g.add(v, f, 0, uv)
    # columns
    for x in (-24.0, 27.0, -70.0, 72.0):
        y = gw_y(x)
        zg = ground_z(x, y) or 0.0
        pts = [Vector((x, y, zg - 1)), Vector((x, y, GW_TOP - 1.9))]
        v, f, uv = tube(pts, [1.05, 1.05], 24)
        g.add(v, f, 0, uv)
        # flared capital
        pts = [Vector((x, y, GW_TOP - 3.2)), Vector((x, y, GW_TOP - 1.85))]
        v, f, uv = tube(pts, [1.05, 1.8], 24)
        g.add(v, f, 0, uv)
    g.build("guideway", [MAT["concrete"]], coll)


def build_train(coll, lead_x=3.0):
    """Three white cars (~12 m) on the guideway heading +X; nose of the lead car
    slipping behind the waterfall veil."""
    car_len, gap, w, h = 11.6, 0.9, 2.8, 3.25
    for c in range(3):
        cx = lead_x - car_len / 2 - c * (car_len + gap)
        bpy.ops.mesh.primitive_cube_add(size=1)
        ob = bpy.context.object
        for cc in ob.users_collection:
            cc.objects.unlink(ob)
        coll.objects.link(ob)
        ob.name = f"train_car_{c}"
        me = ob.data
        # object-space box: x in [-L/2, L/2], z in [0, h]
        for v in me.vertices:
            v.co.x *= car_len
            v.co.y *= w
            v.co.z = (v.co.z + 0.5) * h
            if c == 0 and v.co.x > 0 and v.co.z > h * 0.5:
                v.co.x -= 1.1          # raked nose
            if c == 2 and v.co.x < 0 and v.co.z > h * 0.5:
                v.co.x += 1.1
        me.materials.append(MAT["train"])
        bv = ob.modifiers.new("bevel", "BEVEL")
        bv.width = 0.55
        bv.segments = 5
        sub = ob.modifiers.new("sub", "SUBSURF")
        sub.levels = 1
        sub.render_levels = 1
        for p in me.polygons:
            p.use_smooth = True
        y = gw_y(cx)
        ob.location = (cx, y, GW_TOP + 0.35)
        ob.rotation_euler = (0, 0, math.atan(0.0056 * cx))
        # bogie / connector
        if c < 2:
            bpy.ops.mesh.primitive_cube_add(size=1)
            cn = bpy.context.object
            for cc in cn.users_collection:
                cc.objects.unlink(cn)
            coll.objects.link(cn)
            gx = cx - car_len / 2 - gap / 2
            cn.location = (gx, gw_y(gx), GW_TOP + 0.35 + h * 0.45)
            cn.scale = (gap + 0.4, w * 0.8, h * 0.8)
            cn.rotation_euler = (0, 0, math.atan(0.0056 * gx))
            cn.data.materials.append(MAT["dark"])


# people -----------------------------------------------------------------------
def build_people_lib(lib):
    colls = []
    for k in range(4):
        c = new_coll(f"person_{k}", lib)
        rnd = random.Random(100 + k)
        g = Geo()
        h = 1.6 + rnd.random() * 0.2
        # legs
        for sx in (-0.09, 0.09):
            v, f, uv = tube([Vector((sx, 0, 0)), Vector((sx, 0, h * 0.47))], [0.075, 0.085], 6)
            g.add(v, f, 1 if k % 2 else 0, uv)
        # torso
        v, f, uv = tube([Vector((0, 0, h * 0.45)), Vector((0, 0, h * 0.62)), Vector((0, 0, h * 0.8))],
                        [0.17, 0.2, 0.19], 8)
        g.add(v, f, 0, uv)
        # head
        prof = [(0.105 * sin(pi * i / 6), h * 0.885 - 0.11 * cos(pi * i / 6)) for i in range(7)]
        v, f, uv = revolve(prof, 8)
        g.add(v, f, 2, uv)
        ob = g.build(f"person_mesh_{k}", [MAT["people"], MAT["dark"], MAT["skin"]], c)
        colls.append(c)
    return colls


# ----------------------------------------------------------------------------- vegetation library
def make_frond(g, M, L_, n_pairs, leaflet_len, leaflet_w, arch=0.35, droop=0.9, angle=55, mat=0,
               rnd=None, rachis_r=0.035, leaflet_droop=0.35):
    """Pinnate palm / fern frond, rachis along local +Y arching up then drooping."""
    rnd = rnd or RNG
    pts = []
    nr = 10
    for i in range(nr + 1):
        t = i / nr
        pts.append(Vector((0, L_ * t, L_ * (arch * t - droop * 0.5 * t * t))))
    v, f, uv = tube(pts, [rachis_r * (1 - 0.8 * i / nr) + 0.004 for i in range(nr + 1)], 4)
    g.add(v, f, mat + 1, uv, M)

    def at(t):
        return Vector((0, L_ * t, L_ * (arch * t - droop * 0.5 * t * t)))

    for j in range(n_pairs):
        t = 0.06 + 0.92 * (j + 0.5) / n_pairs
        p = at(t)
        tan = (at(min(1, t + 0.01)) - at(max(0, t - 0.01))).normalized()
        side_axis = Vector((1, 0, 0))
        ll = leaflet_len * (math.sin(pi * (0.12 + 0.88 * t)) ** 0.6) * (0.85 + 0.3 * rnd.random())
        for s in (-1, 1):
            a = radians(angle + rnd.uniform(-8, 8))
            d = (tan * cos(a) + side_axis * s * sin(a))
            d.z -= 0.25 + rnd.uniform(-0.1, 0.15)
            Ml = frame_matrix(p, d, Vector((0, 0, 1)) + tan * 0.3)
            bv, bf, buv = blade(ll, leaflet_w * (0.8 + 0.4 * rnd.random()), 3,
                                droop=leaflet_droop, fold=0.25)
            g.add(bv, bf, mat, buv, M @ Ml)


def lib_palm(lib, name, seed, height, nfronds, frond_len, mat_leaf="palm", lean=0.08):
    rnd = random.Random(seed)
    c = new_coll(name, lib)
    g = Geo()
    # trunk
    pts, rad = [], []
    bend = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), 0)).normalized() * lean * height
    for i in range(13):
        t = i / 12
        pts.append(Vector((0, 0, height * t)) + bend * t * t)
        rad.append(0.24 * (1.25 - 0.45 * t) * (1 + 0.04 * sin(t * 60)))
    v, f, uv = tube(pts, rad, 10)
    g.add(v, f, 2, uv)
    top = pts[-1]
    for k in range(nfronds):
        az = 2 * pi * k / nfronds * 2.618 + rnd.uniform(-0.2, 0.2)    # golden-angle spiral
        el = radians(rnd.uniform(-35, 55))
        M = Matrix.Translation(top) @ Matrix.Rotation(az, 4, "Z") @ Matrix.Rotation(el, 4, "X")
        make_frond(g, M, frond_len * rnd.uniform(0.8, 1.1), 26, frond_len * 0.32, 0.045,
                   arch=0.25, droop=0.75, angle=60, mat=0, rnd=rnd)
    g.build(name + "_mesh", [MAT[mat_leaf], MAT["stem"], MAT["palm_trunk"]], c)
    return c, height + frond_len * 0.3, frond_len * 0.9


def leaf_cluster(g, rnd, center, radii, count, leaf_len, leaf_w, mat=0, shell=0.55):
    for _ in range(count):
        # bias towards the shell of the ellipsoid (but keep some interior leaves)
        d = Vector((rnd.gauss(0, 1), rnd.gauss(0, 1), rnd.gauss(0, 1))).normalized()
        u = rnd.random() ** shell
        off = Vector((d.x * radii[0], d.y * radii[1], d.z * radii[2])) * u
        p = center + off
        out = off.normalized() if off.length > 1e-3 else Vector((0, 0, 1))
        ydir = (out * 0.7 + Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-0.6, 0.3)))).normalized()
        up = (out + Vector((0, 0, 1.2))).normalized()
        M = frame_matrix(p, ydir, up)
        s = rnd.uniform(0.75, 1.25)
        bv, bf, buv = blade(leaf_len * s, leaf_w * s, 2, droop=0.2, fold=0.2)
        g.add(bv, bf, mat, buv, M)


def lib_broadleaf(lib, name, seed, height, crown_r, n_branch, leaf_len, leaf_w, density, mat_leaf,
                  flat=0.7, spread=1.0):
    """Tropical broadleaf tree: leaning trunk, primary + secondary branches, many small
    irregular leaf clusters at the twig ends so the crown silhouette is broken and gappy."""
    rnd = random.Random(seed)
    c = new_coll(name, lib)
    g = Geo()
    trunk_h = height - crown_r * 1.2 * flat
    lean = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), 0)) * 0.1 * trunk_h
    pts = [Vector((0, 0, trunk_h * i / 8)) + lean * (i / 8) ** 2 for i in range(9)]
    base_r = 0.05 * height ** 0.8
    v, f, uv = tube(pts, [base_r * (1.3 - 0.75 * i / 8) for i in range(9)], 8)
    g.add(v, f, 1, uv)
    top = pts[-1]
    tips = [(top + Vector((0, 0, crown_r * 0.45 * flat)), 0.55)]
    for b in range(n_branch):
        az = 2 * pi * b / n_branch + rnd.uniform(-0.5, 0.5)
        el = radians(rnd.uniform(18, 60))
        ln = crown_r * spread * rnd.uniform(0.55, 0.9)
        d = Vector((cos(az) * cos(el), sin(az) * cos(el), sin(el)))
        start = top - Vector((0, 0, rnd.uniform(0, trunk_h * 0.3)))
        end = start + d * ln
        v, f, uv = tube([start, start.lerp(end, 0.5) + Vector((0, 0, 0.15)), end],
                        [base_r * 0.6, base_r * 0.45, base_r * 0.3], 6)
        g.add(v, f, 1, uv)
        for s2 in range(rnd.randint(2, 4)):
            az2 = az + rnd.uniform(-0.9, 0.9)
            el2 = radians(rnd.uniform(-5, 45))
            ln2 = crown_r * spread * rnd.uniform(0.3, 0.6)
            d2 = Vector((cos(az2) * cos(el2), sin(az2) * cos(el2), sin(el2)))
            st2 = start.lerp(end, rnd.uniform(0.5, 1.0))
            e2 = st2 + d2 * ln2
            v, f, uv = tube([st2, e2], [base_r * 0.25, base_r * 0.12], 5)
            g.add(v, f, 1, uv)
            tips.append((e2 + Vector((0, 0, 0.15)), rnd.uniform(0.28, 0.45)))
    w_tot = sum(w ** 2 for _, w in tips)
    for cc, w in tips:
        rr = crown_r * w
        n = int(density * w ** 2 / w_tot)
        leaf_cluster(g, rnd, cc, (rr * 1.15, rr * 1.15, rr * 0.75 * flat + rr * 0.2), n, leaf_len, leaf_w, 0,
                     shell=0.45)
    g.build(name + "_mesh", [MAT[mat_leaf], MAT["bark"]], c)
    return c, height, crown_r * spread


def lib_shrub(lib, name, seed, radius, height, count, leaf_len, leaf_w, mat_leaf):
    rnd = random.Random(seed)
    c = new_coll(name, lib)
    g = Geo()
    leaf_cluster(g, rnd, Vector((0, 0, height * 0.55)), (radius, radius, height * 0.5),
                 count, leaf_len, leaf_w, 0)
    g.build(name + "_mesh", [MAT[mat_leaf]], c)
    return c, height, radius


def lib_fern(lib, name, seed, nfronds, frond_len, mat_leaf):
    rnd = random.Random(seed)
    c = new_coll(name, lib)
    g = Geo()
    for k in range(nfronds):
        az = 2 * pi * k / nfronds + rnd.uniform(-0.3, 0.3)
        el = radians(rnd.uniform(20, 65))
        M = Matrix.Rotation(az, 4, "Z") @ Matrix.Rotation(el, 4, "X")
        make_frond(g, M, frond_len * rnd.uniform(0.7, 1.1), 18, frond_len * 0.22, 0.04,
                   arch=0.3, droop=1.2, angle=70, mat=0, rnd=rnd, rachis_r=0.015, leaflet_droop=0.15)
    g.build(name + "_mesh", [MAT[mat_leaf], MAT["stem"]], c)
    return c, frond_len * 0.6, frond_len * 0.8


def lib_bigleaf(lib, name, seed, nleaves, blade_len, blade_w, petiole, shape, mat_leaf):
    """Banana / alocasia: big blades on long petioles."""
    rnd = random.Random(seed)
    c = new_coll(name, lib)
    g = Geo()
    for k in range(nleaves):
        az = 2 * pi * k / nleaves + rnd.uniform(-0.4, 0.4)
        el = radians(rnd.uniform(35, 75))
        pl = petiole * rnd.uniform(0.7, 1.15)
        d = Vector((cos(az) * cos(el), sin(az) * cos(el), sin(el)))
        tip = d * pl
        v, f, uv = tube([Vector((0, 0, 0)), d * pl * 0.5 + Vector((0, 0, 0.05)), tip],
                        [0.05, 0.035, 0.025], 6)
        g.add(v, f, 1, uv)
        bdir = Vector((cos(az), sin(az), rnd.uniform(-0.1, 0.5))).normalized()
        M = frame_matrix(tip, bdir, Vector((0, 0, 1)) + Vector((rnd.uniform(-.4, .4), rnd.uniform(-.4, .4), 0)))
        s = rnd.uniform(0.8, 1.15)
        bv, bf, buv = blade(blade_len * s, blade_w * s, 8, droop=rnd.uniform(0.15, 0.4), fold=0.12, shape=shape)
        g.add(bv, bf, 0, buv, M)
    g.build(name + "_mesh", [MAT[mat_leaf], MAT["stem"]], c)
    return c, petiole + blade_len * 0.3, blade_len


def build_plant_library(lib):
    P = {}
    P["palm_a"] = lib_palm(lib, "palm_a", 1, 11.0, 15, 4.2)
    P["palm_b"] = lib_palm(lib, "palm_b", 2, 7.0, 13, 3.6)
    P["palm_c"] = lib_palm(lib, "palm_c", 3, 15.0, 16, 4.5, lean=0.04)
    P["tree_a"] = lib_broadleaf(lib, "tree_a", 11, 12.0, 4.0, 6, 0.45, 0.18, 2600, "leaf_dark")
    P["tree_b"] = lib_broadleaf(lib, "tree_b", 12, 9.0, 3.4, 5, 0.40, 0.16, 2200, "leaf_mid")
    P["tree_c"] = lib_broadleaf(lib, "tree_c", 13, 7.5, 3.0, 5, 0.24, 0.10, 3200, "leaf_light")  # fine-leaf bright
    P["tree_d"] = lib_broadleaf(lib, "tree_d", 14, 17.0, 5.0, 7, 0.5, 0.2, 3600, "leaf_dark")
    P["tree_e"] = lib_broadleaf(lib, "tree_e", 15, 11.0, 5.2, 7, 0.26, 0.11, 3400, "leaf_light",
                                flat=0.45, spread=1.15)                                        # umbrella rain-tree
    P["tree_f"] = lib_broadleaf(lib, "tree_f", 16, 14.0, 2.6, 4, 0.55, 0.22, 1600, "leaf_mid", flat=1.4)  # columnar
    P["shrub_a"] = lib_shrub(lib, "shrub_a", 21, 1.3, 1.6, 420, 0.28, 0.12, "leaf_mid")
    P["shrub_b"] = lib_shrub(lib, "shrub_b", 22, 1.8, 2.2, 600, 0.32, 0.14, "leaf_dark")
    P["shrub_c"] = lib_shrub(lib, "shrub_c", 23, 1.0, 1.0, 300, 0.2, 0.09, "leaf_light")
    P["fern_a"] = lib_fern(lib, "fern_a", 31, 14, 1.6, "leaf_mid")
    P["fern_b"] = lib_fern(lib, "fern_b", 32, 18, 2.2, "leaf_light")
    P["banana_a"] = lib_bigleaf(lib, "banana_a", 41, 7, 2.4, 0.42, 2.4, "oblong", "banana")
    P["alocasia_a"] = lib_bigleaf(lib, "alocasia_a", 42, 6, 1.2, 0.5, 1.2, "heart", "banana")
    P["fg_leaf"] = lib_bigleaf(lib, "fg_leaf", 44, 5, 1.3, 0.55, 1.3, "heart", "leaf_dark")
    P["fg_banana"] = lib_bigleaf(lib, "fg_banana", 45, 6, 2.4, 0.42, 2.4, "oblong", "palm")
    P["alocasia_b"] = lib_bigleaf(lib, "alocasia_b", 43, 5, 0.9, 0.4, 0.8, "heart", "leaf_mid")
    return P


# ----------------------------------------------------------------------------- scattering
def make_instance(coll, name, src, loc, rot_z, scale, parent_coll):
    e = bpy.data.objects.new(name, None)
    e.instance_type = "COLLECTION"
    e.instance_collection = src
    e.location = loc
    e.rotation_euler = (0, 0, rot_z)
    e.scale = (scale, scale, scale * RNG.uniform(0.9, 1.1))
    parent_coll.objects.link(e)
    return e


def scatter_vegetation(P, cam, coll):
    """Fill the terraces with layered vegetation, but keep:
       - the waterfall and basin unobstructed (plants in front of them are capped),
       - the train mostly visible,
       - the top of the frame free of foliage."""
    cam_loc = cam.location
    v_basin = Vector((0, -13.5, 0.0)) - cam_loc   # sight line to near basin rim
    n_inst = 0

    def proj(p):
        return world_to_camera_view(scene, cam, p)

    def place(kind, x, y, z, scale, allow_top=0.74):
        nonlocal n_inst
        src, h, rad = P[kind]
        hh = h * scale
        base = Vector((x, y, z))
        c0 = proj(base)
        ctop = proj(base + Vector((0, 0, hh)))
        if c0.z < 3.0:
            return False
        # frustum cull with margin for canopy width
        mx = rad * scale / max(c0.z, 1.0) * 1.6
        if ctop.x < -mx - 0.05 or ctop.x > 1 + mx + 0.05 or c0.x < -mx - 0.05 or c0.x > 1 + mx + 0.05:
            return False
        if c0.y > 1.05:
            return False
        if ctop.y > allow_top:
            return False
        # protect waterfall + basin: plants nearer than the waterfall within its screen band
        if y < -6 and abs(c0.x - 0.5) < 0.17 + rad * scale / c0.z * 0.6:
            # height of sight line from camera to basin rim at this distance
            t = (y - cam_loc.y) / v_basin.y
            z_line = cam_loc.z + v_basin.z * t
            if z + hh > z_line - 0.6:
                return False
        # protect the train (left half, guideway at ~18.4-21.7 m)
        if y < GW_Y0 - 4 and c0.x < 0.52 and ctop.y > 0.43 and RNG.random() < 0.85:
            v_tr = Vector((x, gw_y(x), GW_TOP)) - cam_loc
            t = (y - cam_loc.y) / v_tr.y if abs(v_tr.y) > 1e-3 else 1
            z_line = cam_loc.z + v_tr.z * t
            if z + hh > z_line - 0.3:
                return False
        n_inst += 1
        make_instance(coll, f"{kind}_{n_inst}", src, base, RNG.uniform(0, 2 * pi), scale, coll)
        return True

    canopy_kinds = [("tree_a", 2.0), ("tree_b", 2.0), ("tree_c", 1.4), ("tree_d", 1.5), ("tree_e", 1.3),
                    ("tree_f", 1.1), ("palm_a", 3.0), ("palm_b", 2.0), ("palm_c", 2.6)]
    under_kinds = [("shrub_a", 3), ("shrub_b", 3), ("shrub_c", 1.5), ("fern_a", 2.5), ("fern_b", 2),
                   ("banana_a", 1.4), ("alocasia_a", 1.6), ("alocasia_b", 1.6)]

    def pick(kinds):
        tot = sum(w for _, w in kinds)
        x = RNG.random() * tot
        for k, w in kinds:
            x -= w
            if x <= 0:
                return k
        return kinds[-1][0]

    for k in range(len(TER_H)):
        r_in, r_out = TER_R[k], TER_R[k + 1]
        area = pi * (r_out ** 2 - r_in ** 2)
        for layer, kinds, spacing, smin, smax in (("canopy", canopy_kinds, 5.8, 0.75, 1.3),
                                                 ("under", under_kinds, 2.3, 0.7, 1.4)):
            n = int(area / (spacing * spacing))
            for _ in range(n):
                th = RNG.uniform(0, 2 * pi)
                rr = math.sqrt(RNG.uniform(r_in ** 2, r_out ** 2))
                x, y = rr * cos(th), rr * sin(th)
                # stay off the terrace lip a little
                if rr < ter_radius(k, th) + 0.8:
                    continue
                if k + 1 < len(TER_H) and rr > ter_radius(k + 1, th) - 0.4:
                    continue
                z = ter_height(k, th)
                kind = pick(kinds)
                if layer == "canopy" and k <= 1 and kind in ("tree_d", "palm_c"):
                    kind = "tree_c"
                sc = RNG.uniform(smin, smax)
                dcam = (Vector((x, y, 0)) - Vector((cam_loc.x, cam_loc.y, 0))).length
                if dcam < (26 if layer == "canopy" else 12):
                    continue
                place(kind, x, y, z, sc)
            # trailing plants along the lip so walls read as planted terraces
        n_lip = int(2 * pi * r_in / 2.6)
        for j in range(n_lip):
            th = 2 * pi * (j + RNG.random() * 0.6) / n_lip
            rr = ter_radius(k, th) + 0.5
            x, y = rr * cos(th), rr * sin(th)
            if (Vector((x, y, 0)) - Vector((cam_loc.x, cam_loc.y, 0))).length < 9:
                continue
            place(pick([("shrub_a", 2), ("fern_a", 2), ("shrub_c", 1), ("fern_b", 1)]),
                  x, y, ter_height(k, th) - 0.2, RNG.uniform(0.8, 1.3))
    # planting on the deck edge between basin and first terrace (low)
    for j in range(140):
        th = RNG.uniform(0, 2 * pi)
        rr = RNG.uniform(18.0, 20.5)
        place(pick([("shrub_a", 2), ("shrub_c", 2), ("fern_a", 1), ("alocasia_b", 1.5), ("tree_c", 0.6)]),
              rr * cos(th), rr * sin(th), 0.0, RNG.uniform(0.7, 1.2))
    return n_inst


def foreground_leaves(P, cam, coll):
    """A few big soft banana / alocasia leaves close to the lens in the bottom corners."""
    cm = cam.matrix_world
    spots = [((-0.95, -1.0), 3.6, "fg_banana", 0.62, 0.9), ((0.97, -1.02), 3.4, "fg_leaf", 1.0, 2.6),
             ((-1.08, -0.52), 6.0, "fg_leaf", 0.95, -0.6)]
    tan_v = 12.0 / cam.data.lens
    aspect = scene.render.resolution_x / scene.render.resolution_y
    for i, ((nx, ny), depth, kind, sc, rz) in enumerate(spots):
        d_cam = Vector((nx * tan_v * aspect * depth, ny * tan_v * depth, -depth))
        p = cm @ d_cam
        src = P[kind][0]
        # plant base sits below the visible point so blades enter from the frame edge
        base = p - Vector((0, 0, P[kind][1] * sc * 0.75))
        make_instance(coll, f"fg_{i}", src, base, rz, sc, coll)


def framing_plants(P, coll):
    """Hero plants that frame the composition like the reference photo: a tall palm on the
    left whose fronds arch into the frame, and a broadleaf tree on the right."""
    make_instance(coll, "frame_palm_l", P["palm_c"][0], Vector((-10.5, -43.0, ground_z(-10.5, -43.0))), 0.4, 1.15, coll)
    make_instance(coll, "frame_palm_l2", P["palm_c"][0], Vector((-7.5, -56.0, ground_z(-7.5, -56.0))), 1.2, 1.12, coll)
    make_instance(coll, "frame_banana_l", P["banana_a"][0], Vector((-11.0, -44.0, ground_z(-11.0, -44.0))), 2.2, 1.3, coll)
    make_instance(coll, "frame_tree_r", P["tree_a"][0], Vector((12.5, -40.0, ground_z(12.5, -40.0))), 0.9, 1.0, coll)


def scatter_people(colls, coll):
    n = 0
    for i in range(110):
        th = RNG.uniform(0, 2 * pi)
        if sin(th) < -0.55 and RNG.random() < 0.7:    # fewer on the near side (hidden anyway)
            continue
        rr = RNG.uniform(14.0, 16.5) if RNG.random() < 0.75 else RNG.uniform(16.5, 19.5)
        e = bpy.data.objects.new(f"person_{i}", None)
        e.instance_type = "COLLECTION"
        e.instance_collection = RNG.choice(colls)
        e.location = (rr * cos(th), rr * sin(th), 0.0)
        e.rotation_euler = (0, 0, RNG.uniform(0, 2 * pi))
        s = RNG.uniform(0.93, 1.07)
        e.scale = (s, s, s)
        coll.objects.link(e)
        n += 1
    return n


# ----------------------------------------------------------------------------- world, light, camera
def build_world():
    w = bpy.data.worlds.new("overcast")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = N(nt, "ShaderNodeTexCoord")
    sep = N(nt, "ShaderNodeSeparateXYZ")
    L(nt, tc.outputs["Generated"], sep.inputs[0])
    # soft overcast cloud variation
    nz = N(nt, "ShaderNodeTexNoise", Scale=2.5, Detail=4.0)
    L(nt, tc.outputs["Generated"], nz.inputs["Vector"])
    r = ramp(nt, sep.outputs[2], [(0.0, (0.55, 0.55, 0.52, 1)), (0.45, (1.0, 0.975, 0.94, 1)),
                                  (1.0, (0.96, 0.98, 1.0, 1))])
    mixc = N(nt, "ShaderNodeMix", _data_type="RGBA", _blend_type="MULTIPLY", Factor=0.18)
    L(nt, r.outputs[0], mixc.inputs[6])
    L(nt, nz.outputs["Color"], mixc.inputs[7])
    bg = N(nt, "ShaderNodeBackground", Strength=2.8)
    L(nt, mixc.outputs[2], bg.inputs["Color"])
    L(nt, bg.outputs[0], out.inputs[0])

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.5
    sun.angle = radians(9)
    sun.color = (1.0, 0.94, 0.85)
    so = bpy.data.objects.new("sun", sun)
    ROOT.objects.link(so)
    so.rotation_euler = (radians(28), radians(0), radians(150))   # high, from behind-left of the falls

    # soft fill light from the oculus down the vortex (light pouring through the opening)
    lt = bpy.data.lights.new("oculus_fill", "AREA")
    lt.shape = "DISK"
    lt.size = 11.0
    lt.energy = 15000.0
    lt.color = (1.0, 0.98, 0.95)
    lo = bpy.data.objects.new("oculus_fill", lt)
    ROOT.objects.link(lo)
    lo.location = (0, 0, OCULUS_Z + 3.0)
    lo.visible_camera = False


def build_camera():
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 24.0
    vfov = radians(60.0)
    cd.lens = 12.0 / math.tan(vfov / 2)
    cd.clip_start = 0.3
    cd.clip_end = 600
    cd.dof.use_dof = True
    cd.dof.focus_distance = 74.0
    cd.dof.aperture_fstop = 0.45
    cd.dof.aperture_blades = 7
    cam = bpy.data.objects.new("cam", cd)
    ROOT.objects.link(cam)
    cam.location = (0.0, -72.0, 25.0)
    cam.rotation_euler = (radians(90 + 0.0), 0, 0)
    scene.camera = cam
    bpy.context.view_layer.update()
    return cam


def setup_render():
    r = scene.render
    r.engine = "CYCLES"
    s = ARGS["scale"]
    r.resolution_x = int(round(1440 * s))
    r.resolution_y = int(round(2560 * s))
    r.resolution_percentage = 100
    r.threads_mode = "FIXED"
    r.threads = ARGS["threads"]
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.filepath = ARGS["out"]
    c = scene.cycles
    c.device = "CPU"
    c.samples = ARGS["samples"]
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.025
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    c.max_bounces = 6
    c.diffuse_bounces = 2
    c.glossy_bounces = 2
    c.transmission_bounces = 4
    c.volume_bounces = 1
    c.transparent_max_bounces = 20
    c.sample_clamp_direct = 0
    c.sample_clamp_indirect = 4.0
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.blur_glossy = 1.0
    if hasattr(c, "volume_step_rate"):
        c.volume_step_rate = 4.0
        c.volume_max_steps = 256
    c.seed = 3
    vs = scene.view_settings
    vs.view_transform = "AgX"
    vs.look = "AgX - Medium High Contrast"
    vs.exposure = 0.35
    vs.gamma = 1.0
    scene.display_settings.display_device = "sRGB"


def setup_compositor():
    """Aerial perspective as a post effect (much cheaper than a scene-wide volume):
    blend towards a warm pale haze colour by the mist pass, plus a gentle bloom."""
    vl = bpy.context.view_layer
    vl.use_pass_mist = True
    ms = scene.world.mist_settings
    ms.start, ms.depth, ms.falloff = 12.0, 230.0, "LINEAR"
    ng = bpy.data.node_groups.new("comp", "CompositorNodeTree")
    scene.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    out = ng.nodes.new("NodeGroupOutput")
    fac = ng.nodes.new("ShaderNodeMath")
    fac.operation = "POWER"
    fac.inputs[1].default_value = 1.3
    ng.links.new(rl.outputs["Mist"], fac.inputs[0])
    fac2 = ng.nodes.new("ShaderNodeMath")
    fac2.operation = "MULTIPLY"
    fac2.inputs[1].default_value = HAZE_AMOUNT
    ng.links.new(fac.outputs[0], fac2.inputs[0])
    mix = ng.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    ng.links.new(fac2.outputs[0], mix.inputs[0])
    ng.links.new(rl.outputs["Image"], mix.inputs[6])
    mix.inputs[7].default_value = HAZE_COLOR
    last = mix.outputs[2]
    gl = ng.nodes.new("CompositorNodeGlare")
    for k, v in (("Type", "Bloom"), ("Quality", "Medium"), ("Highlights Threshold", 1.5),
                 ("Strength", 0.22), ("Size", 0.65)):
        if k in gl.inputs:
            gl.inputs[k].default_value = v
    ng.links.new(last, gl.inputs["Image"])
    last = gl.outputs["Image"]
    ng.links.new(last, out.inputs[0])
    scene.render.use_compositing = True
    scene.render.compositor_device = "CPU"   # no GPU context when bpy runs headless


HAZE_AMOUNT = 0.3
HAZE_COLOR = (0.62, 0.65, 0.62, 1.0)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    setup_render()
    build_materials()
    arch = new_coll("architecture")
    water = new_coll("waterfall")
    veg = new_coll("vegetation")
    ppl = new_coll("people")
    lib = new_coll("LIBRARY", exclude=False)

    cam = build_camera()
    build_world()
    build_dome(arch)
    build_oculus(arch)
    build_basin(arch)
    build_terraces(arch)
    build_guideway(arch)
    build_train(arch)
    build_waterfall(water)
    setup_compositor()

    P = build_plant_library(lib)
    pcolls = build_people_lib(lib)
    bpy.context.view_layer.layer_collection.children["LIBRARY"].exclude = True
    bpy.context.view_layer.update()
    n_veg = scatter_vegetation(P, cam, veg)
    foreground_leaves(P, cam, veg)
    framing_plants(P, veg)
    n_ppl = scatter_people(pcolls, ppl)
    print(f"[jewel] scene built in {time.time() - t0:.1f}s: {n_veg} plant instances, {n_ppl} people")

    os.makedirs(os.path.dirname(ARGS["out"]), exist_ok=True)
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[jewel] rendered {scene.render.resolution_x}x{scene.render.resolution_y} "
          f"@{ARGS['samples']} spp in {time.time() - t1:.1f}s -> {ARGS['out']}")


if __name__ == "__main__":
    main()
