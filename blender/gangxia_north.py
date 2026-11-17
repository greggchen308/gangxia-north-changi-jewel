"""
Gangxia North metro station -- procedural interior, rendered with Cycles.

Hero image for the "Twin Wonders" page. Everything in the frame is modelled
here from primitives: no photos, no downloaded assets.

Usage (Blender as a Python module):
    python gangxia_north.py -- --scale 0.25 --samples 48            # preview 360x640
    python gangxia_north.py -- --scale 0.5  --samples 96            # mid check 720x1280
    python gangxia_north.py -- --scale 1.0  --samples 256           # final 1440x2560
Options (also readable from env GN_SCALE, GN_SAMPLES, GN_OUT, GN_THREADS):
    --scale    fraction of 1440x2560            (default 0.25)
    --samples  Cycles samples per pixel         (default 48)
    --out      output PNG path                  (default out/gangxia_north.png next to this file)
    --threads  render threads                   (default 2)

Coordinates: metres. X across the hall, Y down the hall (away from camera), Z up.
"""

import math
import os
import random
import sys
import time

import bpy  # must come first: bmesh/mathutils are provided by bpy
import bmesh
from mathutils import Matrix, Vector

# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {
        "scale": float(os.environ.get("GN_SCALE", 0.25)),
        "samples": int(os.environ.get("GN_SAMPLES", 48)),
        "out": os.environ.get("GN_OUT", os.path.join(HERE, "out", "gangxia_north.png")),
        "threads": int(os.environ.get("GN_THREADS", 2)),
    }
    i = 0
    while i < len(argv):
        key = argv[i].lstrip("-")
        if key in opts and i + 1 < len(argv):
            opts[key] = type(opts[key])(argv[i + 1])
            i += 2
        else:
            i += 1
    return opts


OPTS = parse_args()
RNG = random.Random(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
SCENE = bpy.context.scene
COLL = SCENE.collection

# ----------------------------------------------------------------------------
# Hall dimensions
# ----------------------------------------------------------------------------
X_MIN, X_MAX = -27.0, 23.0          # hall walls
Y_MIN, Y_MAX = -14.0, 150.0
CEIL_BASE = 15.5                     # underside of lattice at the hall centre
MEZZ_Z = 6.5                         # mezzanine (concourse) floor level
TRACK_X = -12.5                      # centreline of the glazed track vault
PSD_X = -7.9                         # platform screen door line


def ceil_z(x, y):
    """Underside of the lattice: a shallow dome, dropping towards the walls and far end."""
    return CEIL_BASE - 0.0045 * x * x - 0.00006 * max(0.0, y - 40.0) ** 2


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    return m, m.node_tree.nodes, m.node_tree.links


def principled(name, color, rough=0.5, metal=0.0, emit=None, emit_strength=0.0,
               coat=0.0, spec=0.5):
    m, nodes, links = new_mat(name)
    b = nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.05
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1.0)
        b.inputs["Emission Strength"].default_value = emit_strength
    return m


def emission(name, color, strength):
    m, nodes, links = new_mat(name)
    nodes.remove(nodes["Principled BSDF"])
    e = nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1.0)
    e.inputs["Strength"].default_value = strength
    links.new(e.outputs[0], nodes["Material Output"].inputs["Surface"])
    return m


def thin_glass(name, tint=(0.86, 0.93, 0.95), alpha=0.82):
    """Cheap architectural glass: tinted transparency + fresnel gloss (no refraction noise)."""
    m, nodes, links = new_mat(name)
    nodes.remove(nodes["Principled BSDF"])
    out = nodes["Material Output"]
    tr = nodes.new("ShaderNodeBsdfTransparent")
    tr.inputs["Color"].default_value = (*tint, 1)
    gl = nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.02
    gl.inputs["Color"].default_value = (1, 1, 1, 1)
    fr = nodes.new("ShaderNodeFresnel")
    fr.inputs["IOR"].default_value = 1.5
    # a little milky diffuse so large panes register as glass
    df = nodes.new("ShaderNodeBsdfDiffuse")
    df.inputs["Color"].default_value = (0.7, 0.78, 0.8, 1)
    mix_a = nodes.new("ShaderNodeMixShader")
    mix_a.inputs[0].default_value = 1.0 - alpha
    links.new(tr.outputs[0], mix_a.inputs[1])
    links.new(df.outputs[0], mix_a.inputs[2])
    mix_b = nodes.new("ShaderNodeMixShader")
    links.new(fr.outputs[0], mix_b.inputs[0])
    links.new(mix_a.outputs[0], mix_b.inputs[1])
    links.new(gl.outputs[0], mix_b.inputs[2])
    links.new(mix_b.outputs[0], out.inputs["Surface"])
    return m


def terrazzo():
    """Pale grey terrazzo: speckled aggregate, faint large tile joints, glossy with soft variation."""
    m, nodes, links = new_mat("terrazzo")
    b = nodes["Principled BSDF"]
    tc = nodes.new("ShaderNodeTexCoord")
    # aggregate speckles
    vor = nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 90.0
    vor.inputs["Randomness"].default_value = 1.0
    links.new(tc.outputs["Object"], vor.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.56, 0.57, 0.58, 1)
    ramp.color_ramp.elements[1].position = 0.35
    ramp.color_ramp.elements[1].color = (0.80, 0.81, 0.82, 1)
    links.new(vor.outputs["Color"], ramp.inputs["Fac"])
    # large soft clouding
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.08
    noise.inputs["Detail"].default_value = 3
    links.new(tc.outputs["Object"], noise.inputs["Vector"])
    # tile joints (1.2 m tiles)
    brick = nodes.new("ShaderNodeTexBrick")
    brick.inputs["Scale"].default_value = 1.0 / 1.2
    brick.inputs["Mortar Size"].default_value = 0.004
    brick.offset = 0.0
    brick.inputs["Brick Width"].default_value = 1.0
    brick.inputs["Row Height"].default_value = 1.0
    brick.inputs["Color1"].default_value = (1, 1, 1, 1)
    brick.inputs["Color2"].default_value = (1, 1, 1, 1)
    brick.inputs["Mortar"].default_value = (0.82, 0.82, 0.82, 1)
    links.new(tc.outputs["Object"], brick.inputs["Vector"])
    mul = nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    links.new(ramp.outputs["Color"], mul.inputs["A"])
    links.new(brick.outputs["Color"], mul.inputs["B"])
    cloud = nodes.new("ShaderNodeMix")
    cloud.data_type = "RGBA"
    cloud.blend_type = "OVERLAY"
    cloud.inputs["Factor"].default_value = 0.12
    links.new(mul.outputs["Result"], cloud.inputs["A"])
    links.new(noise.outputs["Color"], cloud.inputs["B"])
    links.new(cloud.outputs["Result"], b.inputs["Base Color"])
    rr = nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.10
    rr.inputs["To Max"].default_value = 0.30
    links.new(noise.outputs["Fac"], rr.inputs["Value"])
    links.new(rr.outputs["Result"], b.inputs["Roughness"])
    return m


def brushed_steel():
    m, nodes, links = new_mat("steel")
    b = nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.78, 0.79, 0.80, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.28
    b.inputs["Anisotropic"].default_value = 0.6
    return m


def banner_wash():
    """Abstract landscape wash: pale sky, a soft coral glow, two layered blue/teal ridges. No text."""
    m, nodes, links = new_mat("banner")
    nodes.remove(nodes["Principled BSDF"])
    out = nodes["Material Output"]
    tc = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tc.outputs["UV"], sep.inputs[0])
    U, V = sep.outputs["X"], sep.outputs["Y"]

    def math(op, a, b=None, c=None, clamp=False):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    def ridge_noise(scale, seed):
        cxyz = nodes.new("ShaderNodeCombineXYZ")
        links.new(U, cxyz.inputs["X"])
        cxyz.inputs["Y"].default_value = seed
        nz = nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = scale
        nz.inputs["Detail"].default_value = 5.0
        nz.inputs["Roughness"].default_value = 0.5
        links.new(cxyz.outputs[0], nz.inputs["Vector"])
        return nz.outputs["Fac"]

    def mask_below(height, softness):
        # 1 below the ridge line, 0 above, soft edge
        d = math("SUBTRACT", height, V)
        return math("MULTIPLY", d, 1.0 / softness, clamp=True)

    def mix(fac, a, b):
        n = nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        links.new(fac, n.inputs["Factor"])
        for sock, v in (("A", a), ("B", b)):
            if isinstance(v, tuple):
                n.inputs[sock].default_value = (*v, 1)
            else:
                links.new(v, n.inputs[sock])
        return n.outputs["Result"]

    # sky: white at top -> pale aqua at the horizon
    sky_f = math("MULTIPLY", math("SUBTRACT", 1.0, V), 1.6, clamp=True)
    col = mix(math("POWER", sky_f, 2.0), (0.98, 0.99, 1.0), (0.70, 0.88, 0.93))
    # coral glow hugging the far ridge
    far_h = math("MULTIPLY_ADD", ridge_noise(2.6, 3.1), 0.5, 0.12)
    glow = math("MULTIPLY", math("SUBTRACT", 1.0,
                                 math("ABSOLUTE", math("MULTIPLY", math("SUBTRACT", V, far_h), 9.0)),
                                 clamp=True), 0.75)
    col = mix(glow, col, (1.0, 0.45, 0.36))
    # far ridge (soft blue) and near ridge (deep teal) with a lighter crest
    col = mix(mask_below(far_h, 0.03), col, (0.40, 0.66, 0.90))
    near_h = math("MULTIPLY_ADD", ridge_noise(1.6, 7.7), 0.55, -0.06)
    near_m = mask_below(near_h, 0.05)
    crest = mix(math("MULTIPLY", math("SUBTRACT", near_h, V), 6.0, clamp=True),
                (0.36, 0.78, 0.80), (0.05, 0.42, 0.52))
    col = mix(near_m, col, crest)
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 3.0
    links.new(col, em.inputs["Color"])
    links.new(em.outputs[0], out.inputs["Surface"])
    return m


def ceiling_white():
    """Lattice fins: satin white with a whisper of self-glow (stands in for bounce from the bright floor)."""
    return principled("lattice_white", (0.86, 0.87, 0.88), rough=0.35,
                      emit=(0.92, 0.95, 1.0), emit_strength=0.1)


MAT = {
    "floor": terrazzo(),
    "white": principled("white_panel", (0.83, 0.84, 0.85), rough=0.3, coat=0.2),
    "column": principled("column_white", (0.86, 0.87, 0.88), rough=0.22, coat=0.4),
    "lattice": ceiling_white(),
    "void": principled("void_dark", (0.012, 0.013, 0.016), rough=0.9),
    "steel": brushed_steel(),
    "dark_steel": principled("dark_steel", (0.08, 0.085, 0.09), rough=0.35, metal=1.0),
    "rubber": principled("handrail_black", (0.015, 0.015, 0.015), rough=0.4),
    "glass": thin_glass("glass"),
    "glass_clear": thin_glass("glass_clear", tint=(0.92, 0.96, 0.97), alpha=0.9),
    "downlight": emission("downlight", (1.0, 0.97, 0.93), 60.0),
    "panel_light": emission("panel_light", (0.95, 0.97, 1.0), 14.0),
    "shop_glow": emission("shop_glow", (0.93, 0.95, 0.97), 2.2),
    "train_white": principled("train_body", (0.80, 0.81, 0.82), rough=0.2, coat=0.6),
    "train_window": principled("train_window", (0.02, 0.025, 0.03), rough=0.05, coat=1.0),
    "train_interior": emission("train_interior", (0.85, 0.9, 0.95), 1.5),
    "sign_dark": principled("sign_dark", (0.03, 0.035, 0.04), rough=0.3),
    "banner": banner_wash(),
    "stone_step": principled("stair_stone", (0.74, 0.75, 0.76), rough=0.35),
    "esc_step": principled("esc_step", (0.12, 0.12, 0.13), rough=0.4, metal=0.8),
}


# ----------------------------------------------------------------------------
# Mesh building helpers
# ----------------------------------------------------------------------------
class MeshBuilder:
    """Accumulates box/prism geometry into one mesh per material (keeps object count low)."""

    def __init__(self):
        self.verts, self.faces = [], []

    def add(self, verts, faces):
        o = len(self.verts)
        self.verts.extend(verts)
        self.faces.extend([tuple(i + o for i in f) for f in faces])

    def box(self, cmin, cmax):
        (x0, y0, z0), (x1, y1, z1) = cmin, cmax
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        self.add(v, f)

    def box_c(self, c, size):
        h = [s / 2 for s in size]
        self.box((c[0] - h[0], c[1] - h[1], c[2] - h[2]), (c[0] + h[0], c[1] + h[1], c[2] + h[2]))

    def beam(self, p0, p1, w, d, up=Vector((0, 0, 1))):
        """Rectangular bar from p0 to p1 (centreline), width w, depth d."""
        p0, p1 = Vector(p0), Vector(p1)
        ax = (p1 - p0).normalized()
        side = ax.cross(up)
        if side.length < 1e-6:
            side = Vector((1, 0, 0))
        side.normalize()
        nrm = side.cross(ax).normalized()
        sw, sd = side * (w / 2), nrm * (d / 2)
        v = []
        for p in (p0, p1):
            v += [tuple(p - sw - sd), tuple(p + sw - sd), tuple(p + sw + sd), tuple(p - sw + sd)]
        f = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        self.add(v, f)

    def quad(self, a, b, c, d):
        self.add([a, b, c, d], [(0, 1, 2, 3)])

    def build(self, name, mat, bevel=0.0, smooth=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.verts, [], self.faces)
        me.validate()
        me.materials.append(mat)
        if smooth:
            me.shade_smooth()
        ob = bpy.data.objects.new(name, me)
        COLL.objects.link(ob)
        if bevel > 0:
            mod = ob.modifiers.new("bevel", "BEVEL")
            mod.width = bevel
            mod.segments = 2
            mod.limit_method = "ANGLE"
            mod.angle_limit = math.radians(35)
            mod.harden_normals = False
        return ob


def make_object(name, bm, mat, smooth=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    COLL.objects.link(ob)
    return ob


# ----------------------------------------------------------------------------
# Lattice ceiling: three families of deep white fins forming a triangular diagrid
# ----------------------------------------------------------------------------
def clip_line(p, d, xmin, xmax, ymin, ymax):
    """Clip infinite line p + t d to a rectangle; returns (t0, t1) or None."""
    t0, t1 = -1e9, 1e9
    for pc, dc, lo, hi in ((p[0], d[0], xmin, xmax), (p[1], d[1], ymin, ymax)):
        if abs(dc) < 1e-9:
            if pc < lo or pc > hi:
                return None
            continue
        a, b = (lo - pc) / dc, (hi - pc) / dc
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return (t0, t1) if t1 > t0 else None


def build_lattice():
    fins = MeshBuilder()
    W, S = 4.0, 1.75                       # triangle base (across) and row pitch (along)
    FIN_W, FIN_D = 0.05, 0.48
    xmin, xmax, ymin, ymax = X_MIN, X_MAX, Y_MIN, Y_MAX
    lines = []
    # transverse fins
    k = math.floor(ymin / S)
    while k * S <= ymax:
        lines.append(((0.0, k * S), (1.0, 0.0)))
        k += 1
    # two diagonal families through the transverse-line nodes
    slope = Vector((W / 2, S)).normalized()
    for sgn in (1, -1):
        d = (sgn * slope.x, slope.y)
        span = (ymax - ymin) / S * W / 2 + (xmax - xmin)
        n = int(span / W) + 4
        for i in range(-n, n + 1):
            lines.append(((i * W, 0.0), d))

    def station(x, y, tx, ty):
        nx, ny = -ty, tx
        zb = ceil_z(x, y)
        hw = FIN_W / 2
        return [(x - nx * hw, y - ny * hw, zb), (x + nx * hw, y + ny * hw, zb),
                (x + nx * hw, y + ny * hw, zb + FIN_D), (x - nx * hw, y - ny * hw, zb + FIN_D)]

    for p, d in lines:
        c = clip_line(p, d, xmin, xmax, ymin, ymax)
        if not c:
            continue
        t0, t1 = c
        n = max(1, int((t1 - t0) / 1.6))
        verts, faces = [], []
        for j in range(n + 1):
            t = t0 + (t1 - t0) * j / n
            verts += station(p[0] + d[0] * t, p[1] + d[1] * t, d[0], d[1])
        for j in range(n):
            a, b = 4 * j, 4 * (j + 1)
            for e in range(4):
                e2 = (e + 1) % 4
                faces.append((a + e, a + e2, b + e2, b + e))
        faces.append((3, 2, 1, 0))
        faces.append((4 * n, 4 * n + 1, 4 * n + 2, 4 * n + 3))
        fins.add(verts, faces)
    lat = fins.build("lattice", MAT["lattice"], bevel=0.018)
    for poly in lat.data.polygons:
        poly.use_smooth = False

    # dark void above the lattice, peppered with small downlights
    void = MeshBuilder()
    vz = CEIL_BASE + 2.4
    void.quad((xmin, ymin, vz), (xmin, ymax, vz), (xmax, ymax, vz), (xmax, ymin, vz))
    void.build("void", MAT["void"])

    dl = bmesh.new()
    for _ in range(900):
        x, y = RNG.uniform(xmin + 1, xmax - 1), RNG.uniform(ymin, ymax)
        z = vz - 0.03
        bmesh.ops.create_circle(dl, cap_ends=True, segments=10, radius=0.17,
                                matrix=Matrix.Translation((x, y, z)))
    ob = make_object("downlights", dl, MAT["downlight"], smooth=False)
    ob.visible_glossy = True

    # occasional bright rectangular fixtures sitting just inside the lattice
    pan = MeshBuilder()
    for _ in range(46):
        x, y = RNG.uniform(xmin + 3, xmax - 3), RNG.uniform(ymin + 2, ymax - 10)
        y = round(y / S) * S + S / 2
        z = ceil_z(x, y) + FIN_D * 0.6
        pan.box_c((x, y, z), (2.4, 0.5, 0.05))
    pan.build("light_panels", MAT["panel_light"])


# ----------------------------------------------------------------------------
# Shell: floor, walls, mezzanines, columns
# ----------------------------------------------------------------------------
def balustrade(builder_glass, builder_steel, p0, p1, h=1.15):
    """Frameless glass balustrade with a steel cap rail along p0->p1 at floor height p0.z."""
    p0, p1 = Vector(p0), Vector(p1)
    L = (p1 - p0).length
    n = max(1, int(L / 1.6))
    ax = (p1 - p0).normalized()
    side = Vector((-ax.y, ax.x, 0)) * 0.012
    for i in range(n):
        a = p0 + (p1 - p0) * (i / n) + ax * 0.01
        b = p0 + (p1 - p0) * ((i + 1) / n) - ax * 0.01
        builder_glass.quad(tuple(a + Vector((0, 0, 0.05))), tuple(b + Vector((0, 0, 0.05))),
                           tuple(b + Vector((0, 0, h - 0.05))), tuple(a + Vector((0, 0, h - 0.05))))
    builder_steel.beam(p0 + Vector((0, 0, h)), p1 + Vector((0, 0, h)), 0.06, 0.06)
    builder_steel.beam(p0 + Vector((0, 0, 0.04)) + side * 4, p1 + Vector((0, 0, 0.04)) + side * 4, 0.05, 0.08)


def build_shell():
    # main floor
    # main floor
    fl = MeshBuilder()
    fl.quad((X_MIN, Y_MIN - 30, 0), (X_MAX, Y_MIN - 30, 0), (X_MAX, Y_MAX, 0), (X_MIN, Y_MAX, 0))
    floor = fl.build("floor", MAT["floor"])

    walls = MeshBuilder()
    H = CEIL_BASE + 3
    walls.box((X_MIN - 1, Y_MIN - 30, 0), (X_MIN, Y_MAX, H))        # left
    walls.box((X_MAX, Y_MIN - 30, 0), (X_MAX + 1, Y_MAX, H))        # right
    walls.box((X_MIN, Y_MAX, 0), (X_MAX, Y_MAX + 1, H))             # far
    walls.box((X_MIN, Y_MIN - 31, 0), (X_MAX, Y_MIN - 30, H))       # behind camera
    walls.build("walls", MAT["white"])

    slabs, glass, steel, shops, mullions, cols = (MeshBuilder() for _ in range(6))
    T = 0.7   # slab thickness
    # mezzanine slabs (left, right, far)
    mezz = [
        ((X_MIN, Y_MIN - 30), (-19.0, Y_MAX)),
        ((5.0, 34.0), (X_MAX, Y_MAX)),
        ((X_MIN, 118.0), (X_MAX, Y_MAX)),
    ]
    for (x0, y0), (x1, y1) in mezz:
        slabs.box((x0, y0, MEZZ_Z - T), (x1, y1, MEZZ_Z))
    # balustrades on the slab edges facing the void
    balustrade(glass, steel, (-19.1, Y_MIN - 8, MEZZ_Z), (-19.1, 118.0, MEZZ_Z))
    balustrade(glass, steel, (5.1, 34.2, MEZZ_Z), (5.1, 118.0, MEZZ_Z))
    balustrade(glass, steel, (-19.0, 117.9, MEZZ_Z), (5.0, 117.9, MEZZ_Z))
    # escalator/stair opening edge of right mezzanine
    balustrade(glass, steel, (19.0, 34.1, MEZZ_Z), (X_MAX, 34.1, MEZZ_Z))

    # glazed shopfronts / back-of-house on both levels along the walls
    for z0, z1 in ((0.0, MEZZ_Z - T), (MEZZ_Z, MEZZ_Z + 4.2)):
        for y in range(int(Y_MIN), int(Y_MAX), 6):
            if (y // 6) % 3 == 0:
                continue
            shops.quad((X_MIN + 0.02, y + 0.3, z0 + 0.2), (X_MIN + 0.02, y + 5.7, z0 + 0.2),
                       (X_MIN + 0.02, y + 5.7, z1 - 0.6), (X_MIN + 0.02, y + 0.3, z1 - 0.6))
            shops.quad((X_MAX - 0.02, y + 5.7, z0 + 0.2), (X_MAX - 0.02, y + 0.3, z0 + 0.2),
                       (X_MAX - 0.02, y + 0.3, z1 - 0.6), (X_MAX - 0.02, y + 5.7, z1 - 0.6))
        for x in range(int(X_MIN), int(X_MAX), 5):
            if (x // 5) % 2:
                shops.quad((x + 0.3, Y_MAX - 0.02, z0 + 0.2), (x + 4.7, Y_MAX - 0.02, z0 + 0.2),
                           (x + 4.7, Y_MAX - 0.02, z1 - 0.6), (x + 0.3, Y_MAX - 0.02, z1 - 0.6))
    # vertical mullions on shopfronts
    for z0, z1 in ((0.0, MEZZ_Z - T), (MEZZ_Z, MEZZ_Z + 4.2)):
        for y in range(int(Y_MIN), int(Y_MAX), 2):
            mullions.box((X_MIN, y - 0.04, z0), (X_MIN + 0.08, y + 0.04, z1 - 0.4))
            mullions.box((X_MAX - 0.08, y - 0.04, z0), (X_MAX, y + 0.04, z1 - 0.4))
        mullions.box((X_MIN, Y_MIN - 30, z1 - 0.6), (X_MIN + 0.1, Y_MAX, z1 - 0.4))
        mullions.box((X_MAX - 0.1, Y_MIN - 30, z1 - 0.6), (X_MAX, Y_MAX, z1 - 0.4))

    # big square columns running up into the lattice
    col_pos = [(-19.8, y) for y in (6, 30, 54, 78, 102)] + \
              [(2.0, 46.0), (2.0, 80.0), (14.0, 62.0), (14.0, 96.0), (-4.5, 112.0)]
    for x, y in col_pos:
        cols.box((x - 0.8, y - 0.8, 0), (x + 0.8, y + 0.8, ceil_z(x, y) + 1.0))

    slabs.build("mezzanine_slabs", MAT["white"], bevel=0.02)
    glass.build("balustrade_glass", MAT["glass_clear"])
    steel.build("balustrade_steel", MAT["steel"], bevel=0.008)
    shops.build("shopfronts", MAT["shop_glow"])
    mullions.build("mullions", MAT["steel"])
    cols.build("columns", MAT["column"], bevel=0.06)
    return floor


# ----------------------------------------------------------------------------
# Track: train, platform screen doors, glazed barrel-vault canopy
# ----------------------------------------------------------------------------
def build_track():
    y0, y1 = -20.0, 112.0
    R, BASE = 4.6, 4.2
    xl, xr = TRACK_X - R, TRACK_X + R

    walls, glass, frames, signs = (MeshBuilder() for _ in range(4))
    # PSD header band and the closed side wall
    walls.box((xr - 0.15, y0, 2.45), (xr + 0.15, y1, BASE))
    walls.box((xl - 0.3, y0, 0), (xl, y1, BASE))
    # sill below PSDs
    walls.box((xr - 0.15, y0, 0), (xr + 0.15, y1, 0.08))
    # PSD glass + frames (doors every 4 m with fixed panels between)
    y = y0
    while y < y1:
        frames.box((xr - 0.08, y - 0.05, 0), (xr + 0.08, y + 0.05, 2.45))
        glass.quad((xr + 0.02, y + 0.06, 0.08), (xr + 0.02, y + 1.94, 0.08),
                   (xr + 0.02, y + 1.94, 2.4), (xr + 0.02, y + 0.06, 2.4))
        # small dark info strip on the header (no lettering)
        if int(y) % 8 == 0:
            signs.box((xr + 0.15, y + 0.6, 2.75), (xr + 0.18, y + 1.4, 2.95))
        y += 2.0
    frames.box((xr - 0.1, y0, 2.4), (xr + 0.2, y1, 2.5))

    # vault ribs (arches) and longitudinal purlins
    ribs = MeshBuilder()
    NSEG = 14
    arc = [(TRACK_X + R * math.cos(math.pi * i / NSEG), BASE + R * math.sin(math.pi * i / NSEG))
           for i in range(NSEG + 1)]
    y = y0
    while y <= y1:
        for i in range(NSEG):
            (ax, az), (bx, bz) = arc[i], arc[i + 1]
            ribs.beam((ax, y, az), (bx, y, bz), 0.09, 0.14, up=Vector((0, 1, 0)))
        y += 3.0
    for i in range(0, NSEG + 1, 2):
        ax, az = arc[i]
        ribs.beam((ax, y0, az), (ax, y1, az), 0.06, 0.06)
    # eave beams
    ribs.beam((xr, y0, BASE), (xr, y1, BASE), 0.3, 0.3)
    ribs.beam((xl, y0, BASE), (xl, y1, BASE), 0.3, 0.3)

    # glass skin
    vg = MeshBuilder()
    for i in range(NSEG):
        (ax, az), (bx, bz) = arc[i], arc[i + 1]
        vg.quad((ax, y0, az), (ax, y1, az), (bx, y1, bz), (bx, y0, bz))
    # glazed gable at the near end
    gv = [(TRACK_X, y0, BASE)] + [(x, y0, z) for x, z in arc]
    vg.add(gv, [tuple([0] + list(range(1, NSEG + 2)))])

    walls.build("track_walls", MAT["white"], bevel=0.015)
    glass.build("psd_glass", MAT["glass"])
    frames.build("psd_frames", MAT["steel"], bevel=0.01)
    signs.build("psd_signs", MAT["sign_dark"])
    ribs.build("vault_ribs", MAT["steel"])
    vg.build("vault_glass", MAT["glass"])

    # track bed (dark) and the train
    bed = MeshBuilder()
    bed.box((xl, y0, -1.2), (xr - 0.15, y1, -0.01))
    bed.build("track_bed", MAT["dark_steel"])
    body, win, inter = MeshBuilder(), MeshBuilder(), MeshBuilder()
    cx = TRACK_X + 0.8
    car_len, gap = 19.0, 0.6
    y = y0 + 2
    while y + car_len < y1:
        body.box((cx - 1.45, y, 0.05), (cx + 1.45, y + car_len, 3.65))
        inter.box((cx - 1.3, y + 0.3, 0.9), (cx + 1.3, y + car_len - 0.3, 3.2))
        # windows and doors facing the platform (+x side)
        wy = y + 1.2
        while wy < y + car_len - 1.5:
            win.quad((cx + 1.46, wy, 1.0), (cx + 1.46, wy + 1.6, 1.0),
                     (cx + 1.46, wy + 1.6, 2.5), (cx + 1.46, wy, 2.5))
            wy += 2.4
        y += car_len + gap
    body.build("train_body", MAT["train_white"], bevel=0.12)
    win.build("train_windows", MAT["train_window"])


# ----------------------------------------------------------------------------
# Vertical circulation: twin escalators + wide stair on the right
# ----------------------------------------------------------------------------
def build_circulation():
    Y_FOOT, ANG = 21.0, math.radians(30)
    run = MEZZ_Z / math.tan(ANG)
    Y_TOP = Y_FOOT + run
    trusses, steps, glass, rails, steel = (MeshBuilder() for _ in range(5))
    for xc in (7.0, 9.6):
        w = 1.0
        # sloped truss (side cladding)
        for side in (-1, 1):
            xs = xc + side * (w / 2 + 0.12)
            trusses.add([(xs - 0.1, Y_FOOT - 2.5, 0), (xs + 0.1, Y_FOOT - 2.5, 0),
                         (xs + 0.1, Y_TOP + 2.5, MEZZ_Z), (xs - 0.1, Y_TOP + 2.5, MEZZ_Z),
                         (xs - 0.1, Y_FOOT - 2.5, 1.0), (xs + 0.1, Y_FOOT - 2.5, 1.0),
                         (xs + 0.1, Y_TOP + 2.5, MEZZ_Z + 1.0), (xs - 0.1, Y_TOP + 2.5, MEZZ_Z + 1.0)],
                        [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
            # glass balustrade + black handrail
            glass.quad((xs, Y_FOOT - 1.5, 1.0), (xs, Y_TOP + 1.5, MEZZ_Z + 1.0),
                       (xs, Y_TOP + 1.5, MEZZ_Z + 1.9), (xs, Y_FOOT - 1.5, 1.9))
            rails.beam((xs, Y_FOOT - 1.8, 1.0), (xs, Y_FOOT - 1.5, 1.95), 0.08, 0.04)
            rails.beam((xs, Y_FOOT - 1.5, 1.95), (xs, Y_TOP + 1.5, MEZZ_Z + 0.95 + 1.0), 0.08, 0.04)
        # underside soffit
        trusses.add([(xc - w / 2 - 0.2, Y_FOOT, 0.0), (xc + w / 2 + 0.2, Y_FOOT, 0.0),
                     (xc + w / 2 + 0.2, Y_TOP, MEZZ_Z - 0.9), (xc - w / 2 - 0.2, Y_TOP, MEZZ_Z - 0.9)],
                    [(0, 1, 2, 3)])
        # step band (ribbed)
        n = int(run / 0.4)
        for i in range(n):
            ya = Y_FOOT + run * i / n
            za = MEZZ_Z * i / n
            steps.box((xc - w / 2, ya, za + 0.85), (xc + w / 2, ya + 0.4, za + 0.85 + 0.21))
        steps.box((xc - w / 2, Y_FOOT - 2.3, 0.0), (xc + w / 2, Y_FOOT, 0.86))
        steps.box((xc - w / 2, Y_TOP, MEZZ_Z - 0.1), (xc + w / 2, Y_TOP + 2.3, MEZZ_Z + 0.02))

    # wide stair
    sx0, sx1 = 11.4, 18.8
    rise, tread = 0.1625, 0.3
    n = int(MEZZ_Z / rise)
    stair = MeshBuilder()
    for i in range(n):
        y = Y_FOOT + 0.5 + i * tread
        stair.box((sx0, y, 0), (sx1, y + tread, (i + 1) * rise))
    stair.box((sx0, Y_FOOT + 0.5 + n * tread, 0), (sx1, 34.2, MEZZ_Z))
    # stair handrails: sides and centre
    s_top = Y_FOOT + 0.5 + n * tread
    for x in (sx0 + 0.1, (sx0 + sx1) / 2, sx1 - 0.1):
        steel.beam((x, Y_FOOT - 0.3, 0.95), (x, Y_FOOT + 0.5, 0.95), 0.05, 0.05)
        steel.beam((x, Y_FOOT + 0.5, 0.95), (x, s_top, MEZZ_Z + 0.95), 0.05, 0.05)
        for k in range(0, n, 6):
            yy = Y_FOOT + 0.5 + k * tread + tread / 2
            steel.beam((x, yy, (k + 1) * rise), (x, yy, (k + 1) * rise + 0.95), 0.04, 0.04)
    for x in (sx0 - 0.05, sx1 + 0.05):
        glass.quad((x, Y_FOOT + 0.5, 0.1), (x, s_top, MEZZ_Z + 0.1),
                   (x, s_top, MEZZ_Z + 1.0), (x, Y_FOOT + 0.5, 1.0))

    # pole-mounted wayfinding boxes near the escalator foot (blank faces, coloured end tabs)
    signs, tabs = MeshBuilder(), MeshBuilder()
    for (x, y) in ((3.2, 15.5), (-2.5, 30.0)):
        steel.box((x - 0.06, y - 0.06, 0), (x + 0.06, y + 0.06, 3.1))
        signs.box((x - 1.6, y - 0.12, 2.75), (x + 1.6, y + 0.12, 3.15))
        tabs.box((x - 1.55, y - 0.13, 2.82), (x - 1.25, y + 0.13, 3.08))
        tabs.box((x + 1.25, y - 0.13, 2.82), (x + 1.55, y + 0.13, 3.08))
    signs.build("signs", MAT["sign_dark"], bevel=0.01)
    tabs.build("sign_tabs", emission("sign_tab", (0.25, 0.85, 0.55), 3.0))

    trusses.build("escalator_truss", MAT["steel"], bevel=0.01)
    steps.build("escalator_steps", MAT["esc_step"])
    glass.build("escalator_glass", MAT["glass_clear"])
    rails.build("escalator_handrails", MAT["rubber"])
    stair.build("stair", MAT["stone_step"], bevel=0.008)
    steel.build("stair_rails", MAT["steel"])


# ----------------------------------------------------------------------------
# Banner lightbox (abstract wash, no lettering)
# ----------------------------------------------------------------------------
def build_banner():
    cx, cy = -5.0, 26.0
    w, h, z0 = 12.5, 7.4, 7.6
    frame = MeshBuilder()
    frame.box((cx - w / 2 - 0.35, cy, z0 - 0.3), (cx + w / 2 + 0.35, cy + 1.0, z0 + h + 0.3))
    # suspension rods into the lattice
    for dx in (-w / 2 + 1, w / 2 - 1):
        frame.box((cx + dx - 0.04, cy + 0.46, z0 + h), (cx + dx + 0.04, cy + 0.54, ceil_z(cx + dx, cy) + 0.3))
    frame.build("banner_frame", MAT["white"], bevel=0.04)
    me = bpy.data.meshes.new("banner_face")
    v = [(cx - w / 2, cy - 0.01, z0), (cx + w / 2, cy - 0.01, z0),
         (cx + w / 2, cy - 0.01, z0 + h), (cx - w / 2, cy - 0.01, z0 + h)]
    me.from_pydata(v, [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for li, (u, vv) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[li].uv = (u, vv)
    me.materials.append(MAT["banner"])
    ob = bpy.data.objects.new("banner_face", me)
    COLL.objects.link(ob)


# ----------------------------------------------------------------------------
# People: stylised low-poly figures, moving (motion blur)
# ----------------------------------------------------------------------------
CLOTHES = [(0.05, 0.05, 0.06), (0.12, 0.13, 0.15), (0.55, 0.56, 0.58), (0.80, 0.80, 0.78),
           (0.18, 0.22, 0.30), (0.42, 0.36, 0.30), (0.62, 0.66, 0.70), (0.30, 0.34, 0.32)]
_cloth_mats = [principled(f"cloth_{i}", c, rough=0.7) for i, c in enumerate(CLOTHES)]
_skin = principled("skin", (0.55, 0.40, 0.32), rough=0.5)
_hair = principled("hair", (0.03, 0.025, 0.02), rough=0.6)


def ellipsoid(bm, center, radii, seg=10):
    m = Matrix.Translation(center) @ Matrix.Diagonal((*radii, 1.0))
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2 + 2, radius=1.0, matrix=m)


def person(name, loc, heading, speed):
    s = RNG.uniform(0.92, 1.06)
    parts = []
    top_m = RNG.choice(_cloth_mats)
    leg_m = RNG.choice(_cloth_mats[:2] + _cloth_mats[4:5])
    for mat, build in (
        (leg_m, lambda bm: [ellipsoid(bm, (0, sx * 0.09 * s, 0.45 * s), (0.08 * s, 0.075 * s, 0.46 * s))
                            for sx in (-1, 1)]),
        (top_m, lambda bm: ellipsoid(bm, (0, 0, 1.18 * s), (0.15 * s, 0.24 * s, 0.33 * s))),
        (_skin, lambda bm: ellipsoid(bm, (0, 0, 1.6 * s), (0.10 * s, 0.09 * s, 0.12 * s))),
    ):
        bm = bmesh.new()
        build(bm)
        parts.append((mat, bm))
    me = bpy.data.meshes.new(name)
    bm_all = bmesh.new()
    for mi, (mat, bm) in enumerate(parts):
        me.materials.append(mat)
        tmp = bpy.data.meshes.new("tmp")
        bm.to_mesh(tmp)
        bm.free()
        for p in tmp.polygons:
            p.material_index = mi
        bm_all.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    bm_all.to_mesh(me)
    bm_all.free()
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    COLL.objects.link(ob)
    ob.rotation_euler = (0, 0, heading)
    # walking direction is local +X; key two frames so Cycles blurs the motion
    d = Vector((math.cos(heading), math.sin(heading), 0)) * speed
    ob.location = Vector(loc) - d / 2
    ob.keyframe_insert("location", frame=1)
    ob.location = Vector(loc) + d / 2
    ob.keyframe_insert("location", frame=2)
    return ob


def build_people():
    spots = []
    # crowd near escalator feet and along the platform edge
    for _ in range(16):
        spots.append((RNG.uniform(5.0, 12.0), RNG.uniform(14.0, 21.0), 0.0))
    while len(spots) < 48:
        x, y = RNG.uniform(-7.0, 5.0), RNG.uniform(16.0, 75.0)
        spots.append((x, y, 0.0))
    for _ in range(4):
        spots.append((RNG.uniform(0.5, 4.5), RNG.uniform(12.0, 17.0), 0.0))
    # a few on the mezzanines
    for _ in range(6):
        spots.append((RNG.uniform(-25.0, -20.5), RNG.uniform(20.0, 100.0), MEZZ_Z))
    for _ in range(5):
        spots.append((RNG.uniform(6.5, 21.0), RNG.uniform(40.0, 110.0), MEZZ_Z))
    for i, p in enumerate(spots):
        heading = RNG.choice([math.pi / 2, -math.pi / 2, 0.0, math.pi]) + RNG.uniform(-0.5, 0.5)
        speed = RNG.uniform(0.8, 1.5) if RNG.random() > 0.15 else 0.05
        person(f"person_{i:02d}", p, heading, speed)


# ----------------------------------------------------------------------------
# Lighting
# ----------------------------------------------------------------------------
def build_lights():
    # broad soft "daylight" from under the lattice; hidden from camera and reflections
    for x in (-18.0, -6.0, 6.0, 17.0):
        for y in range(-10, 140, 14):
            ld = bpy.data.lights.new(f"fill_{x}_{y}", "AREA")
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = 11.0, 13.0
            ld.energy = 1600.0
            ld.color = (0.95, 0.97, 1.0)
            ld.spread = math.radians(140)
            ob = bpy.data.objects.new(ld.name, ld)
            ob.location = (x, y, ceil_z(x, y) - 0.4)
            ob.visible_camera = False
            ob.visible_glossy = False
            COLL.objects.link(ob)
    # up-lights: stand in for the strong bounce off the pale floor onto the lattice undersides
    for x in (-15.0, 0.0, 14.0):
        for y in range(0, 140, 20):
            ld = bpy.data.lights.new(f"up_{x}_{y}", "AREA")
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = 14.0, 20.0
            ld.energy = 2500.0
            ld.color = (0.95, 0.97, 1.0)
            ob = bpy.data.objects.new(ld.name, ld)
            ob.location = (x, y, 9.5)
            ob.rotation_euler = (math.pi, 0, 0)
            ob.visible_camera = False
            ob.visible_glossy = False
            COLL.objects.link(ob)
    # train interior glow under the vault
    ld = bpy.data.lights.new("vault_light", "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = 3.0, 130.0
    ld.energy = 6000.0
    ld.color = (0.92, 0.96, 1.0)
    ob = bpy.data.objects.new("vault_light", ld)
    ob.location = (TRACK_X, 46.0, 4.0)
    ob.visible_camera = False
    ob.visible_glossy = False
    COLL.objects.link(ob)

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.85, 0.9, 1.0, 1)
    bg.inputs["Strength"].default_value = 0.15
    world.mist_settings.start = 8.0
    world.mist_settings.depth = 150.0
    world.mist_settings.falloff = "LINEAR"
    SCENE.world = world


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def build_camera():
    cd = bpy.data.cameras.new("cam")
    cd.lens = 21.0
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 36.0
    cd.clip_end = 400.0
    cam = bpy.data.objects.new("cam", cd)
    cam.location = (7.0, -4.0, 8.4)
    cam.rotation_euler = (math.radians(88.0), 0.0, math.radians(14.0))
    COLL.objects.link(cam)
    SCENE.camera = cam
    cd.shift_y = 0.0


# ----------------------------------------------------------------------------
# Render settings and compositor (depth haze from the mist pass)
# ----------------------------------------------------------------------------
def setup_render():
    r = SCENE.render
    r.engine = "CYCLES"
    r.resolution_x = int(1440 * OPTS["scale"])
    r.resolution_y = int(2560 * OPTS["scale"])
    r.resolution_percentage = 100
    r.threads_mode = "FIXED"
    r.threads = OPTS["threads"]
    r.use_motion_blur = True
    r.motion_blur_shutter = 1.0
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.film_transparent = False

    c = SCENE.cycles
    c.device = "CPU"
    c.samples = OPTS["samples"]
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.02
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.max_bounces = 6
    c.diffuse_bounces = 3
    c.glossy_bounces = 3
    c.transmission_bounces = 4
    c.transparent_max_bounces = 16
    c.volume_bounces = 0
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 4.0
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.blur_glossy = 1.0

    SCENE.frame_start, SCENE.frame_end = 1, 2
    SCENE.frame_set(1)

    vs = SCENE.view_settings
    vs.view_transform = "AgX"
    vs.look = "AgX - Medium High Contrast"
    vs.exposure = float(os.environ.get("GN_EXPOSURE", -1.6))
    vs.gamma = 1.0
    SCENE.display_settings.display_device = "sRGB"
    SCENE.sequencer_colorspace_settings.name = "sRGB"

    # --- compositor: blend towards a cool bright haze by distance -------------
    vl = SCENE.view_layers[0]
    vl.use_pass_mist = True
    ng = bpy.data.node_groups.new("haze", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = SCENE
    out = ng.nodes.new("NodeGroupOutput")
    fac = ng.nodes.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    fac.inputs[1].default_value = 0.13
    fac.use_clamp = True
    ng.links.new(rl.outputs["Mist"], fac.inputs[0])
    mix = ng.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.inputs["B"].default_value = (2.0, 2.15, 2.3, 1)   # scene-linear, ~pale white after exposure
    ng.links.new(fac.outputs[0], mix.inputs["Factor"])
    ng.links.new(rl.outputs["Image"], mix.inputs["A"])
    ng.links.new(mix.outputs["Result"], out.inputs[0])
    SCENE.compositing_node_group = ng
    SCENE.render.use_compositing = True
    SCENE.render.compositor_device = "CPU"   # default GPU aborts without a GL context


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    build_lattice()
    build_shell()
    build_track()
    build_circulation()
    build_banner()
    build_people()
    build_lights()
    build_camera()
    setup_render()
    out = os.path.abspath(OPTS["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    SCENE.render.filepath = out
    print(f"[gn] scene built in {time.time() - t0:.1f}s; "
          f"{SCENE.render.resolution_x}x{SCENE.render.resolution_y} @ {OPTS['samples']} spp", flush=True)
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[gn] rendered in {time.time() - t1:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
