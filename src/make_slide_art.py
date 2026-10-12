# ============================================================
# make_slide_art.py
# Purpose : Draw the non-data presentation visuals (title and
#           thank-you backgrounds, illustrations, sign cards).
#           Output: slides/art/*.png  (schematic drawings, not
#           real recordings).
#
# How to run:
#   venv\Scripts\python.exe src\make_slide_art.py
# ============================================================

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

DATA_DIR = "data"
OUT_DIR = os.path.join("slides", "art")
CANVA_DIR = os.path.join(OUT_DIR, "canva")
plt.rcParams["svg.fonttype"] = "path"

NAVY, NAVY2, CYAN, WARM = "#0B1F3A", "#123A63", "#3DD5F3", "#FFD166"
INK, MUTED, BLUE, GREEN = "#1F2937", "#6B7280", "#0072B2", "#009E73"
TEAL, PURPLE = "#2A9D8F", "#6C5B9E"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]
OPEN_HAND = np.array([
    (0, 0),
    (-0.55, 0.30), (-0.85, 0.65), (-1.05, 0.95), (-1.20, 1.20),
    (-0.45, 1.05), (-0.50, 1.50), (-0.52, 1.85), (-0.53, 2.15),
    (-0.10, 1.15), (-0.10, 1.65), (-0.10, 2.05), (-0.10, 2.40),
    (0.25, 1.05), (0.28, 1.50), (0.30, 1.85), (0.31, 2.15),
    (0.55, 0.90), (0.62, 1.25), (0.66, 1.50), (0.68, 1.72),
], dtype=float)
FINGERS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
           "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
SPREAD_DEG = {"thumb": 0, "index": 8, "middle": 0, "ring": -6, "pinky": -14}

LIGHT = dict(pose_line="#64748B", pose_dot="#E69F00", hand_line=BLUE, hand_dot=BLUE, edge="white", glow=False)
DARK = dict(pose_line="#DCE6F2", pose_dot=WARM, hand_line=CYAN, hand_dot=CYAN, edge="white", glow=True)


def save(fig, name, transparent=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    hero_img = name.endswith("hero.png") or name.startswith("background")
    fig.savefig(path, dpi=100 if hero_img else 200, transparent=transparent,
                facecolor="none" if transparent else fig.get_facecolor(),
                bbox_inches=None if hero_img else "tight", pad_inches=0.1)
    if transparent:
        os.makedirs(CANVA_DIR, exist_ok=True)
        stem = os.path.splitext(name)[0]
        fig.savefig(os.path.join(CANVA_DIR, stem + ".png"), dpi=300, transparent=True,
                    bbox_inches="tight", pad_inches=0.1)
        fig.savefig(os.path.join(CANVA_DIR, stem + ".svg"), transparent=True,
                    bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("saved", path)


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def make_hand(curl=0.0, spread=0.0, angle=0.0, scale=1.0, origin=(0.0, 0.0), mirror=False):
    out = OPEN_HAND.copy()
    for name, idx in FINGERS.items():
        c = curl * (0.6 if name == "thumb" else 1.0)
        joints = (20, 40, 40) if name == "thumb" else (70, 90, 60)
        cum = np.radians(SPREAD_DEG[name] * spread)
        pos = OPEN_HAND[idx[0]].copy()
        for k in range(3):
            v = OPEN_HAND[idx[k + 1]] - OPEN_HAND[idx[k]]
            cum += np.radians(joints[k]) * c
            pos = pos + rot(cum) @ v
            out[idx[k + 1]] = pos
    if mirror:
        out[:, 0] *= -1
    return (rot(np.radians(angle)) @ out.T).T * scale + np.array(origin)


def draw_hand(ax, pts, line, dot, lw=2.4, ms=46, edge="white", glow=False, z=3):
    if glow:
        for w, a in [(16, 0.05), (10, 0.09), (5, 0.16)]:
            for i, j in HAND_CONNECTIONS:
                ax.plot(pts[[i, j], 0], pts[[i, j], 1], color=line, lw=w, alpha=a,
                        solid_capstyle="round", zorder=z - 1)
        ax.scatter(pts[:, 0], pts[:, 1], s=ms * 7, color=dot, alpha=0.12, zorder=z - 1)
    for i, j in HAND_CONNECTIONS:
        ax.plot(pts[[i, j], 0], pts[[i, j], 1], color=line, lw=lw, solid_capstyle="round", zorder=z)
    ax.scatter(pts[:, 0], pts[:, 1], s=ms, color=dot, edgecolors=edge, linewidths=1.0, zorder=z + 1)


BODY = dict(nose=(5.0, 8.5), l_sh=(3.7, 6.7), r_sh=(6.3, 6.7), l_el=(2.3, 4.9), r_el=(7.7, 4.9),
            l_wr=(2.9, 6.9), r_wr=(7.1, 6.9), l_hip=(4.2, 2.2), r_hip=(5.8, 2.2))


def draw_signer(ax, origin, s, style, raised_curl=0.0, low_curl=0.15, lw=2.4, ms=46):
    o = np.array(origin, dtype=float)
    P = {k: o + np.array(v) * s for k, v in BODY.items()}
    segs = [("l_sh", "r_sh"), ("l_sh", "l_hip"), ("r_sh", "r_hip"), ("l_hip", "r_hip"),
            ("l_sh", "l_el"), ("l_el", "l_wr"), ("r_sh", "r_el"), ("r_el", "r_wr")]
    if style["glow"]:
        for w, a in [(14, 0.05), (8, 0.09)]:
            for a_, b_ in segs:
                ax.plot([P[a_][0], P[b_][0]], [P[a_][1], P[b_][1]], color=style["pose_line"], lw=w,
                        alpha=a, solid_capstyle="round", zorder=1)
    for a_, b_ in segs:
        ax.plot([P[a_][0], P[b_][0]], [P[a_][1], P[b_][1]], color=style["pose_line"], lw=lw,
                solid_capstyle="round", zorder=2)
    neck_top = o + np.array((5.0, 7.7)) * s
    mid_sh = (P["l_sh"] + P["r_sh"]) / 2
    ax.plot([neck_top[0], mid_sh[0]], [neck_top[1], mid_sh[1]], color=style["pose_line"], lw=lw, zorder=2)
    head = Circle(o + np.array((5.0, 8.5)) * s, 0.78 * s, fill=False, ec=style["pose_line"], lw=lw, zorder=2)
    ax.add_patch(head)
    joints = np.array([P[k] for k in ["nose", "l_sh", "r_sh", "l_el", "r_el", "l_wr", "r_wr", "l_hip", "r_hip"]])
    ax.scatter(joints[:, 0], joints[:, 1], s=ms * 1.7, color=style["pose_dot"], edgecolors=style["edge"],
               linewidths=1.0, zorder=4)
    right_side = make_hand(curl=raised_curl, spread=0.6, angle=-7, scale=0.95 * s, origin=P["r_wr"])
    left_side = make_hand(curl=low_curl, spread=0.6, angle=7, scale=0.95 * s, origin=P["l_wr"], mirror=True)
    for h in (right_side, left_side):
        draw_hand(ax, h, style["hand_line"], style["hand_dot"], lw=lw * 0.75, ms=ms * 0.34,
                  edge=style["edge"], glow=style["glow"])


def gradient_background(ax, w, h, glow_center):
    W, H = 384, 216
    X, Y = np.meshgrid(np.linspace(0, w, W), np.linspace(0, h, H))
    t = (X / w)[..., None]
    a, b = np.array(to_rgb(NAVY)), np.array(to_rgb(NAVY2))
    img = a * (1 - t) + b * t
    g = np.exp(-(((X - glow_center[0]) ** 2) / (2 * 3.0 ** 2) + ((Y - glow_center[1]) ** 2) / (2 * 3.6 ** 2)))[..., None]
    img = np.clip(img + g * 0.24 * np.array(to_rgb(CYAN)), 0, 1)
    ax.imshow(img, extent=[0, w, 0, h], origin="lower", aspect="auto", interpolation="bilinear", zorder=0)


def hero(kind):
    w, h = 19.2, 10.8
    fig = plt.figure(figsize=(w, h), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    gradient_background(ax, w, h, glow_center=(15.0, 5.5))
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0.4, 10.5, 70), rng.uniform(0.4, 10.4, 70)])
    ax.scatter(pts[:, 0], pts[:, 1], s=rng.uniform(4, 22, 70), color=CYAN, alpha=0.13, zorder=1)
    for i in range(90):
        x = 1.2 + i * (16.8 / 89)
        tall = 0.36 if i % 15 == 0 else 0.18
        ax.plot([x, x], [0.55, 0.55 + tall], color="white", lw=1.6, alpha=0.12 + 0.30 * i / 89, zorder=1)
    if kind == "plain":
        for coll in list(ax.collections) + list(ax.lines):
            coll.remove()
        save(fig, "background_dark.png")
        return
    if kind == "title":
        draw_signer(ax, (11.2, 1.3), 0.80, DARK, raised_curl=0.0, lw=3.4, ms=95)
    else:
        big = make_hand(curl=0.0, spread=0.9, angle=10, scale=2.7, origin=(15.0, 2.0))
        draw_hand(ax, big, CYAN, CYAN, lw=5.5, ms=210, glow=True)
    save(fig, "title_hero.png" if kind == "title" else "thankyou_hero.png")


def hand_landmarks_light():
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    ax.set_xlim(-3.0, 2.2); ax.set_ylim(-1.1, 3.2); ax.axis("off"); ax.set_aspect("equal")
    pts = make_hand()
    draw_hand(ax, pts, BLUE, BLUE, lw=3.2, ms=110)
    labels = {0: ("0  wrist", (0.0, -0.45), "center"), 4: ("4  thumb tip", (-1.35, 1.45), "right"),
              8: ("8  index tip", (-0.95, 2.75), "right"), 12: ("12  middle tip", (-0.1, 3.0), "center"),
              16: ("16  ring tip", (0.85, 2.7), "left"), 20: ("20  pinky tip", (1.0, 1.7), "left")}
    for i, (txt, xy, ha) in labels.items():
        ax.annotate(txt, xy=pts[i], xytext=xy, ha=ha, va="center", fontsize=13, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.1))
    ax.text(0.0, -0.95, "21 points per hand", ha="center", fontsize=13, color=MUTED, style="italic")
    save(fig, "hand_landmarks_light.png", transparent=True)


def callout(ax, xy_text, xy_point, text, ha):
    ax.annotate(text, xy=xy_point, xytext=xy_text, ha=ha, va="center", fontsize=15, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#F3F6FA", ec="#9CA3AF", lw=1.3),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.5))


def pose_hands_diagram():
    fig, ax = plt.subplots(figsize=(12, 7.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis("off")
    o, s = np.array((3.0, 0.95)), 0.62
    draw_signer(ax, o, s, LIGHT, lw=3, ms=90)
    pose = lambda k: o + np.array(BODY[k]) * s
    hand_up = np.array([0.0, 1.1 * 0.95 * s])
    callout(ax, (0.2, 6.3), pose("l_wr") + hand_up, "Right hand\n21 points x 3 values\n= 63 numbers", "left")
    callout(ax, (11.8, 6.3), pose("r_wr") + hand_up, "Left hand\n21 points x 3 values\n= 63 numbers", "right")
    callout(ax, (0.2, 2.2), pose("l_hip"), "Body pose\n33 points x 4 values\n= 132 numbers", "left")
    ax.text(6.0, 0.35, "132 + 63 + 63 = 258 numbers per frame", ha="center", fontsize=20, fontweight="bold", color=INK)
    ax.text(11.8, 0.05, "schematic drawing", ha="right", fontsize=10, color=MUTED, style="italic")
    save(fig, "pose_hands_diagram.png", transparent=True)


def frames_strip():
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.2); ax.axis("off")
    frames = [1, 18, 36, 54, 72, 90]
    cw, gap = 2.0, 0.36
    for i, f in enumerate(frames):
        t = (f - 1) / 89
        x0 = 0.15 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x0, 1.1), cw, 2.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#F3F6FA", ec="#9CA3AF", lw=1.4))
        pts = make_hand(curl=0.05 + 0.30 * np.sin(np.pi * t), spread=0.5, angle=-18 + 36 * t,
                        scale=0.6, origin=(x0 + cw / 2 + 0.25 * (t - 0.5), 1.62 + 0.15 * np.sin(np.pi * t)))
        draw_hand(ax, pts, BLUE, BLUE, lw=2.4, ms=30)
        ax.text(x0 + cw / 2, 0.82, f"frame {f}", ha="center", fontsize=13, color=INK)
        if i < len(frames) - 1:
            ax.text(x0 + cw + gap / 2, 2.45, "...", ha="center", fontsize=16, color=MUTED)
    ax.add_patch(FancyArrowPatch((0.25, 0.35), (13.75, 0.35), arrowstyle="-|>", mutation_scale=20, lw=2, color=MUTED))
    ax.text(7.0, 0.05, "one clip = 90 frames (about 3 seconds), each frame = 258 numbers   |   schematic",
            ha="center", va="top", fontsize=13, color=MUTED)
    ax.set_ylim(-0.3, 4.2)
    save(fig, "frames_strip.png", transparent=True)


def sign_cards(summary=True, name="sign_cards.png"):
    labels = list(np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True))
    X, y = np.load(os.path.join(DATA_DIR, "X.npy")), np.load(os.path.join(DATA_DIR, "y.npy"))
    one_handed = {labels[i] for i in range(len(labels))
                  if (np.abs(X[y == i][:, :, 195:258]).sum(2) > 0).mean() < 0.05}
    order = ["how_are_you", "again", "correct", "incorrect", "dont_understand", "understand",
             "deaf_person", "hearing_person", "delicious", "sorry", "thank_you"]
    pretty = lambda w: w.replace("dont_understand", "don't understand").replace("_", " ")
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    tw, th = 3.15, 1.95
    for n in range(12):
        c, r = n % 4, n // 4
        x0, y0 = 0.25 + c * (tw + 0.25), 7.3 - (r + 1) * (th + 0.25) + 0.25
        if n == 11:
            if summary:
                ax.add_patch(FancyBboxPatch((x0, y0), tw, th, boxstyle="round,pad=0.02,rounding_size=0.15", fc=NAVY, ec=NAVY))
                ax.text(x0 + tw / 2, y0 + th / 2, "11 signs\n30 clips each\n330 clips", ha="center", va="center",
                        fontsize=17, fontweight="bold", color="white", linespacing=1.4)
            continue
        w = order[n]
        col = TEAL if w in one_handed else PURPLE
        ax.add_patch(FancyBboxPatch((x0, y0), tw, th, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#F8FAFC", ec=col, lw=2.6))
        ax.text(x0 + 0.2, y0 + th - 0.42, f"{n + 1}", fontsize=13, fontweight="bold", color=col)
        ax.text(x0 + 0.2, y0 + 1.12, pretty(w), fontsize=15, fontweight="bold", color=INK, va="center")
        tag = "one hand" if w in one_handed else "two hands"
        ax.text(x0 + 0.2, y0 + 0.4, tag, fontsize=12.5, color=col, va="center")
        for k in range(1 if w in one_handed else 2):
            hp = make_hand(curl=0.0, spread=0.4, angle=0, scale=0.2, origin=(x0 + tw - 0.5 - 0.5 * k, y0 + 0.2))
            draw_hand(ax, hp, col, col, lw=1.4, ms=9)
    save(fig, name, transparent=True)


def problem_illustration():
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.4); ax.axis("off")
    draw_signer(ax, (0.3, 0.95), 0.40, LIGHT, lw=2.4, ms=48)
    ax.add_patch(FancyBboxPatch((5.35, 1.95), 3.3, 2.45, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#EEF2F6", ec="#9CA3AF", lw=2))
    ax.add_patch(FancyBboxPatch((4.9, 1.55), 4.2, 0.4, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#D1D5DB", ec="#9CA3AF", lw=1.5))
    ax.add_patch(Circle((7.0, 4.22), 0.07, color=INK, zorder=5))
    draw_signer(ax, (6.0, 2.05), 0.20, LIGHT, lw=1.6, ms=14)
    ax.add_patch(Polygon([[10.8, 2.7], [10.45, 1.95], [11.7, 2.7]], closed=True, fc="#DDF2E6", ec=GREEN, lw=2.6, zorder=1))
    ax.add_patch(FancyBboxPatch((10.2, 2.55), 3.4, 1.7, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="#DDF2E6", ec=GREEN, lw=2.6, zorder=2))
    ax.add_patch(Polygon([[10.78, 2.62], [11.30, 2.62], [11.30, 2.48], [10.78, 2.48]], closed=True,
                         fc="#DDF2E6", ec="none", zorder=3))
    ax.text(11.9, 3.4, "Thank you", ha="center", va="center", fontsize=26, fontweight="bold", color=INK, zorder=4)
    for (x1, x2) in [(3.9, 4.75), (9.25, 10.05)]:
        ax.add_patch(FancyArrowPatch((x1, 3.1), (x2, 3.1), arrowstyle="-|>", mutation_scale=28, lw=3, color=MUTED))
    for x, txt in [(2.3, "1  A person signs"), (7.0, "2  Webcam + MediaPipe"), (11.9, "3  The word appears")]:
        ax.text(x, 0.45, txt, ha="center", fontsize=17, fontweight="bold", color=INK)
    save(fig, "problem_illustration.png", transparent=True)


def element_signer():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0.9, 9.1); ax.set_ylim(1.7, 9.7); ax.set_aspect("equal"); ax.axis("off")
    draw_signer(ax, (0, 0), 1.0, DARK, lw=3.4, ms=95)
    save(fig, "title_signer.png", transparent=True)


def element_hand():
    fig, ax = plt.subplots(figsize=(6.4, 7.4))
    ax.set_xlim(-4.6, 2.9); ax.set_ylim(-0.6, 7.0); ax.set_aspect("equal"); ax.axis("off")
    big = make_hand(curl=0.0, spread=0.9, angle=10, scale=2.7, origin=(0, 0))
    draw_hand(ax, big, CYAN, CYAN, lw=5.5, ms=210, glow=True)
    save(fig, "thankyou_hand.png", transparent=True)


def element_ticks():
    fig, ax = plt.subplots(figsize=(16.8, 0.7))
    ax.set_xlim(0, 16.8); ax.set_ylim(0, 0.7); ax.axis("off")
    for i in range(90):
        x = 0.1 + i * (16.6 / 89)
        tall = 0.5 if i % 15 == 0 else 0.28
        ax.plot([x, x], [0.1, 0.1 + tall], color="white", lw=2.0, alpha=0.35 + 0.45 * i / 89)
    save(fig, "frame_ticks.png", transparent=True)


def main():
    hero("title")
    hero("thanks")
    hero("plain")
    element_signer()
    element_hand()
    element_ticks()
    hand_landmarks_light()
    pose_hands_diagram()
    frames_strip()
    sign_cards()
    sign_cards(summary=False, name="sign_cards_clean.png")
    problem_illustration()


if __name__ == "__main__":
    main()
