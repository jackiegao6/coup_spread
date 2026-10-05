"""Draw the Section 3 example as a vector figure; no probabilities are invented.

Run from any directory with Python and matplotlib. The two independent
coupon realizations are exactly those described in paper-v4.tex.
"""
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


BLUE = "#236A9B"
ORANGE = "#B35C21"
INK = "#222222"
POS = {"s1": (.85, 2.05), "s2": (.85, .90),
       "w": (3.65, 1.475), "u": (6.25, 2.05), "v": (6.25, .90)}
LABEL = {"s1": r"$s_1$", "s2": r"$s_2$", "w": r"$w$",
         "u": r"$u$", "v": r"$v$"}
RADIUS = .20


def arrow(ax, src, dst, color, style="solid"):
    a, b = POS[src], POS[dst]
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy)
    ux, uy = dx/length, dy/length
    start = (a[0]+RADIUS*ux, a[1]+RADIUS*uy)
    end = (b[0]-(RADIUS+.025)*ux, b[1]-(RADIUS+.025)*uy)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=11, linewidth=1.8,
                                color=color, linestyle=style,
                                shrinkA=0, shrinkB=0, zorder=1))


def ticket(ax, x, y, number, color):
    from matplotlib.patches import FancyBboxPatch
    # The ticket shape gives the colored path a concrete object to follow.
    ax.add_patch(FancyBboxPatch((x-.36, y-.12), .72, .24,
                 boxstyle="round,pad=0.035,rounding_size=0.025",
                 linewidth=.85, edgecolor=color, facecolor="white", zorder=4))
    ax.plot([x+.19,x+.19], [y-.105,y+.105], color=color, linewidth=.65,
            linestyle=(0,(1,1)), zorder=5)
    ax.text(x-.06,y,"Coupon",ha="center",va="center",fontsize=7.8,color=color,zorder=5)
    ax.text(x+.25,y,str(number),ha="center",va="center",fontsize=9,
            fontweight="bold",color=color,zorder=5)


fig = plt.figure(figsize=(7.05, 2.72), facecolor="white")
ax = fig.add_axes([.015,.02,.97,.96])
ax.set_xlim(0,7.8)
ax.set_ylim(-.40,2.80)
ax.axis("off")
# Display coordinates are almost equally scaled; use equal aspect for circles.
ax.set_aspect("equal", adjustable="box")
for x, title in [(.9,"1. Issue"),(3.65,"2. Forward"),(6.4,"3. Consume")]:
    ax.text(x,2.58,title,ha="center",va="center",fontsize=11,fontweight="bold")
for src,dst in [("s1","w"),("w","u")]:
    arrow(ax,src,dst,BLUE)
for src,dst in [("s2","w"),("w","v")]:
    arrow(ax,src,dst,ORANGE,(0,(4,2)))
for key,(x,y) in POS.items():
    active=key in {"u","v"}
    ax.add_patch(Circle((x,y),RADIUS,facecolor=INK if active else "white",
                        edgecolor=INK,linewidth=1.1,zorder=3))
    ax.text(x,y,LABEL[key],ha="center",va="center",fontsize=13,
            color="white" if active else INK,zorder=4)
for x,y,num,color in [(1.82,2.00,1,BLUE),(1.82,.94,2,ORANGE)]:
    ticket(ax,x,y,num,color)
ax.text(.85,.41,"One coupon\nper seed user",ha="center",va="center",fontsize=9)
ax.text(3.65,.47,"Forwards both coupons",ha="center",va="center",fontsize=9)
ax.text(3.65,.20,"Not activated",ha="center",va="center",fontsize=9,fontweight="bold")
for y in [2.05,.90]:
    ax.text(6.58,y,"Consumes\nand activates",ha="left",va="center",fontsize=9)
ax.text(6.4,.28,"Two distinct consumers",ha="center",va="center",fontsize=9)
ax.plot([.2,7.58],[-.065,-.065],color="#D3D3D3",lw=.65)
ax.text(3.9,-.27,"In this example: 2 coupons consumed, 2 users activated.",
        ha="center",va="center",fontsize=10,fontweight="bold")
fig.savefig(OUT / "coupon-model-example.pdf", metadata={"Title": "Issue, forward, consume: two coupons activate two users"})
fig.savefig(OUT / "coupon-model-example.svg")
fig.savefig(OUT / "coupon-model-example.png", dpi=240)
plt.close(fig)
print("Generated PDF, SVG, and PNG in", OUT)
