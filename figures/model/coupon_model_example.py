"""Draw the Section 3 example as a vector figure; no probabilities are invented.

Run from any directory with Python and matplotlib. The panels illustrate possible realizations, not expected outcomes.
Panels (a)-(c) use two coupons; panel (d) shows a single-coupon revisit.
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



from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

BLUE = "#236A9B"
ORANGE = "#B35C21"
INK = "#222222"
GRAY = "#CCD0D4"
RADIUS = .14
POS = {"s1": (.38, 1.62), "s2": (.38, .64),
       "w": (1.63, 1.13), "u": (2.88, 1.62), "v": (2.88, .64)}
LABEL = {"s1": r"$s_1$", "s2": r"$s_2$", "w": r"$w$",
         "u": r"$u$", "v": r"$v$"}
EDGES = [("s1","w"),("s2","w"),("w","u"),("w","v")]


def arrow(ax, a, b, color, style="solid", curve=0, width=1.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
        connectionstyle=f"arc3,rad={curve}", mutation_scale=9,
        linewidth=width, color=color, linestyle=style,
        shrinkA=11, shrinkB=11, zorder=2))


def node(ax, point, label, active=False):
    ax.add_patch(Circle(point, RADIUS, facecolor=INK if active else "white",
                       edgecolor=INK, linewidth=.95, zorder=4))
    ax.text(*point,label,ha="center",va="center",fontsize=11,
            color="white" if active else INK,zorder=5)


def badge(ax, x, y, number, color):
    ax.add_patch(FancyBboxPatch((x-.095,y-.07),.19,.14,
                 boxstyle="round,pad=0.015,rounding_size=0.015",
                 linewidth=.8,edgecolor=color,facecolor="white",zorder=5))
    ax.text(x,y,str(number),ha="center",va="center",fontsize=8,
            color=color,fontweight="bold",zorder=6)


def summary(ax, note, result):
    ax.text(1.63,.18,note,ha="center",va="center",fontsize=8.5)
    ax.text(1.63,-.10,result,ha="center",va="center",fontsize=9,fontweight="bold")


fig = plt.figure(figsize=(7.05, 5.12), facecolor="white")
axes=[]
for row in range(2):
    for col in range(2):
        ax=fig.add_axes([.015+col*.5,.535-row*.43,.47,.415])
        ax.set_xlim(0,3.3);ax.set_ylim(-.30,2.18)
        ax.set_aspect("equal");ax.axis("off");axes.append(ax)

titles=["(a) Different consumers", "(b) Repeated consumption",
        "(c) A coupon is dropped", "(d) Revisiting the same user"]
for ax,title in zip(axes,titles):
    ax.text(1.63,2.09,title,ha="center",va="center",fontsize=10,fontweight="bold")

for i,ax in enumerate(axes[:3]):
    for a,b in EDGES:
        arrow(ax,POS[a],POS[b],GRAY,width=.7)
    for a,b in [("s1","w"),("w","u")]:
        arrow(ax,POS[a],POS[b],BLUE,curve=.10 if i==1 and a=="w" else 0)
    arrow(ax,POS['s2'],POS['w'],ORANGE,(0,(3,1.5)))
    if i==0:
        arrow(ax,POS['w'],POS['v'],ORANGE,(0,(3,1.5)))
    elif i==1:
        arrow(ax,POS['w'],POS['u'],ORANGE,(0,(3,1.5)),curve=-.13)
    active={"u","v"} if i==0 else {"u"}
    for key,point in POS.items():node(ax,point,LABEL[key],key in active)
    badge(ax,.67,1.72,1,BLUE);badge(ax,.67,.54,2,ORANGE)
    if i==0:
        ax.text(1.63,.70,"Forwards only",ha="center",fontsize=8.5)
        summary(ax,"Forwarding alone does not activate a user.","2 coupons consumed / 2 users activated")
    elif i==1:
        ax.text(2.65,1.88,"Consumes both",ha="center",fontsize=8.5)
        ax.text(1.63,.70,"Forwards only",ha="center",fontsize=8.5)
        summary(ax,"The same consumer is counted only once.","2 coupons consumed / 1 user activated")
    else:
        ax.text(1.63,.65,"Drops coupon 2",ha="center",fontsize=8.5,color=ORANGE)
        summary(ax,"Dropping ends that coupon's propagation.","1 coupon consumed / 1 user activated")

ax=axes[3]
points=[(.32,1.28),(1.18,1.28),(2.04,1.28),(2.90,1.28)]
for a,b in zip(points,points[1:]):arrow(ax,a,b,BLUE)
for i,(point,label) in enumerate(zip(points,[r"$s$",r"$w$",r"$x$",r"$w$"])):
    node(ax,point,label,i==3)
badge(ax,.32,1.65,1,BLUE)
ax.text(1.18,1.72,"First visit",ha="center",fontsize=8.5)
ax.text(2.90,1.72,"Second visit",ha="center",fontsize=8.5)
ax.text(1.18,.96,"Forwards",ha="center",fontsize=8.5)
ax.text(2.90,.96,"Consumes",ha="center",fontsize=8.5)
ax.plot([1.18,1.18,2.90,2.90],[.77,.67,.67,.77],color="#777777",lw=.8)
ax.text(2.04,.48,"Same user; a fresh decision",ha="center",fontsize=8.5)
summary(ax,"One coupon's trajectory, shown in visit order.","1 coupon consumed / 1 user activated")

# Shared legend keeps panel labels short and remains readable in grayscale.
handles=[Line2D([0],[0],color=BLUE,lw=1.6,label="Coupon 1"),
         Line2D([0],[0],color=ORANGE,lw=1.6,linestyle=(0,(3,1.5)),label="Coupon 2"),
         Line2D([0],[0],marker="o",markersize=6,color=INK,markerfacecolor=INK,
                linestyle="none",label="Activated"),
         Line2D([0],[0],marker="o",markersize=6,color=INK,markerfacecolor="white",
                linestyle="none",label="Not activated")]
fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.018),ncol=4,
           frameon=False,fontsize=8.5,columnspacing=1.6,handlelength=2)
fig.lines.extend([Line2D([.50,.50],[.13,.94],transform=fig.transFigure,color="#E0E0E0",lw=.6),
                  Line2D([.035,.965],[.53,.53],transform=fig.transFigure,color="#E0E0E0",lw=.6)])
fig.savefig(OUT / "coupon-model-example.pdf",metadata={"Title":"Coupon propagation: distinct consumers, overlap, dropping, and revisits"})
fig.savefig(OUT / "coupon-model-example.svg")
fig.savefig(OUT / "coupon-model-example.png",dpi=240)
plt.close(fig)
print("Generated PDF, SVG, and PNG in",OUT)
