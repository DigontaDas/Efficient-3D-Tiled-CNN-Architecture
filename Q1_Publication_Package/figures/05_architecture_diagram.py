#!/usr/bin/env python3
"""
05_architecture_diagram.py
=============================================================================
Publication-Grade Vector Architecture Diagram for RASNet.

Author: Digonta Das / Nafis Mehedi
Project: Efficient 3D Tiled CNN Architecture (RASNet, ImageCAS Dataset)
Target: Q1 Medical Imaging Journal Submission

Renders:
  1. Multi-scale 3D ResNet Encoder Pathway
  2. AttentionGate3D Inset Schematic (W_g, W_x, ReLU, Sigmoid psi, Multiplier)
  3. Multi-Scale Decoder with Skip Attention Gates
  4. Auxiliary Deep Supervision Prediction Heads (aux2, aux3)
  5. StenosisAwareLoss Formulation Block
  6. Double-Column Journal Width (~7.2 in), 300 DPI Raster + SVG Vector
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.0


def draw_box(ax, xy, width, height, text, facecolor, edgecolor="#222222", text_color="#111111", fontsize=8.5, fontweight='bold', alpha=0.95):
    """Draw a clean rounded box with centered text."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        alpha=alpha,
        zorder=3
    )
    ax.add_patch(box)
    
    cx = xy[0] + width / 2.0
    cy = xy[1] + height / 2.0
    ax.text(cx, cy, text, ha='center', va='center', color=text_color,
            fontsize=fontsize, fontweight=fontweight, multialignment='center', zorder=4)
    return box


def draw_arrow(ax, start, end, color="#333333", style="->", lw=1.3, linestyle='solid', rad=0.0):
    """Draw a clean arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=11,
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2
    )
    ax.add_patch(arrow)
    return arrow


def create_architecture_figure():
    """Build the comprehensive RASNet publication architecture diagram."""
    fig, ax = plt.subplots(figsize=(14.5, 8.5), dpi=300)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')
    
    # Background panel container
    bg = FancyBboxPatch((-0.3, -0.3), 14.8, 8.6, boxstyle="square,pad=0", facecolor="#fafafa", edgecolor="#cccccc", linewidth=1.0)
    ax.add_patch(bg)
    
    # Title Header
    ax.text(7.1, 8.05, "RASNet: Residual Attention Segmentation Network with Deep Supervision & StenosisAwareLoss",
            ha='center', va='center', fontsize=13, fontweight='bold', color="#0f2027")
    
    # Color palette
    c_enc = "#d0e1fd"       # Encoder blocks (Soft Blue)
    c_dec = "#fed9c9"       # Decoder blocks (Soft Coral)
    c_gate = "#c7e9c0"      # Attention Gate (Soft Green)
    c_ds = "#fdd0a2"        # Deep Supervision (Soft Orange)
    c_loss = "#ffffcc"      # Loss block (Soft Yellow)
    c_io = "#e2e2e2"        # Input / Output (Light Gray)
    
    # ── 1. INPUT ────────────────────────────────────────────────────────────────
    draw_box(ax, (0.0, 4.6), 1.6, 0.9, "Input CCTA\nVolume\n(1×96×96×96)", c_io, edgecolor="#555555")
    draw_arrow(ax, (1.6, 5.05), (2.1, 5.05))
    
    # ── 2. ENCODER HIERARCHY ──────────────────────────────────────────────────
    # Initial Conv
    draw_box(ax, (2.1, 4.5), 1.5, 1.1, "Init Conv 3D\n(16 Channels)\n[96³]", c_enc, edgecolor="#1f4e79")
    draw_arrow(ax, (3.6, 5.05), (4.0, 5.05))
    
    # Encoder Level 1
    draw_box(ax, (4.0, 4.5), 1.7, 1.1, "ResBlock L1\n(16 Channels)\n[96³]", c_enc, edgecolor="#1f4e79")
    draw_arrow(ax, (4.85, 4.5), (4.85, 3.8), color="#1f4e79", style="->")
    
    # Encoder Level 2 (Downsampled)
    draw_box(ax, (4.0, 2.7), 1.7, 1.1, "ResBlock L2\n(32 Channels)\n[48³ ↓]", c_enc, edgecolor="#1f4e79")
    draw_arrow(ax, (4.85, 2.7), (4.85, 2.0), color="#1f4e79", style="->")
    
    # Encoder Level 3 (Downsampled)
    draw_box(ax, (4.0, 0.9), 1.7, 1.1, "ResBlock L3\n(64 Channels)\n[24³ ↓]", c_enc, edgecolor="#1f4e79")
    draw_arrow(ax, (4.85, 0.9), (4.85, 0.2), color="#1f4e79", style="->")
    
    # Bottleneck Level 4 (Downsampled)
    draw_box(ax, (4.0, -0.1), 1.7, 0.3, "Bottleneck L4 (128 Ch) [12³ ↓]", "#9ecae1", edgecolor="#08519c", fontsize=7.5)
    
    # ── 3. ATTENTION GATES (SKIP PATHS) ───────────────────────────────────────
    # Gate 3 (Deepest skip: L3)
    draw_box(ax, (6.5, 0.9), 1.6, 1.1, "AttentionGate 3D\n(F_g=64, F_l=64)\n[24³ Skip]", c_gate, edgecolor="#238b45")
    # Gate 2 (Mid skip: L2)
    draw_box(ax, (6.5, 2.7), 1.6, 1.1, "AttentionGate 3D\n(F_g=32, F_l=32)\n[48³ Skip]", c_gate, edgecolor="#238b45")
    # Gate 1 (Shallow skip: L1)
    draw_box(ax, (6.5, 4.5), 1.6, 1.1, "AttentionGate 3D\n(F_g=16, F_l=16)\n[96³ Skip]", c_gate, edgecolor="#238b45")
    
    # Arrows from Encoder to Attention Gates (x skip)
    draw_arrow(ax, (5.7, 1.45), (6.5, 1.45), color="#238b45", lw=1.4)
    draw_arrow(ax, (5.7, 3.25), (6.5, 3.25), color="#238b45", lw=1.4)
    draw_arrow(ax, (5.7, 5.05), (6.5, 5.05), color="#238b45", lw=1.4)
    
    # ── 4. DECODER PATHWAY ────────────────────────────────────────────────────
    # Bottom connection from L4 to Decoder 3
    draw_arrow(ax, (5.7, 0.05), (9.75, 0.05), color="#b30000", lw=1.3)
    draw_arrow(ax, (9.75, 0.05), (9.75, 0.9), color="#b30000", lw=1.3)
    
    # Decoder Level 3
    draw_box(ax, (8.9, 0.9), 1.7, 1.1, "Decoder Up L3\n(64 Channels)\n[24³]", c_dec, edgecolor="#b30000")
    draw_arrow(ax, (8.1, 1.45), (8.9, 1.45), color="#238b45", lw=1.4)  # Attended skip to decoder
    draw_arrow(ax, (8.9, 1.3), (8.1, 1.3), color="#6baed6", lw=1.2, style="->")  # Gating signal to gate
    draw_arrow(ax, (9.75, 2.0), (9.75, 2.7), color="#b30000", lw=1.3)
    
    # Decoder Level 2
    draw_box(ax, (8.9, 2.7), 1.7, 1.1, "Decoder Up L2\n(32 Channels)\n[48³]", c_dec, edgecolor="#b30000")
    draw_arrow(ax, (8.1, 3.25), (8.9, 3.25), color="#238b45", lw=1.4)
    draw_arrow(ax, (8.9, 3.1), (8.1, 3.1), color="#6baed6", lw=1.2, style="->")
    draw_arrow(ax, (9.75, 3.8), (9.75, 4.5), color="#b30000", lw=1.3)
    
    # Decoder Level 1 (Final Stage)
    draw_box(ax, (8.9, 4.5), 1.7, 1.1, "Decoder Up L1\n(16 Channels)\n[96³]", c_dec, edgecolor="#b30000")
    draw_arrow(ax, (8.1, 5.05), (8.9, 5.05), color="#238b45", lw=1.4)
    draw_arrow(ax, (8.9, 4.9), (8.1, 4.9), color="#6baed6", lw=1.2, style="->")
    
    # Final Output Conv
    draw_arrow(ax, (10.6, 5.05), (11.0, 5.05), color="#b30000", lw=1.3)
    draw_box(ax, (11.0, 4.5), 1.4, 1.1, "Conv Final\n1×1×1\n(2 Channels)", "#d9d9d9", edgecolor="#333333")
    draw_arrow(ax, (12.4, 5.05), (12.8, 5.05), color="#333333", lw=1.3)
    
    # Output Mask
    draw_box(ax, (12.8, 4.6), 1.5, 0.9, "Predicted 3D\nCoronary Mask\n(2×96³)", "#bcbddc", edgecolor="#54278f", fontsize=8)
    
    # ── 5. DEEP SUPERVISION HEADS (aux2, aux3) ────────────────────────────────
    draw_box(ax, (11.0, 2.8), 1.5, 0.9, "Aux Head 2\n(1×1×1 Conv)\n[48³ Pred]", c_ds, edgecolor="#d94801", fontsize=8)
    draw_arrow(ax, (10.6, 3.25), (11.0, 3.25), color="#d94801", lw=1.2)
    
    draw_box(ax, (11.0, 1.0), 1.5, 0.9, "Aux Head 3\n(1×1×1 Conv)\n[24³ Pred]", c_ds, edgecolor="#d94801", fontsize=8)
    draw_arrow(ax, (10.6, 1.45), (11.0, 1.45), color="#d94801", lw=1.2)
    
    # ── 6. LOSS FORMULATION BLOCK ─────────────────────────────────────────────
    loss_text = (
        "Multi-Scale StenosisAwareLoss Formulation:\n"
        "L_total = 1.0 · L_main + 0.4 · L_aux2 + 0.2 · L_aux3\n"
        "where L_task = 0.4 · L_Dice + 0.6 · L_Focal(γ=2.5)\n"
        "  • L_Dice preserves continuous vessel topology & thin distal tips\n"
        "  • L_Focal (γ=2.5) suppresses severe background false-positives"
    )
    draw_box(ax, (8.5, 6.0), 5.7, 1.7, loss_text, c_loss, edgecolor="#b58900", text_color="#583b00", fontsize=8.0, fontweight='semibold')
    
    # Connect predictions to loss box
    draw_arrow(ax, (11.7, 5.6), (11.7, 6.0), color="#b58900", lw=1.2, linestyle='dashed')
    draw_arrow(ax, (12.5, 3.25), (12.5, 6.0), color="#d94801", lw=1.1, linestyle='dashed')
    draw_arrow(ax, (12.5, 1.45), (12.8, 6.0), color="#d94801", lw=1.1, linestyle='dashed', rad=0.2)
    
    # ── 7. ATTENTION GATE 3D INTERNAL INSET (Bottom Left) ─────────────────────
    inset_box = FancyBboxPatch((-0.2, 0.2), 3.8, 3.8, boxstyle="round,pad=0.05,rounding_size=0.1",
                               facecolor="#f7fcf5", edgecolor="#74c476", linewidth=1.3, alpha=0.95)
    ax.add_patch(inset_box)
    ax.text(1.7, 3.75, "AttentionGate3D Internal Schematic", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color="#005a32")
    
    # Internal Gate components
    # Inputs: x (skip) and g (gating)
    draw_box(ax, (0.0, 2.8), 0.9, 0.6, "Skip (x)\n[Encoder]", "#c7e9c0", edgecolor="#238b45", fontsize=7.0)
    draw_box(ax, (0.0, 1.9), 0.9, 0.6, "Gate (g)\n[Decoder]", "#fed9c9", edgecolor="#b30000", fontsize=7.0)
    
    # Convolutions W_x and W_g
    draw_box(ax, (1.2, 2.8), 0.8, 0.6, "W_x\n(1×1×1)", "#e5f5e0", edgecolor="#238b45", fontsize=7.0)
    draw_box(ax, (1.2, 1.9), 0.8, 0.6, "W_g\n(1×1×1)", "#fee5d9", edgecolor="#b30000", fontsize=7.0)
    draw_arrow(ax, (0.9, 3.1), (1.2, 3.1))
    draw_arrow(ax, (0.9, 2.2), (1.2, 2.2))
    
    # Element-wise Addition & ReLU
    draw_box(ax, (2.3, 2.35), 0.6, 0.7, "+\nReLU", "#ffffcc", edgecolor="#d95f02", fontsize=7.5)
    draw_arrow(ax, (2.0, 3.1), (2.3, 2.8))
    draw_arrow(ax, (2.0, 2.2), (2.3, 2.6))
    
    # 1x1 Conv + Sigmoid (psi)
    draw_box(ax, (2.3, 1.2), 0.6, 0.7, "ψ Conv\n+ σ(·)", "#d9f0a3", edgecolor="#78c679", fontsize=6.8)
    draw_arrow(ax, (2.6, 2.35), (2.6, 1.9))
    
    # Element-wise Multiplier (x * psi)
    draw_box(ax, (1.2, 0.5), 0.8, 0.6, "⊗ (Mult)", "#c7e9c0", edgecolor="#238b45", fontsize=8.0)
    draw_arrow(ax, (2.3, 1.55), (1.6, 1.1), rad=-0.2)  # psi to mult
    draw_arrow(ax, (0.45, 2.8), (0.45, 0.8), color="#238b45", rad=0.2)  # skip x to mult
    draw_arrow(ax, (0.45, 0.8), (1.2, 0.8), color="#238b45")
    
    # Output Attended Skip
    draw_box(ax, (2.3, 0.5), 1.1, 0.6, "Attended\nSkip (x̂)", "#74c476", edgecolor="#00441b", text_color="#ffffff", fontsize=7.0)
    draw_arrow(ax, (2.0, 0.8), (2.3, 0.8))
    
    # Inset label
    ax.text(1.7, 0.3, "x̂ = x ⊗ σ(ψ(ReLU(W_g·g + W_x·x)))", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color="#00441b")
    
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "architecture_diagram.png")
    svg_path = os.path.join(OUTPUT_DIR, "architecture_diagram.svg")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {png_path} (300 DPI)")
    print(f"Saved: {svg_path} (Vector)")


if __name__ == "__main__":
    print("=" * 80)
    print("STEP 5: PUBLICATION-GRADE VECTOR ARCHITECTURE DIAGRAM GENERATION")
    print("=" * 80)
    create_architecture_figure()
    print("=" * 80)
