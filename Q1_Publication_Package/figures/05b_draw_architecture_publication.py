#!/usr/bin/env python3
"""
05b_draw_architecture_publication.py
=============================================================================
Next-Generation Publication-Grade Vector Architecture Diagram for RASNet.
Designed specifically for Q1 Medical Imaging Journals (IEEE TMI / MedIA / RSNA).
All arrows cleanly routed around text and container boxes.
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = r"H:\Thesis_Trainings\Q1_Publication_Package\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PNG_PATH = os.path.join(OUTPUT_DIR, "architecture_diagram_v2.png")
SVG_PATH = os.path.join(OUTPUT_DIR, "architecture_diagram_v2.svg")

def draw_rounded_card(ax, xy, w, h, bg_color, border_color, border_width=1.5, radius=0.08, zorder=2, shadow=True):
    """Draws a premium rounded rectangular card with subtle drop shadow."""
    x, y = xy
    if shadow:
        shadow_box = FancyBboxPatch(
            (x + 0.04, y - 0.04), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor="#000000", edgecolor="none",
            alpha=0.07, zorder=zorder-1
        )
        ax.add_patch(shadow_box)
        
    card = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color,
        linewidth=border_width, zorder=zorder
    )
    ax.add_patch(card)
    return card

def draw_arrow_curve(ax, start, end, color="#374151", lw=1.6, rad=0.0, style='-|>', ls='-', zorder=5):
    """Draws a crisp publication-quality connection arrow."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=12,
        color=color,
        linewidth=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow

def build_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300, facecolor='#ffffff')
    ax.set_xlim(-0.2, 16.2)
    ax.set_ylim(-0.2, 9.8)
    ax.axis('off')

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Header & Title Banner
    # ─────────────────────────────────────────────────────────────────────────
    ax.text(8.0, 9.48, "RASNet: 3D Residual Attention Network with Multi-Scale Deep Supervision",
            ha='center', va='center', fontsize=15.0, fontweight='bold', color='#0f172a')
    ax.text(8.0, 9.18, "High-Precision Coronary Artery Segmentation & Stenosis Quantification in CCTA",
            ha='center', va='center', fontsize=10.5, color='#475569')

    # ─────────────────────────────────────────────────────────────────────────
    # Color Palette Tokens
    # ─────────────────────────────────────────────────────────────────────────
    C_ENC_BG = "#eff6ff"     # Blue-50
    C_ENC_BD = "#3b82f6"     # Blue-500
    C_ENC_TX = "#1e3a8a"     # Blue-900

    C_BOT_BG = "#ede9fe"     # Purple-50
    C_BOT_BD = "#8b5cf6"     # Purple-500
    C_BOT_TX = "#4c1d95"     # Purple-900

    C_AG_BG  = "#ecfdf5"     # Emerald-50
    C_AG_BD  = "#10b981"     # Emerald-500
    C_AG_TX  = "#064e3b"     # Emerald-900

    C_DEC_BG = "#fff7ed"     # Orange-50
    C_DEC_BD = "#f97316"     # Orange-500
    C_DEC_TX = "#7c2d12"     # Orange-900

    C_AUX_BG = "#fdf4ff"     # Fuchsia-50
    C_AUX_BD = "#d946ef"     # Fuchsia-500
    C_AUX_TX = "#701a75"     # Fuchsia-900

    C_LOSS_BG = "#fefce8"    # Yellow-50
    C_LOSS_BD = "#eab308"    # Yellow-500
    C_LOSS_TX = "#713f12"    # Yellow-900

    C_IO_BG   = "#f8fafc"    # Slate-50
    C_IO_BD   = "#64748b"    # Slate-500
    C_IO_TX   = "#0f172a"    # Slate-900

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Left Column: Input Volume & Inset Attention Gate Details
    # ─────────────────────────────────────────────────────────────────────────
    # Input Volume Card
    draw_rounded_card(ax, (0.3, 7.3), 1.9, 1.3, C_IO_BG, C_IO_BD, radius=0.06)
    ax.text(1.25, 8.25, "Input CCTA Patch", ha='center', va='center', fontsize=9.5, fontweight='bold', color=C_IO_TX)
    ax.text(1.25, 7.85, "1 × 96 × 96 × 96", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0284c7')
    ax.text(1.25, 7.55, "0.5 mm Isotropic\nHU: [-100, 800]", ha='center', va='center', fontsize=7.5, color='#64748b')

    # Attention Gate Inset Container Card
    draw_rounded_card(ax, (0.3, 0.4), 4.2, 6.4, "#f8fafc", "#cbd5e1", border_width=1.5, radius=0.08)
    ax.text(2.4, 6.50, "AttentionGate3D Detail Mechanism", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#065f46')
    ax.text(2.4, 6.25, "Spatial & Channel Attention Skip Filtering", ha='center', va='center',
            fontsize=8.0, color='#047857', style='italic')

    # Inset components
    # Skip input (x)
    draw_rounded_card(ax, (0.55, 5.25), 1.4, 0.7, C_ENC_BG, C_ENC_BD, radius=0.05)
    ax.text(1.25, 5.68, "Skip Feature (x)", ha='center', va='center', fontsize=8.0, fontweight='bold', color=C_ENC_TX)
    ax.text(1.25, 5.42, "Encoder Layer l", ha='center', va='center', fontsize=7.0, color='#3b82f6')

    # Gating input (g)
    draw_rounded_card(ax, (0.55, 4.10), 1.4, 0.7, C_DEC_BG, C_DEC_BD, radius=0.05)
    ax.text(1.25, 4.53, "Gating Signal (g)", ha='center', va='center', fontsize=8.0, fontweight='bold', color=C_DEC_TX)
    ax.text(1.25, 4.27, "Decoder Layer l+1", ha='center', va='center', fontsize=7.0, color='#f97316')

    # W_x 1x1x1 Conv
    draw_rounded_card(ax, (2.25, 5.25), 0.9, 0.7, "#ffffff", "#94a3b8", radius=0.04)
    ax.text(2.7, 5.65, "Conv3D", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#334155')
    ax.text(2.7, 5.40, "1×1×1, W_x", ha='center', va='center', fontsize=7.0, color='#64748b')

    # W_g 1x1x1 Conv
    draw_rounded_card(ax, (2.25, 4.10), 0.9, 0.7, "#ffffff", "#94a3b8", radius=0.04)
    ax.text(2.7, 4.50, "Conv3D", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#334155')
    ax.text(2.7, 4.25, "1×1×1, W_g", ha='center', va='center', fontsize=7.0, color='#64748b')

    # Connect x -> Wx, g -> Wg
    draw_arrow_curve(ax, (1.95, 5.60), (2.25, 5.60), lw=1.2)
    draw_arrow_curve(ax, (1.95, 4.45), (2.25, 4.45), lw=1.2)

    # Element-wise Addition node
    circle_add = plt.Circle((3.5, 4.85), 0.22, facecolor="#fef08a", edgecolor="#ca8a04", linewidth=1.3, zorder=4)
    ax.add_patch(circle_add)
    ax.text(3.5, 4.85, "+", ha='center', va='center', fontsize=11, fontweight='bold', color="#854d0e", zorder=5)

    draw_arrow_curve(ax, (3.15, 5.60), (3.35, 5.0), lw=1.2)
    draw_arrow_curve(ax, (3.15, 4.45), (3.35, 4.7), lw=1.2)

    # Activation blocks: ReLU -> Conv_psi -> Sigmoid
    draw_rounded_card(ax, (0.55, 2.75), 3.7, 1.05, "#f1f5f9", "#94a3b8", radius=0.05)
    ax.text(2.4, 3.48, "Non-Linear Activation & Attention Coefficient", ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#334155')
    ax.text(2.4, 3.10, r"$\psi = \sigma \left(\, W_{\psi} \cdot \mathrm{ReLU}( W_g \cdot g + W_x \cdot x + b_g ) \,\right)$",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='#059669')

    draw_arrow_curve(ax, (3.5, 4.63), (3.5, 3.80), lw=1.2)

    # Multiplier node
    circle_mult = plt.Circle((3.5, 1.7), 0.22, facecolor="#bbf7d0", edgecolor="#16a34a", linewidth=1.3, zorder=4)
    ax.add_patch(circle_mult)
    ax.text(3.5, 1.7, "⨂", ha='center', va='center', fontsize=11, fontweight='bold', color="#166534", zorder=5)

    draw_arrow_curve(ax, (3.5, 2.75), (3.5, 1.92), lw=1.2)

    # Clean bypass from skip input x down to Multiplier (around the left, clear of text)
    draw_arrow_curve(ax, (0.55, 5.5), (0.42, 3.0), rad=0.2, lw=1.3, color="#2563eb", ls='--', style='-')
    draw_arrow_curve(ax, (0.42, 3.0), (3.28, 1.7), rad=-0.15, lw=1.3, color="#2563eb", ls='--')
    ax.text(0.48, 2.1, "Original Skip (x)", ha='left', va='center', fontsize=7.0, color="#2563eb", style='italic', rotation=25)

    # Output: Filtered Skip
    draw_rounded_card(ax, (0.55, 0.65), 3.7, 0.75, C_AG_BG, C_AG_BD, radius=0.05)
    ax.text(2.4, 1.15, r"Attended Feature Map:  $\hat{x} = x \odot \psi$", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=C_AG_TX)
    ax.text(2.4, 0.85, "Suppresses Non-Vascular Background & Enhances Distal Tips", ha='center', va='center',
            fontsize=7.0, color='#047857')

    draw_arrow_curve(ax, (3.5, 1.48), (3.5, 1.40), lw=1.2)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Main Network Grid Layout (Encoder, AG, Decoder, Heads)
    # ─────────────────────────────────────────────────────────────────────────
    COL_ENC = 5.0
    COL_AG  = 7.7
    COL_DEC = 10.4
    COL_AUX = 13.1

    ROW_L1  = 7.3
    ROW_L2  = 5.3
    ROW_L3  = 3.3
    ROW_BOT = 1.3

    BOX_W   = 1.9
    BOX_H   = 1.3

    # Connect Input -> Encoder L1
    draw_arrow_curve(ax, (2.2, 7.95), (COL_ENC, 7.95), lw=1.8, color="#3b82f6")

    # ── ENCODER BLOCKS ──
    # Level 1
    draw_rounded_card(ax, (COL_ENC, ROW_L1), BOX_W, BOX_H, C_ENC_BG, C_ENC_BD)
    ax.text(COL_ENC + BOX_W/2, ROW_L1 + 0.95, "InitConv + ResBlock 1", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_ENC_TX)
    ax.text(COL_ENC + BOX_W/2, ROW_L1 + 0.65, "16 Channels", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#2563eb')
    ax.text(COL_ENC + BOX_W/2, ROW_L1 + 0.35, "Spatial: 96 × 96 × 96", ha='center', va='center', fontsize=7.5, color='#64748b')

    # Down L1 -> L2
    draw_arrow_curve(ax, (COL_ENC + BOX_W/2, ROW_L1), (COL_ENC + BOX_W/2, ROW_L2 + BOX_H), lw=1.6, color="#3b82f6")
    ax.text(COL_ENC + BOX_W/2 + 0.08, (ROW_L1 + ROW_L2 + BOX_H)/2, "Conv3D Str=2", ha='left', va='center', fontsize=7.0, color='#64748b')

    # Level 2
    draw_rounded_card(ax, (COL_ENC, ROW_L2), BOX_W, BOX_H, C_ENC_BG, C_ENC_BD)
    ax.text(COL_ENC + BOX_W/2, ROW_L2 + 0.95, "ResBlock Level 2", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_ENC_TX)
    ax.text(COL_ENC + BOX_W/2, ROW_L2 + 0.65, "32 Channels", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#2563eb')
    ax.text(COL_ENC + BOX_W/2, ROW_L2 + 0.35, "Spatial: 48 × 48 × 48", ha='center', va='center', fontsize=7.5, color='#64748b')

    # Down L2 -> L3
    draw_arrow_curve(ax, (COL_ENC + BOX_W/2, ROW_L2), (COL_ENC + BOX_W/2, ROW_L3 + BOX_H), lw=1.6, color="#3b82f6")
    ax.text(COL_ENC + BOX_W/2 + 0.08, (ROW_L2 + ROW_L3 + BOX_H)/2, "Conv3D Str=2", ha='left', va='center', fontsize=7.0, color='#64748b')

    # Level 3
    draw_rounded_card(ax, (COL_ENC, ROW_L3), BOX_W, BOX_H, C_ENC_BG, C_ENC_BD)
    ax.text(COL_ENC + BOX_W/2, ROW_L3 + 0.95, "ResBlock Level 3", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_ENC_TX)
    ax.text(COL_ENC + BOX_W/2, ROW_L3 + 0.65, "64 Channels", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#2563eb')
    ax.text(COL_ENC + BOX_W/2, ROW_L3 + 0.35, "Spatial: 24 × 24 × 24", ha='center', va='center', fontsize=7.5, color='#64748b')

    # Down L3 -> Bottleneck
    draw_arrow_curve(ax, (COL_ENC + BOX_W/2, ROW_L3), (COL_ENC + BOX_W/2, ROW_BOT + BOX_H), lw=1.6, color="#3b82f6")
    ax.text(COL_ENC + BOX_W/2 + 0.08, (ROW_L3 + ROW_BOT + BOX_H)/2, "Conv3D Str=2", ha='left', va='center', fontsize=7.0, color='#64748b')

    # Bottleneck Level 4
    draw_rounded_card(ax, (COL_ENC, ROW_BOT), BOX_W, BOX_H, C_BOT_BG, C_BOT_BD)
    ax.text(COL_ENC + BOX_W/2, ROW_BOT + 0.95, "Bottleneck Level 4", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_BOT_TX)
    ax.text(COL_ENC + BOX_W/2, ROW_BOT + 0.65, "128 Channels", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#7c3aed')
    ax.text(COL_ENC + BOX_W/2, ROW_BOT + 0.35, "Spatial: 12 × 12 × 12", ha='center', va='center', fontsize=7.5, color='#64748b')

    # ── ATTENTION GATES (COLUMN 2) ──
    # AG Level 1
    draw_rounded_card(ax, (COL_AG, ROW_L1), BOX_W, BOX_H, C_AG_BG, C_AG_BD)
    ax.text(COL_AG + BOX_W/2, ROW_L1 + 0.95, "AttentionGate3D (1)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_AG_TX)
    ax.text(COL_AG + BOX_W/2, ROW_L1 + 0.65, "F_g = 16, F_l = 16", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#059669')
    ax.text(COL_AG + BOX_W/2, ROW_L1 + 0.35, "Output: 16 × 96³", ha='center', va='center', fontsize=7.5, color='#047857')

    # AG Level 2
    draw_rounded_card(ax, (COL_AG, ROW_L2), BOX_W, BOX_H, C_AG_BG, C_AG_BD)
    ax.text(COL_AG + BOX_W/2, ROW_L2 + 0.95, "AttentionGate3D (2)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_AG_TX)
    ax.text(COL_AG + BOX_W/2, ROW_L2 + 0.65, "F_g = 32, F_l = 32", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#059669')
    ax.text(COL_AG + BOX_W/2, ROW_L2 + 0.35, "Output: 32 × 48³", ha='center', va='center', fontsize=7.5, color='#047857')

    # AG Level 3
    draw_rounded_card(ax, (COL_AG, ROW_L3), BOX_W, BOX_H, C_AG_BG, C_AG_BD)
    ax.text(COL_AG + BOX_W/2, ROW_L3 + 0.95, "AttentionGate3D (3)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_AG_TX)
    ax.text(COL_AG + BOX_W/2, ROW_L3 + 0.65, "F_g = 64, F_l = 64", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#059669')
    ax.text(COL_AG + BOX_W/2, ROW_L3 + 0.35, "Output: 64 × 24³", ha='center', va='center', fontsize=7.5, color='#047857')

    # Encoder -> AG horizontal connections
    draw_arrow_curve(ax, (COL_ENC + BOX_W, ROW_L1 + 0.65), (COL_AG, ROW_L1 + 0.65), lw=1.6, color="#059669")
    draw_arrow_curve(ax, (COL_ENC + BOX_W, ROW_L2 + 0.65), (COL_AG, ROW_L2 + 0.65), lw=1.6, color="#059669")
    draw_arrow_curve(ax, (COL_ENC + BOX_W, ROW_L3 + 0.65), (COL_AG, ROW_L3 + 0.65), lw=1.6, color="#059669")

    # ── DECODER BLOCKS (COLUMN 3) ──
    # Decoder Up L3
    draw_rounded_card(ax, (COL_DEC, ROW_L3), BOX_W, BOX_H, C_DEC_BG, C_DEC_BD)
    ax.text(COL_DEC + BOX_W/2, ROW_L3 + 0.95, "Decoder Up L3", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_DEC_TX)
    ax.text(COL_DEC + BOX_W/2, ROW_L3 + 0.65, "64 Channels", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#ea580c')
    ax.text(COL_DEC + BOX_W/2, ROW_L3 + 0.35, "Spatial: 24 × 24 × 24", ha='center', va='center', fontsize=7.5, color='#64748b')

    # Decoder Up L2
    draw_rounded_card(ax, (COL_DEC, ROW_L2), BOX_W, BOX_H, C_DEC_BG, C_DEC_BD)
    ax.text(COL_DEC + BOX_W/2, ROW_L2 + 0.95, "Decoder Up L2", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_DEC_TX)
    ax.text(COL_DEC + BOX_W/2, ROW_L2 + 0.65, "32 Channels", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#ea580c')
    ax.text(COL_DEC + BOX_W/2, ROW_L2 + 0.35, "Spatial: 48 × 48 × 48", ha='center', va='center', fontsize=7.5, color='#64748b')

    # Decoder Up L1
    draw_rounded_card(ax, (COL_DEC, ROW_L1), BOX_W, BOX_H, C_DEC_BG, C_DEC_BD)
    ax.text(COL_DEC + BOX_W/2, ROW_L1 + 0.95, "Decoder Up L1", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_DEC_TX)
    ax.text(COL_DEC + BOX_W/2, ROW_L1 + 0.65, "16 Channels", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#ea580c')
    ax.text(COL_DEC + BOX_W/2, ROW_L1 + 0.35, "Spatial: 96 × 96 × 96", ha='center', va='center', fontsize=7.5, color='#64748b')

    # Bottleneck up to Decoder L3 (curved cleanly)
    draw_arrow_curve(ax, (COL_ENC + BOX_W, ROW_BOT + 0.65), (COL_DEC, ROW_L3 + 0.15), rad=-0.22, lw=1.8, color="#8b5cf6")
    ax.text(8.2, 2.3, "Trilinear Upsample ×2 + Conv", ha='center', va='center', fontsize=7.5, color="#8b5cf6", fontweight='semibold', rotation=18)

    # Decoder vertical upsampling
    draw_arrow_curve(ax, (COL_DEC + BOX_W/2, ROW_L3 + BOX_H), (COL_DEC + BOX_W/2, ROW_L2), lw=1.6, color="#ea580c")
    ax.text(COL_DEC + BOX_W/2 + 0.08, (ROW_L3 + BOX_H + ROW_L2)/2, "Up ×2", ha='left', va='center', fontsize=7.0, color="#ea580c")

    draw_arrow_curve(ax, (COL_DEC + BOX_W/2, ROW_L2 + BOX_H), (COL_DEC + BOX_W/2, ROW_L1), lw=1.6, color="#ea580c")
    ax.text(COL_DEC + BOX_W/2 + 0.08, (ROW_L2 + BOX_H + ROW_L1)/2, "Up ×2", ha='left', va='center', fontsize=7.0, color="#ea580c")

    # AG -> Decoder connections (concatenation / fusion)
    draw_arrow_curve(ax, (COL_AG + BOX_W, ROW_L1 + 0.65), (COL_DEC, ROW_L1 + 0.65), lw=1.6, color="#059669")
    draw_arrow_curve(ax, (COL_AG + BOX_W, ROW_L2 + 0.65), (COL_DEC, ROW_L2 + 0.65), lw=1.6, color="#059669")
    draw_arrow_curve(ax, (COL_AG + BOX_W, ROW_L3 + 0.65), (COL_DEC, ROW_L3 + 0.65), lw=1.6, color="#059669")

    # Gating signals from Decoder back into Attention Gates (dotted green arrows)
    draw_arrow_curve(ax, (COL_DEC, ROW_L1 + 0.95), (COL_AG + BOX_W, ROW_L1 + 0.95), lw=1.2, color="#059669", ls=':')
    draw_arrow_curve(ax, (COL_DEC, ROW_L2 + 0.95), (COL_AG + BOX_W, ROW_L2 + 0.95), lw=1.2, color="#059669", ls=':')
    draw_arrow_curve(ax, (COL_DEC, ROW_L3 + 0.95), (COL_AG + BOX_W, ROW_L3 + 0.95), lw=1.2, color="#059669", ls=':')

    # ── AUXILIARY HEADS & FINAL PREDICTION (COLUMN 4) ──
    # Aux Head 3
    draw_rounded_card(ax, (COL_AUX, ROW_L3), 2.2, BOX_H, C_AUX_BG, C_AUX_BD)
    ax.text(COL_AUX + 1.1, ROW_L3 + 0.95, "Deep Supervision aux3", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_AUX_TX)
    ax.text(COL_AUX + 1.1, ROW_L3 + 0.65, "Conv 1×1×1 (2 Ch)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#c026d3')
    ax.text(COL_AUX + 1.1, ROW_L3 + 0.35, "Scale: 1/4 (24³ → 96³)", ha='center', va='center', fontsize=7.5, color='#86198f')

    # Aux Head 2
    draw_rounded_card(ax, (COL_AUX, ROW_L2), 2.2, BOX_H, C_AUX_BG, C_AUX_BD)
    ax.text(COL_AUX + 1.1, ROW_L2 + 0.95, "Deep Supervision aux2", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_AUX_TX)
    ax.text(COL_AUX + 1.1, ROW_L2 + 0.65, "Conv 1×1×1 (2 Ch)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#c026d3')
    ax.text(COL_AUX + 1.1, ROW_L2 + 0.35, "Scale: 1/2 (48³ → 96³)", ha='center', va='center', fontsize=7.5, color='#86198f')

    # Main Output Head (Level 1)
    draw_rounded_card(ax, (COL_AUX, ROW_L1), 2.2, BOX_H, "#f8fafc", "#0f172a", border_width=2.0)
    ax.text(COL_AUX + 1.1, ROW_L1 + 0.95, "Final Output Head", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#0f172a')
    ax.text(COL_AUX + 1.1, ROW_L1 + 0.65, "Conv 1×1×1 (2 Ch)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#2563eb')
    ax.text(COL_AUX + 1.1, ROW_L1 + 0.35, "Full Scale: 2 × 96³", ha='center', va='center', fontsize=7.5, color='#475569')

    # Decoder -> Heads arrows
    draw_arrow_curve(ax, (COL_DEC + BOX_W, ROW_L1 + 0.65), (COL_AUX, ROW_L1 + 0.65), lw=1.8, color="#0f172a")
    draw_arrow_curve(ax, (COL_DEC + BOX_W, ROW_L2 + 0.65), (COL_AUX, ROW_L2 + 0.65), lw=1.6, color="#c026d3")
    draw_arrow_curve(ax, (COL_DEC + BOX_W, ROW_L3 + 0.65), (COL_AUX, ROW_L3 + 0.65), lw=1.6, color="#c026d3")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Multi-Scale StenosisAwareLoss Formulation Block (Bottom Right Banner)
    # ─────────────────────────────────────────────────────────────────────────
    draw_rounded_card(ax, (9.0, 0.4), 6.7, 2.2, C_LOSS_BG, C_LOSS_BD, border_width=1.8, radius=0.08)
    ax.text(12.35, 2.25, "Multi-Scale StenosisAwareLoss Formulation (Deep Supervision)", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=C_LOSS_TX)
    
    eq1 = r"$\mathcal{L}_{\mathrm{total}} = 1.0 \cdot \mathcal{L}_{\mathrm{main}} + 0.4 \cdot \mathcal{L}_{\mathrm{aux2}} + 0.2 \cdot \mathcal{L}_{\mathrm{aux3}}$"
    eq2 = r"$\mathcal{L}(\hat{y}, y) = 0.4 \cdot \mathcal{L}_{\mathrm{Dice}}(\hat{y}, y) + 0.6 \cdot \mathcal{L}_{\mathrm{Focal}}(\hat{y}, y; \gamma=2.5)$"
    
    ax.text(12.35, 1.80, eq1, ha='center', va='center', fontsize=9.5, fontweight='bold', color="#b45309")
    ax.text(12.35, 1.35, eq2, ha='center', va='center', fontsize=9.0, fontweight='bold', color="#b45309")
    
    ax.text(12.35, 0.92, "• Focal Loss (γ = 2.5) suppresses easy background negatives (severe class imbalance)",
            ha='center', va='center', fontsize=7.5, color="#78350f")
    ax.text(12.35, 0.65, "• Dice Loss + Attention Gates preserve sub-millimeter distal vascular tree topology",
            ha='center', va='center', fontsize=7.5, color="#78350f")

    # Clean non-overlapping Loss routing from the right side of heads:
    draw_arrow_curve(ax, (COL_AUX + 2.2, ROW_L1 + 0.65), (15.8, 4.0), rad=-0.15, lw=1.4, color="#b45309", ls='--', style='-')
    draw_arrow_curve(ax, (COL_AUX + 2.2, ROW_L2 + 0.65), (15.75, 4.0), rad=-0.1, lw=1.4, color="#b45309", ls='--', style='-')
    draw_arrow_curve(ax, (COL_AUX + 2.2, ROW_L3 + 0.65), (15.7, 4.0), rad=-0.05, lw=1.4, color="#b45309", ls='--', style='-')
    
    # Combined line down to loss block
    draw_arrow_curve(ax, (15.75, 4.0), (14.2, 2.6), rad=0.15, lw=1.5, color="#b45309", ls='--')
    ax.text(15.9, 3.3, "Supervised Loss", ha='center', va='center', fontsize=7.5, color="#b45309", fontweight='bold', rotation=-90)

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.savefig(SVG_PATH, format='svg', bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    
    print(f"[OK] Upgraded architecture diagram successfully regenerated:")
    print(f"     -> PNG: {PNG_PATH}")
    print(f"     -> SVG: {SVG_PATH}")

if __name__ == "__main__":
    build_architecture_diagram()
