#!/usr/bin/env python3
"""
resume_rasnet_200ep.py — Seamless Forwarder to train_rasnet_200ep.py
====================================================================
User Directive: Stage 5 (RASNet) now runs a true 0 -> 200 epoch training from scratch
with random initialization, matching all other benchmark models.
Historical champion checkpoints and metrics are archived in _archive_rasnet_champion_70ep/.
"""

import sys
import train_rasnet_200ep

if __name__ == "__main__":
    train_rasnet_200ep.main()
