#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Set path to save the figures and data files
FIGURE_PATH = "../Figures"


def fig_path(fig_id):
    """

    """
    return os.path.join(FIGURE_PATH + "/", fig_id)
