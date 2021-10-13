#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper
See LICENCE.txt for licensing and contact information.
"""

__all__ = ['ColoredRenderer']

from copy import deepcopy

import numpy as np
from chumpy import *
from opendr.contexts._constants import *
from opendr.renderer import ColoredRenderer as OpendrRenderer
from opendr.renderer import draw_colored_verts


class ColoredRenderer(OpendrRenderer):
    terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
    dterms = 'vc', 'camera', 'bgcolor'

    def draw_color_image(self, gl):
        self._call_on_changed()
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # use face colors if given
        # FIXME: this won't work for 2 channels
        draw_colored_verts(gl, self.v.r, self.f, self.vc.r)

        result = np.asarray(deepcopy(gl.getImage()[:, :, :self.num_channels].squeeze()), np.float64)

        bg_px = np.tile(np.atleast_3d(self.visibility_image) == 4294967295, (1, 1, self.num_channels)).squeeze()
        fg_px = 1 - bg_px
        if hasattr(self, 'background_image'):
            result = bg_px * self.background_image + fg_px * result

        result = np.concatenate([result, fg_px[:, :, :1]], axis=2)
        return result
