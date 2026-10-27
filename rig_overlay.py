"""Puntos visibles del mismo fotograma usado por reconocimiento y captura."""
import numpy as np
from recognition_features import regions


def rig_payload(points, width, height):
    pose, face, left, right = regions(np.asarray(points))

    def visible_points(region, visibility=False):
        output = []
        for point in region:
            present = np.any(point[:3] != 0) and np.isfinite(point).all()
            if visibility:
                present = present and point[3] >= .5
            output.append([float(point[0]), float(point[1])] if present else None)
        return output

    return {'width': width, 'height': height, 'pose': visible_points(pose, True),
            'face': visible_points(face), 'left': visible_points(left), 'right': visible_points(right)}
