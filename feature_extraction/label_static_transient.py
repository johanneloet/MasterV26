import numpy as np



def transient_score(ax, ay, az, gx, gy, gz):

    def mean_abs_temporal_diff(signal):
        return np.mean(np.abs(np.diff(signal)))

    score = (
        mean_abs_temporal_diff(ax) +
        mean_abs_temporal_diff(ay) +
        mean_abs_temporal_diff(az) +
        mean_abs_temporal_diff(gx) +
        mean_abs_temporal_diff(gy) +
        mean_abs_temporal_diff(gz)
    ) / 6.0

    return score