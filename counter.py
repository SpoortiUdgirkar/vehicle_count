"""
Vehicle Counting & Line-Crossing Event Manager
Detects line crossings, prevents duplicates, logs timestamps, and formats CSV/JSON reports.
"""
import time
import numpy as np


def ccw(A, B, C):
    """Checks counter-clockwise orientation of points A, B, C."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """
    Returns True if line segment AB intersects line segment CD.
    A, B: Previous and current centroids of tracked object [(x1,y1), (x2,y2)]
    C, D: Virtual line endpoints [(lx1,ly1), (lx2,ly2)]
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def get_crossing_direction(p_prev, p_curr, line_start, line_end):
    """
    Determines crossing direction relative to line segment.
    Returns 'top_to_bottom' or 'bottom_to_top'.
    Uses the signed cross product of the line normal with the movement vector.
    """
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]

    # Normal vector to the line (points "downward" for horizontal lines)
    nx = -dy
    ny = dx

    # Movement vector
    mx = p_curr[0] - p_prev[0]
    my = p_curr[1] - p_prev[1]

    dot_product = mx * nx + my * ny
    if dot_product > 0:
        return "top_to_bottom"
    else:
        return "bottom_to_top"


class VehicleCounter:
    def __init__(self, line_coords=((220, 420), (1080, 420)), target_direction="top_to_bottom"):
        self.line_start = tuple(line_coords[0])
        self.line_end   = tuple(line_coords[1])
        self.target_direction = target_direction

        # Per-ID duplicate prevention
        self.counted_ids       = set()
        self.previous_centroids = {}   # track_id -> (x, y)
        self.events            = []    # all crossing event records
        self.alert_queue       = []    # pending SSE events

        # Spatial+temporal cooldown: prevents same physical car (with a new ID
        # after an ID-switch) from being counted a second time.
        # Stores (timestamp_sec, centroid_x, centroid_y) of recent crossings.
        self._recent_crossing_positions = []
        self.SPATIAL_COOLDOWN_PX  = 130   # pixels — cars within this radius are suppressed
        self.TEMPORAL_COOLDOWN_S  = 1.8   # seconds — cooldown window after a crossing

        self.counts = {
            "total": 0,
            "car": 0, "truck": 0, "bus": 0,
            "motorcycle": 0, "other": 0,
            "top_to_bottom": 0, "bottom_to_top": 0,
        }

    def update_line(self, line_coords, target_direction="top_to_bottom"):
        self.line_start = tuple(line_coords[0])
        self.line_end   = tuple(line_coords[1])
        self.target_direction = target_direction

    def reset(self):
        self.counted_ids.clear()
        self.previous_centroids.clear()
        self.events.clear()
        self.alert_queue.clear()
        self._recent_crossing_positions.clear()
        for k in self.counts:
            self.counts[k] = 0

    def _is_spatiotemporal_duplicate(self, ts, cx, cy):
        """
        Returns True if a crossing at (cx, cy) at time ts is too close in
        space AND time to a previous crossing — indicating an ID-switched
        re-detection of the same physical vehicle.
        """
        # Prune stale entries first
        self._recent_crossing_positions = [
            (t, x, y) for (t, x, y) in self._recent_crossing_positions
            if ts - t < self.TEMPORAL_COOLDOWN_S
        ]
        for (prev_ts, prev_cx, prev_cy) in self._recent_crossing_positions:
            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            if dist < self.SPATIAL_COOLDOWN_PX:
                return True   # same car, new ID — suppress
        return False

    def process_tracks(self, tracked_objects, video_name="cars.mp4", current_timestamp=0.0):
        """
        tracked_objects: list/array of [x1, y1, x2, y2, track_id, cls_name]
        video_name: name of current processing video
        current_timestamp: elapsed video time in seconds
        Returns list of new crossing events generated this frame.
        """
        new_events = []

        for obj in tracked_objects:
            x1, y1, x2, y2 = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
            track_id = int(obj[4])
            cls_name = str(obj[5]).lower() if len(obj) > 5 else "car"

            # Normalise COCO labels to standard vehicle types
            if "car" in cls_name:
                cls_type = "car"
            elif "truck" in cls_name:
                cls_type = "truck"
            elif "bus" in cls_name:
                cls_type = "bus"
            elif "motorcycle" in cls_name or "bike" in cls_name:
                cls_type = "motorcycle"
            else:
                cls_type = "car"

            curr_centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            if track_id in self.previous_centroids:
                prev_centroid = self.previous_centroids[track_id]

                # Check if centroid trajectory crosses the virtual line
                if intersect(prev_centroid, curr_centroid, self.line_start, self.line_end):
                    if track_id not in self.counted_ids:
                        direction = get_crossing_direction(
                            prev_centroid, curr_centroid, self.line_start, self.line_end
                        )

                        # ── Spatial+temporal duplicate guard ──
                        # Catches the case where SORT dropped the track ID and the
                        # same physical car was re-assigned a new ID a few frames later.
                        if self._is_spatiotemporal_duplicate(current_timestamp,
                                                             curr_centroid[0], curr_centroid[1]):
                            # Silently absorb this ID into counted_ids so it won't
                            # fire again, but don't increment the counter.
                            self.counted_ids.add(track_id)
                            self.previous_centroids[track_id] = curr_centroid
                            continue

                        self.counted_ids.add(track_id)
                        self._recent_crossing_positions.append(
                            (current_timestamp, curr_centroid[0], curr_centroid[1])
                        )

                        # Increment stats
                        self.counts["total"] += 1
                        if cls_type in self.counts:
                            self.counts[cls_type] += 1
                        else:
                            self.counts["other"] += 1
                        if direction in self.counts:
                            self.counts[direction] += 1

                        event_record = {
                            "video_name":        video_name,
                            "timestamp_seconds": round(current_timestamp, 2),
                            "event_type":        "line_cross",
                            "vehicle_class":     cls_type,
                            "track_id":          track_id,
                            "direction":         direction,
                            "wall_time":         time.strftime("%H:%M:%S"),
                        }
                        self.events.append(event_record)
                        new_events.append(event_record)
                        self.alert_queue.append(event_record)

            self.previous_centroids[track_id] = curr_centroid

        return new_events

    def pop_alerts(self):
        """Return and clear pending alert queue (for SSE streaming)."""
        alerts = list(self.alert_queue)
        self.alert_queue.clear()
        return alerts

    def get_csv_data(self):
        """Generates CSV content string matching sample_output_format.csv."""
        lines = ["video_name,timestamp_seconds,event_type,vehicle_class,track_id,direction"]
        for ev in self.events:
            lines.append(
                f"{ev['video_name']},{ev['timestamp_seconds']:.2f},"
                f"{ev['event_type']},{ev['vehicle_class']},{ev['track_id']},{ev['direction']}"
            )
        return "\n".join(lines)

    def get_json_summary(self, duration_sec=0.0):
        """Returns a JSON-serialisable summary dict."""
        return {
            "total_crossings": self.counts["total"],
            "by_class": {
                "car": self.counts["car"],
                "truck": self.counts["truck"],
                "bus": self.counts["bus"],
                "motorcycle": self.counts["motorcycle"],
                "other": self.counts["other"],
            },
            "by_direction": {
                "top_to_bottom": self.counts["top_to_bottom"],
                "bottom_to_top": self.counts["bottom_to_top"],
            },
            "video_duration_sec": round(duration_sec, 2),
            "counting_line": {
                "start": list(self.line_start),
                "end": list(self.line_end),
                "direction": self.target_direction,
            },
            "events": self.events,
        }
