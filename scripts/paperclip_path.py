"""Five-segment paperclip reference. Run this file to preview its geometry."""

import math

import numpy as np

from models import wrap


class PaperclipReference:
    """Straight, large left semicircle, straight, small left semicircle, straight.

    World x/y are metres; headings are counterclockwise radians.
    Geometry stays here so the preview and robot use the same definition.
    """

    def __init__(self, speed):
        if speed <= 0:
            raise ValueError("Reference speed must be positive.")

        self.speed = speed

        # Initial experiment dimensions [m]. Edit these to change the path.
        self.first_straight_length = 4
        self.large_radius = 1.5
        self.second_straight_length = 4.0
        self.small_radius = 0.5
        self.final_straight_length = 3.0

        # Start at the previous assignment's nominal pose, facing world +x.
        self.start_position = np.array([0.0, -2.0])
        self.start_heading = 0.0

        # Geometric arrival tolerance [m] and reference braking rate [m/s^2].
        # These are initial choices to validate with the real tracking response.
        self.arrival_tolerance = 0.05
        self.deceleration = 0.1

        # Each pair is (length along segment, radius); radius zero means line.
        definitions = [
            (self.first_straight_length, 0.0),
            (math.pi * self.large_radius, self.large_radius),
            (self.second_straight_length, 0.0),
            (math.pi * self.small_radius, self.small_radius),
            (self.final_straight_length, 0.0),
        ]

        self.segments = []
        position = self.start_position.copy()
        heading = self.start_heading
        progress = 0.0

        for length, radius in definitions:
            segment = {
                "start": position.copy(),
                "heading": heading,
                "length": length,
                "radius": radius,
                "start_progress": progress,
            }
            self.segments.append(segment)

            # The next segment starts exactly where this one ends, with the
            # same tangent heading. Curvature still steps at line/arc joins.
            position, heading = self._segment_point(segment, length)
            progress += length

        self.total_length = progress
        self.end_position = position
        self.reset()

    def reset(self):
        """Restart segment progression when beginning a new experiment."""
        self.segment_index = 0
        self.finished = False

    def _segment_point(self, segment, distance):
        """Return position and tangent at distance along one segment."""
        heading = segment["heading"]
        forward = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-math.sin(heading), math.cos(heading)])
        radius = segment["radius"]

        if radius == 0:
            position = segment["start"] + distance * forward
            return position, heading

        # Same circle geometry as Assignment 3, expressed relative to each
        # segment's start heading so the two turns can have different radii.
        turn_angle = distance / radius
        position = segment["start"] + radius * (
            math.sin(turn_angle) * forward
            + (1 - math.cos(turn_angle)) * left
        )
        return position, heading + turn_angle

    def _reference(self, segment, distance):
        """Build the shared reference dictionary from the path geometry."""
        position, heading = self._segment_point(segment, distance)
        radius = segment["radius"]

        return {
            "x": float(position[0]),
            "y": float(position[1]),
            "heading": float(wrap(heading)),
            "curvature": 0.0 if radius == 0 else 1 / radius,
            "progress": segment["start_progress"] + distance,
            "speed": self.speed,
            "stop": False,
        }

    def reference_at(self, progress):
        """Return geometry at distance along the path, without changing state.

        Used by the plot and by geometric checks. Runtime update() additionally
        selects a segment from estimated pose and requests final deceleration.
        """
        progress = float(np.clip(progress, 0.0, self.total_length))

        for segment in self.segments:
            segment_end = segment["start_progress"] + segment["length"]
            if progress < segment_end:
                return self._reference(
                    segment, progress - segment["start_progress"]
                )

        reference = self._reference(self.segments[-1], self.segments[-1]["length"])
        reference["speed"] = 0.0
        reference["stop"] = True
        return reference

    def _project(self, segment, position):
        """Project onto the active line or semicircle, returning distance [m]."""
        heading = segment["heading"]
        forward = np.array([math.cos(heading), math.sin(heading)])
        left = np.array([-math.sin(heading), math.cos(heading)])
        offset = position - segment["start"]
        radius = segment["radius"]

        if radius == 0:
            distance = float(offset @ forward)
        else:
            # Centre is one radius left of the starting tangent. atan2 gives
            # angle from the start radius, increasing through this left turn.
            from_centre = offset - radius * left
            turn_angle = math.atan2(from_centre @ forward, -from_centre @ left)

            # Near the end, atan2 may wrap +pi to -pi after a small overshoot.
            # Near the start, small negative angles should instead clamp to 0.
            if turn_angle < -math.pi / 2:
                turn_angle += 2 * math.pi

            distance = radius * turn_angle

        return float(np.clip(distance, 0.0, segment["length"]))

    def update(self, pose):
        """Return the active reference using estimated (x, y, heading).

        Advance in order when close to each join, avoiding jumps to a nearby
        parallel straight. Call reset() before replaying another experiment.
        """
        position = np.asarray(pose[:2], dtype=float)

        segment = self.segments[self.segment_index]
        endpoint, _ = self._segment_point(segment, segment["length"])

        if np.linalg.norm(position - endpoint) <= self.arrival_tolerance:
            if self.segment_index < len(self.segments) - 1:
                self.segment_index += 1
                segment = self.segments[self.segment_index]
            else:
                self.finished = True

        distance = self._project(segment, position)
        reference = self._reference(segment, distance)

        if self.segment_index == len(self.segments) - 1:
            # v^2 = 2*a*d gives a speed envelope approaching the final stop.
            # Actual stopping still depends on the tracker and wheel response.
            remaining = segment["length"] - distance
            reference["speed"] = min(
                self.speed, math.sqrt(2 * self.deceleration * remaining)
            )

        if self.finished:
            reference = self.reference_at(self.total_length)

        return reference


def main():
    """Plot points sampled from exactly the geometry used by update()."""
    import matplotlib.pyplot as plt

    # Speed does not affect plotted geometry; this is only a preview value.
    path = PaperclipReference(speed=0.2)
    figure, axis = plt.subplots(figsize=(10, 5))

    labels = [
        "First straight",
        "Large-radius turn",
        "Second straight",
        "Small-radius turn",
        "Final straight",
    ]

    for segment, label in zip(path.segments, labels):
        # Sample roughly every 0.1 m. Include each join explicitly.
        count = max(2, math.ceil(segment["length"] / 0.1) + 1)
        distances = np.linspace(0.0, segment["length"], count)
        points = np.array([
            path._segment_point(segment, distance)[0]
            for distance in distances
        ])
        axis.plot(points[:, 0], points[:, 1], ".-", markersize=3, label=label)

        # Direction arrow at the middle of each segment.
        position, heading = path._segment_point(segment, segment["length"] / 2)
        axis.arrow(
            *position,
            0.35 * math.cos(heading),
            0.35 * math.sin(heading),
            width=0.015,
            head_width=0.15,
            color="black",
            length_includes_head=True,
        )

    axis.scatter(*path.start_position, color="green", s=70, label="Start", zorder=5)
    axis.scatter(*path.end_position, color="red", marker="X", s=90,
                 label="Stop", zorder=5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("World x [m]")
    axis.set_ylabel("World y [m]")
    axis.set_title("Paperclip reference: three straights and two different turns")
    axis.grid(True)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
