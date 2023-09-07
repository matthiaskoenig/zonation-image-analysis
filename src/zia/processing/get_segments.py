from typing import Tuple, List

import numpy as np


class LineSegmentsFinder:
    def __init__(self, pixels: List[Tuple[int, int]], image_shape: Tuple[int, int]):
        self.pixels = pixels
        self.image_shape = image_shape
        self.segments_finished = []
        self.segments_to_do = []
        self.nodes = []

    def get_neighbors(self, pixel: Tuple[int, int]):
        neighbors = []
        h, w = pixel
        ih, iw = self.image_shape
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for dh, dw in directions:
            nh, nw = h + dh, w + dw
            if 0 <= nh < ih and 0 <= nw < iw:
                neighbors.append((nh, nw))
        return neighbors

    def get_connected_pixels(self, neighbors: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n_pixels = [n for n in neighbors if n in self.pixels]
        for n_pixel in n_pixels:
            self.pixels.remove(n_pixel)
        return n_pixels

    def get_end_pixels(self, neighbors: List[Tuple[int, int]], segment: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [n for n in neighbors if (n in self.nodes) ^ (n == segment[0])]

    def walk_segment(self, segment: List[tuple[int, int]]) -> None:
        """
        should take a segment until there is no or multiple connected pixels
        """
        print("walk segment")
        this_pixel = segment[-1]
        neighbors = self.get_neighbors(this_pixel)
        neighbors.remove(segment[-2])
        end_pixels = self.get_end_pixels(neighbors, segment)

        # break when the pixel is connected to the beginning of the segment or a node
        if len(end_pixels) != 0:
            segment.append(end_pixels[0])
            self.segments_finished.append(segment)
            return

            # get connected pixels
        connected_pixels = self.get_connected_pixels(neighbors)

        # if one connected pixel is found -> walk on
        if len(connected_pixels) == 1:
            segment.append(connected_pixels[0])
            self.walk_segment(segment)
            return

        # if no connection found -> branch ending in the void
        if len(connected_pixels) == 0:
            self.segments_finished.append(segment)
            return

        # if multiple connections found -> pixel is node, initialize new segments
        if len(connected_pixels) > 1:
            self.nodes.append(this_pixel)
            self.segments_finished.append(segment)

            for connected_pixel in connected_pixels:
                self.segments_to_do.append([this_pixel, connected_pixel])

            return
        return

    def process_segments(self):
        print("process segments")
        while len(self.segments_to_do) != 0:
            next_segment = self.segments_to_do.pop()
            self.walk_segment(next_segment)

    def run(self) -> None:
        # find a pixel with a neighbor that's not been processed yet
        print("run segmentation")
        while len(self.pixels) > 0:
            pixel = self.pixels.pop()
            neighbors = self.get_neighbors(pixel)

            connected_pixels = self.get_connected_pixels(neighbors)
            print(f"len con pixels = {len(connected_pixels)}")

            if len(connected_pixels) != 0:
                self.segments_to_do.append([pixel, connected_pixels[0]])
                self.process_segments()
