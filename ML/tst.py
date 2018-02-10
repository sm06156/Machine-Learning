from collections import defaultdict
from unittest import TestCase
import unittest

import math



class Point:
    def __str__(self):
        return "<" + self.label + ": " + repr(self.values) + ">"

    def __repr__(self):
        return "<" + self.label + ": " + repr(self.values) + ">"

    def __init__(self, label, values):
        self.label = label
        self.values = values



data = [
    Point('College', [24, 40000]),
    Point('No College', [53, 52000]),
    Point('No College', [23, 25000]),
    Point('College', [25, 77000]),
    Point('College', [32, 48000]),
    Point('College', [52, 110000]),
    Point('College', [22, 38000]),
    Point('No College', [43, 44000]),
    Point('No College', [52, 27000]),
    Point('College', [48, 65000])
]
