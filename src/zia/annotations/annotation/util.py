from enum import IntEnum

class PyramidalLevel(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7

    @classmethod
    def get_by_numeric_level(cls, level: int) -> "PyramidalLevel":
        return PyramidalLevel(level)
