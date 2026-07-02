from typing import Any

class Metadata(dict[str, Any]):
    """Dictionary-like container for data that is not critical to data reduction"""

    def __repr__(self) -> str:
        return f"Metadata({super().__repr__()})"
