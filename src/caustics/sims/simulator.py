from typing import Annotated, Optional

__all__ = ("NameType",)

NameType = Annotated[Optional[str], "Name of the simulator"]
