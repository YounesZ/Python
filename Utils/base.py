#!/usr/bin/env python
"""Several Utilities.

"""

class hashable_dict(dict):
    """A dictionary that is hashable."""

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

