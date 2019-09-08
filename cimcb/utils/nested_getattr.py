from functools import reduce


def nested_getattr(model, attributes):
    """getattr for nested attributes."""

    def _getattr(model, attributes):
        return getattr(model, attributes)

    return reduce(_getattr, [model] + attributes.split("."))
