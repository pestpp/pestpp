# a test marked own_job gets its own ci entry, so it is dropped when its file is run as a whole
# and only runs when it is named on the command line (basic_tests.py::basic_xplat_g07_test).
# this is what __test__ = False did under nose; pytest wont run a __test__ = False function
# even when it is named, so it cant be used for this


def pytest_collection_modifyitems(config, items):
    named = set(a.split("::", 1)[1] for a in config.args if "::" in a)
    keep, drop = [], []
    for item in items:
        if item.get_closest_marker("own_job") is not None and item.name not in named:
            drop.append(item)
        else:
            keep.append(item)
    if len(drop) > 0:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
