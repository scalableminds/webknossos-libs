def get_git_version() -> str:
    import git

    git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    return git_hash


def get_package_version() -> str:
    # pylint: disable=no-name-in-module,import-error
    from wkcuber.version import __version__

    return __version__


def get_available_version() -> str:
    """
    If the current code lies in the published package, a version.py file will exist which was created at
    publish-time. That version info is returned here.
    Otherwise, the version is derived using the git repository in which the code lives.
    """

    try:
        return get_package_version()
    except ModuleNotFoundError:
        return get_git_version()
