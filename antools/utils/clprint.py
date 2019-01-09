import types


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def bold(msg: str) -> str:
    """Bold version of the message
    """
    return bcolors.BOLD + msg + bcolors.ENDC


def info(msg: str) -> str:
    """Info version (blue) of the message
    """
    return bcolors.OKBLUE + msg + bcolors.ENDC


def warn(msg: str) -> str:
    """Warning version (orange) of the message
    """
    return bcolors.WARNING + msg + bcolors.ENDC


def err(msg: str) -> str:
    """Error version (red) of the message
    """
    return bcolors.FAIL + msg + bcolors.ENDC


def ok(msg: str) -> str:
    """Ok version (green) of the message
    """
    return bcolors.OKGREEN + msg + bcolors.ENDC


def ptinfo(msg: str) -> None:
    """Print the message next to an bold, info tilde
    """
    op = bold('[~]')
    print(op, msg)


def ptrun(msg: str) -> None:
    """Print the message next to a bald, warning tilde
    """
    op = bold(warn('[~]'))
    print(op, msg)


def ptok(msg: str) -> None:
    """Print the message next to a bald, ok tilde
    """
    op = bold(ok('[+]'))
    print(op, msg)


def ptflags(flags: types.SimpleNamespace) -> None:
    """Print all the items in the namespace, with alignment
    """
    all_keys = list(vars(flags))
    length = len(max(all_keys, key=len)) + 8
    longest = '%-' + str(length) + 's %s'

    for k, v in enumerate(vars(flags)):
        ptok(longest % (bold(v), getattr(flags, v)))
