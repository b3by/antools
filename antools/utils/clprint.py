class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def bold(msg):
    """Bold version of the message
    """
    return bcolors.BOLD + msg + bcolors.ENDC


def info(msg):
    """Info version (blue) of the message
    """
    return bcolors.OKBLUE + msg + bcolors.ENDC


def warn(msg):
    """Warning version (orange) of the message
    """
    return bcolors.WARNING + msg + bcolors.ENDC


def err(msg):
    """Error version (red) of the message
    """
    return bcolors.FAIL + msg + bcolors.ENDC


def ok(msg):
    """Ok version (green) of the message
    """
    return bcolors.OKGREEN + msg + bcolors.ENDC


def ptinfo(msg):
    """Print the message next to an bold, info tilde
    """
    op = bold('[~]')
    print(op, msg)


def ptrun(msg):
    """Print the message next to a bald, warning tilde
    """
    op = bold(warn('[~]'))
    print(op, msg)


def ptok(msg):
    """Print the message next to a bald, ok tilde
    """
    op = bold(ok('[+]'))
    print(op, msg)
