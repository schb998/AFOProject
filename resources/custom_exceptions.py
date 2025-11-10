class MissingLoadbearingPathException(Exception):
    """Exception raised when a load bearing path is missing.

    Attributes:
        message: description of the missing path.
    """

    def __init__(self, missing_path):
        self.message = f"Missing path: {missing_path}. Unable to run app."
        super().__init__(self.message)


class InvalidPathException(Exception):
    """Exception raised when a path is not valid.

    Attributes:
        message: description of the issue, including problematic path.
    """

    def __init__(self, expected: str, given: str, reason: str = None):
        if reason is None:
            self.message = f"Invalid path: expected \"{expected}\", given \"{given}\"."
        else:
            self.message = f"Invalid path: expected \"{expected}\", given \"{given}\": {reason}."
        super().__init__(self.message)


class UnwritablePathException(InvalidPathException):
    """Exception raised when a given path should be writeable but is not.

    Attributes:
        message: description of the missing path
    """

    def __init__(self, expected: str, given: str, reason: str = None):
        super().__init__(expected, given, reason)


class MissingPathException(InvalidPathException):
    """Exception raised when a given path should be pre-existing but is not.

    Attributes:
        message: description of the missing path
    """

    def __init__(self, expected: str, given: str, reason: str = None):
        super().__init__(expected, given, reason)


class WrongExtensionException(InvalidPathException):
    """Exception raised when a given file has the wrong extension.

    Attributes:
        message: description of the missing path
    """

    def __init__(self, expected: str, given: str, reason: str = None):
        super().__init__(expected, given, reason)
