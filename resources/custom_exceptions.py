import os.path


class InvalidPathException(Exception):
    """Exception raised when a path is not valid.

    Attributes:
        message: description of the issue, including problematic path.
    """

    def __init__(self, expected: str, given: str, detail: str = None):
        """Overwrite initialization method.

        Exception message will be in the form of "Invalid path: expected, given: reason."

        Args:
            expected: description of expected path.
            given: invalid data.
            detail: additional details. Optional.
        """
        if detail is None:
            self.message = f"Invalid path: expected {expected}, given {given}."
        else:
            self.message = f"Invalid path: expected {expected}, given {given}: {detail}."
        super().__init__(self.message)


class UnwritablePathException(InvalidPathException):
    """Exception raised when a given path should be writeable but is not.

    Attributes:
        message: description of the missing path
    """

    def __init__(self, expected: str, given: str):
        """Overwrite initialization method.

        Exception message will be in the form of "Invalid path: expected, given: reason."

        Args:
            expected: description of expected path.
            given: given data.
        """
        super().__init__(expected, given, "given path should be writeable")


class MissingPathException(InvalidPathException):
    """Exception raised when a given path should be pre-existing but is not.

    Attributes:
        message: description of the missing path
    """

    def __init__(self, expected: str, detail: str = None):
        """Overwrite initialization method.

        Exception message will be in the form of "Invalid path: expected, given none: reason."

        Args:
            expected: description of expected path.
            detail: additional details. Optional.
        """
        super().__init__(expected, "none", detail)


class WrongExtensionException(InvalidPathException):
    """Exception raised when a given file has the wrong extension.

    Attributes:
        message: description of the missing path
    """

    def __init__(self, expected: str, given: str, expected_ext: str, given_ext: str = None):
        """Overwrite initialization method.

        Exception message will be in the form of "Invalid path: expected, given: reason."

        Args:
            expected: description of expected path.
            given: invalid data.
            expected_ext: expected extension.
            given_ext: given extension. Optional
        """
        if given_ext is None:
            given_ext = os.path.basename(given).split(".")[1]
        super().__init__(expected, given, f"given file is of extension \"{given_ext}\", should be \"{expected_ext}\"")


# Usage example:
if __name__ == "__main__":
    print(UnwritablePathException("Excel file", "data.xml").message)
    print(MissingPathException("Excel file", "excel file is missing").message)
    print(WrongExtensionException("Excel file", "data.txt", "xml", "txt").message)
    print(WrongExtensionException("Excel file", "data.txt", "xml").message)


