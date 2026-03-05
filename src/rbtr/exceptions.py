"""rbtr exception hierarchy."""


class RbtrError(Exception):
    """Base error for rbtr. Caught at the top level and printed to the console."""


class PortBusyError(RbtrError):
    """Raised when an OAuth callback server cannot bind its port."""


class TaskCancelled(Exception):
    """Raised inside a task thread when the user requests cancellation."""
