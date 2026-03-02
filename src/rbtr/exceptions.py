"""rbtr exception hierarchy."""


class RbtrError(Exception):
    """Base error for rbtr. Caught at the top level and printed to the console."""


class TaskCancelled(Exception):
    """Raised inside a task thread when the user requests cancellation."""
