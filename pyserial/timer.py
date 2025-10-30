""" timer
"""

import typing
import threading


class Timer(object):
    """ Timer
    """
    def __init__(self,
                 interval: float,
                 function: typing.Callable,
                 *args: typing.Any,
                 **kwargs: typing.Any) \
            -> None:
        self._interval = interval
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._thread = threading.Thread(target=self._loop)
        self._stopevent = threading.Event()

    def __del__(self) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stopevent.is_set():
            self._function(*self._args, **self._kwargs)
            self._stopevent.wait(self._interval)

    def start(self) -> None:
        self._stopevent.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stopevent.set()

    def is_running(self) -> bool:
        return self._thread.is_alive()

