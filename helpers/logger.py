"""Provide multiprocessing-safe logging.

The main function provided by this module is |get_logger|.  This function
provides a logger unique to the calling process.

The |set_logging_format| function can be used to change the global logging
format.

Before these functions can be used in a child process, |setup_logging| must be
called from the child process.  The queue passed to |setup_logging| can be
obtained by calling |get_logging_queue| from the parent process or any other
process that has already called |setup_logging|.

.. |get_logger| replace:: :py:func:`.get_logger`
.. |set_logging_format| replace:: :py:func:`.set_logging_format`
.. |setup_logging| replace:: :py:func:`.setup_logging`
.. |get_logging_queue| replace:: :py:func:`.get_logging_queue`

"""

__description__ = 'Provide multiprocessing-safe logging'
__author__ = 'dgerbrandh@gmail.com'

# external imports
import atexit
import logging
import logging.handlers
import multiprocessing
from queue import Empty as QueueEmpty

_logging_queue = None


def _logging_listener(queue):
    """Await and handle logging events in the given queue.

    Meant to be run in a separate process, this function awaits logging events
    in the given queue and logs them to std error.  The events in the queue
    may come from different processes or threads.

    If a string is found in the queue, it is used as the new format string
    for the logger.

    If ``None`` is found in the queue, the function returns.

    Args:
        queue (multiprocessing.Queue): Queue that will be monitored for events.

    """
    parent_process = multiprocessing.parent_process()
    parent_process_dead = False

    handler_root = logging.StreamHandler()
    handler_root.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    logger_root = logging.getLogger()
    logger_root.setLevel(logging.ERROR)
    logger_root.addHandler(handler_root)

    def handle_item(item):
        if isinstance(item, str):
            handler_root.setFormatter(logging.Formatter(item))
        else:
            handler_root.emit(item)

    while True:
        try:
            item = queue.get(True, 60)
            if item is None:
                break
            else:
                handle_item(item)
        except QueueEmpty:
            if parent_process is not None and not parent_process.is_alive():
                if parent_process_dead:
                    # The parent process has been dead for at least one minute.
                    #   We should have been told to return by now.
                    queue.put_nowait(None)
                else:
                    # The parent process is dead.  Wait another minute for
                    #   items in the queue.
                    parent_process_dead = True
        except (SystemExit, KeyboardInterrupt):
            queue.put_nowait(None)
        except EOFError:
            logger_root.debug('The logging process encountered an EOF error.')
            break
        except Exception as ex:
            logger_root.error(
                'An error occurred while processing log entry: {}.'
                .format(type(ex))
            )


def _start_listener():
    """Start the main logging process if required and possible.

    Start the main logging process if it has not been started and if the
    current process is the main process.

    """
    global _logging_queue

    process = multiprocessing.parent_process()

    if process is None and _logging_queue is None:
        queue = multiprocessing.Queue()

        atexit.register(queue.put_nowait, None)

        process = multiprocessing.Process(
            target=_logging_listener, args=(queue,)
        )
        process.start()

        _logging_queue = queue


def setup_logging(queue):
    """Set up logging in the current process.

    This function must be called in any child process before any of the other
    functions in this module can be used in that process.  It does not need to
    be called by the main process.

    Example:

    .. code-block:: python
        :linenos:

        import multiprocessing
        from mustnet.utils.logger import get_logging_queue, setup_logging

        def child_wrapper(queue, *args, **kwargs):
            setup_logging(queue)
            # `child_process()` is the function to be run in a child process
            return child_process(*args, **kwargs)

        queue = get_logging_queue()
        proc = multiprocessing.Process(
            target=child_wrapper,
            args=(queue, *other_args), kwargs={**other_kwargs}
        )
        proc.start()

    Args:
        queue (multiprocessing.queues.Queue): Logging queue.

    """
    global _logging_queue

    if not isinstance(queue, multiprocessing.queues.Queue):
        raise RuntimeError(
            'The argument `queue` must have type '
            '`multiprocessing.queues.Queue`, not `{}`'.format(type(queue))
        )

    if _logging_queue is None:
        _logging_queue = queue


def get_logging_queue():
    """Get the logging queue.

    The logging queue must be passed to |setup_logging| from any created child
    processes.

    Before this function can be used in a child process, |setup_logging| must
    be called from that child process.

    Returns:
        multiprocessing.Queue: Logging queue.

    """
    _start_listener()

    queue = _logging_queue
    if queue is None:
        raise RuntimeError(
            'The main logging process has not been started, or '
            '`setup_logging()` has not been called in this process'
        )

    return queue


def get_logger():
    """Configure and return a logger for the current process.

    The returned logger is not the root logger, and the root logger should
    **not** be directly used or modified.  The messages logged using the
    returned logger are passed to the main logging process where they are
    logged to std error.

    Before this function can be used in a child process, |setup_logging| must
    be called from that child process.

    The ``propagate`` attribute of the returned logger should remain ``False``.
    Setting this to ``True`` might duplicate log messages.

    The main logging process coordinates messages from different processes, and
    any logging handlers that do not require this coordination can be directly
    attached to the returned logger.  If, for example, log messages from a
    **single** process should also be sent to a file, the handler for the file
    can be directly attached to the returned logger.  If, however, messages
    from **multiple** processes should be sent to the same file, the logging
    structure would have to be modified.

    Returns:
        logging.Logger: Logger for the current process.

    """
    queue = get_logging_queue()

    process = multiprocessing.current_process()

    logger = logging.getLogger('process{}'.format(process.pid))
    logger.propagate = False

    if len(logger.handlers) == 0:
        handler = logging.handlers.QueueHandler(queue)
        logger.addHandler(handler)

    return logger


def set_logging_format(fmt):
    """Change the format string used by the main logger.

    This change affects the messages logged to std error by any child process.

    Before this function can be used in a child process, |setup_logging| must
    be called from that child process.

    To change the format string of any process-local handlers, the methods
    provided by the ``logging`` module can be used (the root logger should
    just not be modified).

    Args:
        fmt (str): New format string to be used.

    """
    queue = get_logging_queue()
    queue.put_nowait(fmt)
