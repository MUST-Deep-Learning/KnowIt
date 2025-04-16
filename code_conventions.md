We have a few in-house conventions we try to stick to when developing KnowIt.
Updating and enforcing these conventions is an ongoing task.

# Layout

> Check check

> `KnowIt.knowit.KnowIt` is the top-level module that calls and runs all lower-level operations.
> It is assumed that the user only interacts directly with the toolkit through this module.

> Below `KnowIt.knowit.KnowIt` are the three main modules `KnowIt.data`, `KnowIt.trainer`, and `KnowIt.interpret` 
> along with sub- and side classes and methods in between.

# Logging

> We use a global `logger` variable set up in `KnowIt.knowit.KnowIt`. All other scripts can create 
> a logger variable and log messages at three levels, as depicted below.
> 
> from helpers.logger import get_logger\
> logger = get_logger()
> 
> logger.info('Here is some info to log.')\
> logger.warning('Here is some important info to log.')\
> logger.error('Here is something wrong. I will describe it and then abort.')\
> exit(101)


# Importing
> Do not import anything that is not used by the script in which it is imported.

> When importing methods or modules (either internal or external to KnowIt), 
> always use the `from x import y, z` format and then use `y` and `z` in your code, 
> do not `import x` and then use `x.y` and `x.z`.

> All imports happen at the top of a script, not in between the main body of code.

# Classes

> Use classes and inheritance only where sensible (e.g. to contain re-usable objects that are meant to be maintained 
> and passed around the code). Methods are fine if you only want to prevent code duplication.

> If a method in a class can be static, make it a static method with `@staticmethod` decorator.

# Annotating

> For all method definitions, annotate the input and output argument types.

> Use kwargs sparingly. We use kwargs to pass user arguments and default values at the top levels. 
> Lower level methods should use named arguments, with as few default values as possible.

> Usually if a variable has the value `None` it means that it is not provided (if it is an argument) or
> not created yet (if it is an attribute).

# Documentation

> All methods must have docstring in [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

> If a script contains a sizable set of functions and/or classes, provide an overview about what it includes 
> and how the parts fit together in the header of the script.

> Don't add docstrings inside the `__init__()` class constructor. Rather include it in 
> the class definition right above it.

> If you want a particular class or method to be included in the auto-generated 
> API reference docs, include it in the local (same directory as the current .py script) 
> `__init__.py` file.

> Every script has the following top-matter:
> 
>    - """ Overview of contents... """
>    - \_\_copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
>    - \_\_licence__ = 'Apache 2.0; see LICENSE file for details.'
>    - \_\_author__ = 'tiantheunissen@gmail.com, randlerabe@gmail.com'
>    - \_\_description__ = 'One-liner of the contents.'
>    - \# external imports
>    - ...
>    - \# internal imports
>    - ...
>    - logger = get_logger()

# Naming conventions

> Use descriptive names.

> Script names are always written in lower-case letters seperated by `_`.

> Class names are written in camel case without spaces between words.

> Method names are written the same as scripts.

> Use a single `_` in front of a method name to indicate that it is not meant to be used 
> outside the current local context (within current class or script).

# Helper functions

Some functions have the potential to be useful throughout the codebase. We keep these in 
specific scripts, as defined below.

> When specific file- or directory paths are to be constructed, 
> define it in `env.env_paths` and import it from there.

> Ad-hoc methods that load, dump, or convert stored files or directories should be defined in 
> `helpers.file_dir_procs` and imported from there.

> As far as possible we try to only define default arguments in `setup.setup_action_args`.
