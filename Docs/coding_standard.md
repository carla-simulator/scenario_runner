<h1>Coding standard</h1>

> _This document is a work in progress and might be incomplete._

General
-------

  * Use spaces, not tabs.
  * Avoid adding trailing whitespace as it creates noise in the diffs.
  * Comments should not exceed 120 columns, code may exceed this limit a bit in
    rare occasions if it results in clearer code.

Python
------

  * All code must be Python 3. Python 2 compatibility was dropped, and so were
    the `from __future__` imports that went with it.
  * [Pylint][pylintlink] should not give any error or warning (few exceptions
    apply with external classes like `numpy`, see our `.pylintrc`).
  * Python code follows [PEP8 style guide][pep8link] (use `autopep8` whenever
    possible).

[pylintlink]: https://www.pylint.org/
[pep8link]: https://www.python.org/dev/peps/pep-0008/

ScenarioRunner is pure Python; the C++ guidelines of the CARLA simulator do not
apply here.
