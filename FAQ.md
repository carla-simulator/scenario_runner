# Frequently Asked Questions

## I receive the error "TypeError: 'instancemethod' object has no attribute '__getitem__'" in the agent navigation

This issue is most likely caused by an outdated version of the Python Networkx package. Please remove the current installation
(e.g. sudo apt-get remove python-networkx) and install it using "pip install --user networkx".
