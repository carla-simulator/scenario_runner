"""
    A simple factory module that returns instances of possible modules 

"""

from .models import CoILICRA


def CoILModel(architecture_name, architecture_configuration):
    """ Factory function

        Note: It is defined with the first letter as uppercase even though is a function to contrast
        the actual use of this function that is making classes
    """
    # TODO: this could be extended to some more arbitrary definition

    if architecture_name == 'coil-icra':

        return CoILICRA(architecture_configuration)

    else:

        raise ValueError(" Not found architecture name")