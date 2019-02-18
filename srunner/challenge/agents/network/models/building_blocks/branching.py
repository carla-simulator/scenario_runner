
import torch.nn as nn


class Branching(nn.Module):

    def __init__(self, branched_modules=None):
        """

        Args:
            branch_config: A tuple containing number of branches and the output size.
        """
        # TODO: Make an auto naming function for this.

        super(Branching, self).__init__()

        """ ---------------------- BRANCHING MODULE --------------------- """
        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)




    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):
        # get only the speeds from measurement labels



        # TODO: we could easily place this speed outside

        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(x))

        return branches_outputs


