# import typing
# from py_trees import behaviour, composites, common
#
# # add by zyy
# class Repeat(behaviour.Behaviour):
#     def __init__(self, name: str="Repeat", count: int=1, children=None):
#         super().__init__(name)
#         self.period = count
#         self.count = 0
#         self.child = []
#         # if children is None:
#         #     self.children = []
#         #     self.child = None
#         # else:
#             # self.children.append(composites.Sequence("Sequence", children=children))
#             # self.child = self.children[0]
#         if children is not None:
#             for child in children:
#                 self.add_child(child)
#         else:
#             self.children = []
#
#     def add_child(self, child):
#         """
#         Adds a child.
#
#         Args:
#             child (:class:`~py_trees.behaviour.Behaviour`): child to add
#
#         Returns:
#             uuid.UUID: unique id of the child
#         """
#         assert isinstance(child, behaviour.Behaviour), "children must be behaviours, but you passed in %s" % type(child)
#         self.child.append(child)
#         child.parent = self
#         return child.id
#
#     def update(self):
#         self.logger.debug("%s.update()" % self.__class__.__name__)
#         # for child in self.children:
#         #     child.tick_once()
#         self.child.tick_once()
#
#         if self.child.status == common.Status.FAILURE or self.child.status == common.Status.SUCCESS:
#             self.count += 1
#
#         if self.count < self.period:
#             return common.Status.RUNNING
#         else:
#             return common.Status.SUCCESS
from py_trees import behaviour, common


class Decorator(behaviour.Behaviour):
    """
    Parent class for decorating a child/subtree with some additional logic.
    Args:
        child: the child to be decorated
        name: the decorator name
    Raises:
        TypeError: if the child is not an instance of :class:`~py_trees.behaviour.Behaviour`
    """

    def __init__(
            self,
            child: behaviour.Behaviour,
            name
    ):
        # Checks
        if not isinstance(child, behaviour.Behaviour):
            raise TypeError("A decorator's child must be an instance of py_trees.behaviours.Behaviour")
        # Initialise
        super().__init__(name=name)
        self.children.append(child)
        # Give a convenient alias
        self.decorated = self.children[0]
        self.decorated.parent = self

    def tick(self):
        """
        Manage the decorated child through the tick.
        Yields:
            :class:`~py_trees.behaviour.Behaviour`: a reference to itself or one of its children
        """
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # initialise just like other behaviours/composites
        if self.status != common.Status.RUNNING:
            self.initialise()
        # interrupt proceedings and process the child node
        # (including any children it may have as well)
        for node in self.decorated.tick():
            yield node
        # resume normal proceedings for a Behaviour's tick
        new_status = self.update()
        if new_status not in list(common.Status):
            self.logger.error(
                "A behaviour returned an invalid status, setting to INVALID [%s][%s]" % (new_status, self.name)
            )
            new_status = common.Status.INVALID
        if new_status != common.Status.RUNNING:
            self.stop(new_status)
        self.status = new_status
        yield self

    def stop(self, new_status):
        """
        Check if the child is running (dangling) and stop it if that is the case.
        Args:
            new_status (:class:`~py_trees.common.Status`): the behaviour is transitioning to this new status
        """
        self.logger.debug("%s.stop(%s)" % (self.__class__.__name__, new_status))
        self.terminate(new_status)
        # priority interrupt handling
        if new_status == common.Status.INVALID:
            self.decorated.stop(new_status)
        # if the decorator returns SUCCESS/FAILURE and should stop the child
        if self.decorated.status == common.Status.RUNNING:
            self.decorated.stop(common.Status.INVALID)
        self.status = new_status

    def tip(self):
        """
        Retrieve the *tip* of this behaviour's subtree (if it has one).
        This corresponds to the the deepest node that was running before the
        subtree traversal reversed direction and headed back to this node.
        Returns:
            :class:`~py_trees.behaviour.Behaviour` or :obj:`None`: child behaviour,
                itself or :obj:`None` if its status is :data:`~py_trees.common.Status.INVALID`
        """
        if self.decorated.status != common.Status.INVALID:
            return self.decorated.tip()
        else:
            return super().tip()



class SuccessIsRunning(Decorator):
    """
    It never ends...
    """
    def __init__(self, child, name, count):
        super().__init__(child, name=name)
        self.period = int(float(count))
        self.repaet_num = 0

    def update(self):
        """
        Reflect :data:`~py_trees.common.Status.SUCCESS` as :data:`~py_trees.common.Status.RUNNING`.
        Returns:
            :class:`~py_trees.common.Status`: the behaviour's new status :class:`~py_trees.common.Status`
        """
        if self.decorated.status == common.Status.SUCCESS:
            self.feedback_message = "success is running [%s]" % self.decorated.feedback_message
            if self.repaet_num < self.period:
                return common.Status.SUCCESS
            self.repaet_num += 1
            self.initialise()
            return common.Status.RUNNING
        self.feedback_message = self.decorated.feedback_message
        return self.decorated.status