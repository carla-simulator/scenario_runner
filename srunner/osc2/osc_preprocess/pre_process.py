# Preprocess the osc file to expand the import
import os

from srunner.osc2.osc_preprocess.import_file import ImportFile

# File preprocessor class
from srunner.osc2.osc_preprocess.import_msg import create_ImportMsg as import_msg
from srunner.osc2.utils.log_manager import *


class Preprocess:
    def __init__(self, current_path):
        self.import_msg = import_msg
        # The path to the current file, converted to an absolute path
        self.current_path = os.getcwd() + "/" + current_path
        # invocation stack
        self.stack = []
        # invocation record
        self.note = []
        # final documents
        self.result = "result"
        self.file = None

    # Determine whether it is recorded
    def exit(self, current, note):
        for member in note:
            if current.same_as(member):
                return True
        return False

    # Return import preprocessing results and import information
    def import_process(self):
        self.file = open(self.result, "w+", encoding="utf-8")
        current = ImportFile(self.current_path)
        self.__import_process(current)
        self.file.close()
        return self.result, self.import_msg

    def __import_process(self, current):
        # Record the current node to the call stack
        self.stack.append(current)
        # Get the child node and store it in the queue
        child_queue = current.get_import_members()

        # We recursively deal with children
        while not child_queue.empty():
            child = child_queue.get()
            # If the child node is already contained in the stack, it is a circular reference
            if self.exit(child, self.stack):
                child.get_path()
                msg = "[Error] circular import file " + child.get_path()
                LOG_ERROR(msg)
                return
            # If the child node appears in note, it is a duplicate reference
            if self.exit(child, self.note):
                continue
            self.__import_process(child)
        # The child node is processed, and the current node is processed
        # Record
        self.note.append(current)
        # Write content, record import information
        content, line = current.get_content()
        self.file.write(content)
        self.import_msg.add(current.get_path(), line)
