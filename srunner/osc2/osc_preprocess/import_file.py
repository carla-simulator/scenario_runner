"""
The referenced file class
__base_path: file path
__file: The file itself
"""
import queue


class ImportFile:
    def __init__(self, base_path):
        self.__base_path = base_path
        self.__file = open(base_path, encoding="utf-8")

    # Get current path
    def get_path(self):
        return self.__base_path

    # Verify that it is the same file by comparing whether the paths are the same
    def same_as(self, another_file):
        if self.__base_path == another_file.get_path():
            return True
        return False

    def get_true_path(self, import_file_path):
        """
        Pass in the import relative path, converted to the absolute path of the reference file
        Args:
            import_file_path: Relative path of the referenced file
        Returns:

        """
        current_path = self.get_path()
        import_file_path = import_file_path.rsplit("../")
        up_levels = len(import_file_path)
        true_path = current_path.rsplit("/", up_levels)[0] + "/" + import_file_path[-1]
        return true_path

    # Get import members and return a queue of members
    def get_import_members(self):
        file = open(self.get_path(), encoding="utf-8")
        is_continue = True
        import_files = queue.Queue()
        while is_continue:
            current_line = file.readline().strip()
            # comment line
            if current_line.startswith("#"):
                continue
            # blank line
            elif not len(current_line):
                continue
            # import line
            elif current_line.startswith("import"):
                # Gets the relative path to the referenced file
                current_line = current_line.lstrip("import")
                current_line = current_line.lstrip(" ")
                # The absolute path to the file converted to Import
                import_file_path = self.get_true_path(current_line)
                import_files.put(ImportFile(import_file_path))
            # Other lines, exit
            else:
                is_continue = False
                file.close()
        return import_files

    # Gets the content of the import file and returns the number of lines
    def get_content(self):
        lines = -1  # count the number of rows
        content = ""
        file = open(self.get_path(), encoding="utf-8")
        current_line = "start"
        pre_line = ""
        while current_line:
            lines += 1
            pre_line = current_line
            current_line = file.readline()
            # In order to locate errors, leave the original file line number unchanged
            if current_line.startswith("import"):
                content += "#" + current_line
                # content += current_line
                continue
            else:
                content += current_line
        if pre_line.endswith("\n"):
            return content, lines
        else:
            index = len(pre_line)
            print(
                "[Error] file '"
                + self.get_path()
                + "' line "
                + lines.__str__()
                + ":"
                + index.__str__()
                + " mismatched input '<EOF>'"
            )
            return content, lines
