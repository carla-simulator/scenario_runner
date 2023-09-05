class ImportMsg:
    def __init__(self):
        self.files = []
        self.index = [0]

    def add(self, file_name, line):
        self.files.append(file_name)
        self.index.append(line + self.index[-1])

    # Enter the old line and return the file and the new line
    def get_msg(self, line):
        for i in range(len(self.index)):
            if line <= self.index[i]:
                return self.files[i - 1], line - self.index[i - 1]
        print("relocation failed!")
        return

    # Get the stored information
    def msg(self):
        msg = []
        for i in range(len(self.index) - 1):
            print(self.files[i] + " : " + str(self.index[i]))

    # Empty stored information for running multiple files at once
    def clear_msg(self):
        self.files = []
        self.index = [0]


create_ImportMsg = ImportMsg()
