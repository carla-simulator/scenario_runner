class LogMsg:
    def __init__(self):
        self.__msg = []
        self.is_open = False

    def add_log_msg(self, log_msg):
        self.__msg.append(log_msg)

    def get_log_msg(self):
        return self.__msg

    def clean_msg(self):
        self.__msg = []


# Singleton pattern
create_LogMsg = LogMsg()
