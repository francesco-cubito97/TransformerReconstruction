import os

class Logger():
    def __init__(self, name, log_dir):
        self.name = name
        self.log_dir = log_dir
        self.logging_file = "{}logs.txt".format(log_dir)
        #self.resetLog()

    def resetLog(self):
        # Reset logging file
        if os.path.exists(self.logging_file):
            os.remove(self.logging_file)
            file = open(self.logging_file, "w")
            file.close()

    def createLog(self, log_id, new_log):
        # Compose log
        log = "{} {}".format(log_id, new_log)
        
        # Open logging file
        file = open(self.logging_file, "a")
        
        # Inseriting the new log in a new line
        file.write(log + "\n")

        # Close the logging file to save
        file.close()

        # Print in the command the log
        print(log)

        
    