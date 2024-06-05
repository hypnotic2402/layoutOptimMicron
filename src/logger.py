class Logger():
    instance = None
    def __init__(self, filename="log.txt") -> None:
        self.filename = filename
        self.logFile = open(filename , 'w')
        
    @staticmethod
    def getInstance(filename="log.txt"):
        if Logger.instance == None:
            Logger.instance = Logger(filename=filename)
        return Logger.instance

    def log(self, string):
        self.logFile.write(string + "\n")

    def close(self):
        self.logFile.close()