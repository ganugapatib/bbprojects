import os


class ReadFiles:
    def __init__(self):
        pass

    def readfile(self):
        pass


if __name__ == "__main__":
    cwdpath = os.getcwd()
    datafilespath = os.path.join(cwdpath, "bbprojects\\copilotbot\\app\data\\")
    # print("CWD PATH{}", cwdpath)
    # print("DATA FILE PATH {}", datafilespath)

    files_read = os.listdir(datafilespath)
    print(
        "files to read from path {} \n files that are available in path {}",
        datafilespath,
        files_read,
    )
    rf = ReadFiles()
