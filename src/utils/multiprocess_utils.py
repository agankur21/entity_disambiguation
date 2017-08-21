import os
import threading
from subprocess import Popen, PIPE

from src.utils.decorator_utils import *


class ProcessRunner:
    def __init__(self, command, directory):
        self.command = command
        if not os.path.isdir(directory):
            raise IOError("No such directory : %s" % directory)
        self.directory = directory
        self.result = None
        self.error = None
        self.status = -1
        self.process = None
        self.thread = threading.Thread(target=self.run_process)

    def start(self, timeout):
        self.thread.start()
        self.thread.join(timeout)
        if not self.terminate():
            print "------------Process completed safely--------------"
        else:
            print "------------Process terminated in between---------"

    def terminate(self):
        if self.thread.is_alive():
            print 'Terminating thread'
            self.process.terminate()
            self.thread.join()
        return self.status

    @timeit
    @print_exception
    def run_process(self):
        print '-' * 40
        print 'Running: %r in directory %r' % (self.command, self.directory)
        print "Thread started"
        self.process = Popen(self.command, stderr=PIPE, stdout=PIPE, cwd=self.directory)
        self.result, self.error = self.process.communicate()
        self.status = self.process.returncode
        print "Thread finished"

