import os
import subprocess
import time
from threading import Thread, Event


class DaemonTensorboard(Thread):
    def __init__(self, log_dir, port):
        super().__init__()
        self.validate_requirements(log_dir)
        self.log_dir = log_dir
        self.port = port
        self.event = Event()
        self.daemon = True

    @staticmethod
    def _cmd_exists(cmd):
        return any(
            os.access(os.path.join(path, cmd), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )

    @staticmethod
    def validate_requirements(log_dir):
        assert DaemonTensorboard._cmd_exists('tensorboard'), 'TensorBoard not found'
        os.makedirs(log_dir, exist_ok=True)

    @staticmethod
    def kill_old():
        try:
            subprocess.check_output(['killall', 'tensorboard'], stderr=subprocess.DEVNULL)
            print('Killed some stale Tensorboard process before running a managed daemon')
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    def create_tensorboard_process(self):
        self.kill_old()
        pid = subprocess.Popen(
            ["tensorboard", "--logdir", self.log_dir, "--host", "localhost", "--port", str(self.port)],
            stdout=open(os.path.join(self.log_dir, 'tensorboard_server.log'), 'a'),
            stderr=subprocess.STDOUT,
        )
        time.sleep(5)
        assert pid.poll() is None, 'TensorBoard launch failed (port occupied?)'
        return pid

    def run(self):
        pid = self.create_tensorboard_process()
        print(f'Running TensorBoard daemon on port {self.port}')

        while not self.event.is_set():
            time.sleep(1)
            assert pid.poll() is None, 'TensorBoard was killed'

        print('Stopping TensorBoard daemon')
        pid.terminate()
        pid.communicate()

    def stop(self):
        self.event.set()
        self.join()
