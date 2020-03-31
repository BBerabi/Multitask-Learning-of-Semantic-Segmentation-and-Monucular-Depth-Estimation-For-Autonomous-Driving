import os
import subprocess
import time
from threading import Thread, Event


class DaemonNgrok(Thread):
    def __init__(self, auth_token, port):
        super().__init__()
        self.validate_requirements()
        self.kill_old()
        self.register_authtoken(auth_token)
        self.port = port
        self.event = Event()
        self.daemon = True

    @staticmethod
    def validate_requirements():
        assert os.access(os.path.join(os.path.dirname(__file__), '..', '..', 'ngrok'), os.X_OK)

    @staticmethod
    def kill_old():
        try:
            subprocess.check_output(['killall', 'ngrok'], stderr=subprocess.DEVNULL)
            print('Killed some stale ngrok process before running a managed daemon')
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    @staticmethod
    def register_authtoken(authtoken):
        subprocess.check_output(
            ["./ngrok", "authtoken", authtoken],
            stderr=subprocess.DEVNULL,
            cwd=os.path.join(os.path.dirname(__file__), '..', '..')
        )

    def create_tensorboard_process(self):
        pid = subprocess.Popen(
            ["./ngrok", "http", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.path.join(os.path.dirname(__file__), '..', '..')
        )
        time.sleep(5)
        assert pid.poll() is None, 'ngrok launch failed'
        return pid

    def run(self):
        pid = self.create_tensorboard_process()
        print(f'Running ngrok daemon forwarding from port {self.port}')

        while not self.event.is_set():
            time.sleep(1)
            assert pid.poll() is None, 'ngrok was killed'

        print('Stopping ngrok daemon')
        pid.terminate()
        pid.communicate()

    def stop(self):
        self.event.set()
        self.join()
