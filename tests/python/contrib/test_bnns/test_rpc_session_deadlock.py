from tvm import rpc
from tvm.rpc import tracker
from tvm.contrib import xcode
from tvm.autotvm.measure import request_remote


KEY = "some_key"


class SomeHolder:
    def __init__(self, host, port):
        self.field_which_needed_for_deadlock = "arm64"
        self.deadlock_field = lambda output, objects, **kwargs: xcode.create_dylib(
            output, objects, arch=self.field_which_needed_for_deadlock
        )
        self.remote_session = request_remote(KEY, host, port, timeout=1000)

    def __del__(self):
        print("No references for {}".format(self))


def wrapper_for_object_destruction(host, port):
    SomeHolder(host, port)


def test_rpc_session_deadlock():
    local_host = "127.0.0.1"
    tracker_server = tracker.Tracker(local_host, 8888)
    remote_server = rpc.Server(
        local_host, port=9099, tracker_addr=(tracker_server.host, tracker_server.port), key=KEY
    )

    # Problem place
    for _ in [1, 2]:
        wrapper_for_object_destruction(tracker_server.host, tracker_server.port)

    remote_server.terminate()
    tracker_server.terminate()


if __name__ == '__main__':
    test_rpc_session_deadlock()
