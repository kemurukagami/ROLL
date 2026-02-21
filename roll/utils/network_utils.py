import socket


def get_node_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except PermissionError:
        # Some sandboxed environments disallow socket creation; default to loopback.
        return "127.0.0.1"


def collect_free_port():
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]
