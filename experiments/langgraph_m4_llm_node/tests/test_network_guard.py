import socket

import pytest


def test_offline_guard_rejects_external_connections_and_dns():
    address = ("192.0.2.1", 443)
    with socket.socket() as sock:
        with pytest.raises(AssertionError, match="numeric loopback"):
            sock.connect(address)
        with pytest.raises(AssertionError, match="numeric loopback"):
            sock.connect_ex(address)
    with pytest.raises(AssertionError, match="numeric loopback"):
        socket.getaddrinfo("example.invalid", 443)

    assert socket.getaddrinfo("127.0.0.1", 443)
