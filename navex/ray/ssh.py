import os
import threading
import socket
import time
import warnings
import logging
from functools import partial

import paramiko
import select
from paramiko import Transport, SSHException

try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer


# based on paramiko demos forward.py and rforward.py
class Connection:
    TIMEOUT = 30    # in seconds
    reverse_map = {}

    def __init__(self, host, username=None, keyfile=None, proxy=None, local_forwarded_port=None):
        self._host = host
        self._username = username or None
        self._keyfile = keyfile or None
        self._proxy = proxy or None
        self.local_forwarded_port = local_forwarded_port

        self._forwarding_thread = None
        self._reversing_thread = None
        self._proxy_client = None
        self._host_client = None
        self._sftp = None

        if self._proxy is not None:
            self._init_forwarding()
        self._open_connection()

    def _init_forwarding(self):
        def forward():
            c = paramiko.SSHClient()
            c.load_system_host_keys()
            c.set_missing_host_key_policy(paramiko.WarningPolicy())
            c.connect(self._proxy, username=self._username, key_filename=self._keyfile)
            Connection._forward_tunnel(self.local_forwarded_port, self._host, 22, c.get_transport(), timeout=None)
            self._proxy_client = c

        self._forwarding_thread = threading.Thread(target=forward, daemon=True)
        self._forwarding_thread.start()

    def _open_connection(self):
        if self._host_client is None:
            if self._forwarding_thread is None:
                host, port = self._host, 22
            else:
                host, port = '127.0.0.1', self.local_forwarded_port
            c = paramiko.SSHClient()
            c.load_system_host_keys()
            c.set_missing_host_key_policy(paramiko.WarningPolicy())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Unknown \S+ host key")
                c.connect(host, port=port, username=self._username, key_filename=self._keyfile)
            self._host_client = c

    def _close_connection(self):
        if self._host_client is not None:
            self._host_client.close()
        if self._proxy_client is not None:
            self._proxy_client.close()
        if self._sftp is not None:
            self._sftp.close()

    def __del__(self):
        self._close_connection()

    def tunnel(self, src_port, dst_port, dst_host='localhost'):
        logging.debug('init tunnel %d => %s:%d' % (src_port, dst_host, dst_port))
        Connection._forward_tunnel(src_port, dst_host, dst_port, self._host_client.get_transport())

    def reverse_tunnel(self, local_host, local_port, remote_host='localhost', remote_port=0):
        transport = self._host_client.get_transport()
        override_transport(transport)
        remote_port = transport.request_port_forward(remote_host, remote_port)
        Connection.reverse_map[remote_port] = (local_host, local_port)

        def reverse():
            while True:
                chan = transport.accept(1000)
                if chan is not None:
                    _reverse_handler(chan)

        threading.Thread(target=reverse, daemon=True).start()
        time.sleep(0.1)
        return remote_port

    def exec(self, cmd):
        stdin, stdout, stderr = self._host_client.exec_command(cmd)
        out = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        return out, error

    def fetch(self, src, dst):
        if self._sftp is None:
            self._sftp = self._host_client.open_sftp()
        self._sftp.get(src, dst)

    @staticmethod
    def _forward_tunnel(local_port, remote_host, remote_port, transport, timeout=TIMEOUT):
        # this is a little convoluted, but lets me configure things for the Handler
        # object.  (SocketServer doesn't give Handlers any way to access the outer
        # server normally.)
        class SubHandler(Connection._ForwardHandler):
            _timeout = timeout
            chain_host = remote_host
            chain_port = remote_port
            ssh_transport = transport

        Connection._ForwardServer(("", local_port), SubHandler).serve_forever()

    class _ForwardServer(SocketServer.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    class _ForwardHandler(SocketServer.BaseRequestHandler):
        def handle(self):
            try:
                local_addr = self.request.getsockname()
                remote_addr = self.chain_host, self.chain_port
                chan = self.ssh_transport.open_channel(
                    "direct-tcpip",
                    remote_addr,
                    self.request.getpeername(),
                    timeout=Connection.TIMEOUT,
                )
            except Exception as e:
                logging.error(
                    "Incoming request to %s:%d failed: %s"
                    % (*remote_addr, repr(e))
                )
                return
            if chan is None:
                logging.error(
                    "Incoming request to %s:%d was rejected by the SSH server."
                    % remote_addr
                )
                return

            logging.info(
                "Connected! Tunnel open %r -> %r -> %r"
                % (
                    local_addr,
                    chan.getpeername(),
                    remote_addr,
                )
            )
            while True:
                r, w, x = select.select([self.request, chan], [], [], self._timeout)
                if not r:
                    logging.warning(
                        'fw-tunnel channel read timeout %fs (%s => %s)' % (Connection.TIMEOUT, local_addr, remote_addr))
                    break
                if self.request in r:
                    data = self.request.recv(1024)
                    if len(data) == 0:
                        break
                    chan.send(data)
                if chan in r:
                    data = chan.recv(1024)
                    if len(data) == 0:
                        break
                    self.request.send(data)

            peername = self.request.getpeername()
            chan.close()
            self.request.close()
            logging.info("Tunnel closed from %r" % (peername,))


def override_transport(transport):
    class MyTransport(Transport):
        def request_port_forward(self, address, port, handler=None):
            if not self.active:
                raise SSHException("SSH session not active")
            port = int(port)
            response = self.global_request(
                "tcpip-forward", (address, port), wait=True
            )
            if response is None:
                raise SSHException("TCP forwarding request denied")
            if port == 0:
                port = response.get_int()
            if handler is None:
                def default_handler(channel, src_addr, dest_addr_port):
                    # src_addr, src_port = src_addr_port
                    # dest_addr, dest_port = dest_addr_port
                    channel.origin_addr = dest_addr_port        # THE ONLY CHANGE VS "Transport" IS THIS LINE
                    self._queue_incoming_channel(channel)

                handler = default_handler
            self._tcp_handler = handler
            return port
    transport.__class__ = MyTransport


def _reverse_handler(chan):
    rem_dst_host, rem_dst_port = rem_dst_addr = chan.origin_addr
    loc_dst_host, loc_dst_port = loc_dst_addr = Connection.reverse_map[rem_dst_port]

    sock = socket.socket()
    try:
        sock.connect((loc_dst_host, loc_dst_port))
    except Exception as e:
        logging.error("Reverse forwarding request to %s:%d failed: %r" % (loc_dst_host, loc_dst_port, e))
        return

    logging.info(
        "Connected! Reverse tunnel open %r <- %r <- %r"
        % ((loc_dst_host, loc_dst_port), chan.getpeername(), chan.origin_addr)
    )
    while True:
        r, w, x = select.select([sock, chan], [], [], Connection.TIMEOUT)
        if not r:
            logging.warning('rev-tunnel channel read timeout %fs (%s <= %s)' % (Connection.TIMEOUT, loc_dst_addr, rem_dst_addr))
            break
        if sock in r:
            data = sock.recv(1024)
            if len(data) == 0:
                break
            chan.send(data)
        if chan in r:
            data = chan.recv(1024)
            if len(data) == 0:
                break
            sock.send(data)
    chan.close()
    sock.close()
    logging.info("Tunnel closed from %r" % (chan.origin_addr,))
