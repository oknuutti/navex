import threading
import socket
import warnings
import logging

import paramiko
import select

try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer


# based on paramiko demos forward.py and rforward.py
class Connection:
    def __init__(self, host, username, keyfile, proxy, local_forwarded_port):
        self._host = host
        self._username = username
        self._keyfile = keyfile
        self._proxy = proxy
        self.local_forwarded_port = local_forwarded_port

        self.remote_forwarded_port = None
        self._forwarding_thread = None
        self._reversing_thread = None
        self._proxy_client = None
        self._host_client = None
        self._sftp = None

        self._init_forwarding()
        self._open_connection()

    def _init_forwarding(self):
        def forward():
            c = paramiko.SSHClient()
            c.load_system_host_keys()
            c.set_missing_host_key_policy(paramiko.WarningPolicy())
            c.connect(self._proxy, username=self._username, key_filename=self._keyfile)
            Connection._forward_tunnel(self.local_forwarded_port, self._host, 22, c.get_transport())
            self._proxy_client = c

        self._forwarding_thread = threading.Thread(target=forward, daemon=True)
        self._forwarding_thread.start()

    def _open_connection(self):
        if self._host_client is None:
            c = paramiko.SSHClient()
            c.load_system_host_keys()
            c.set_missing_host_key_policy(paramiko.WarningPolicy())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Unknown \S+ host key")
                c.connect('127.0.0.1', port=self.local_forwarded_port,
                          username=self._username, key_filename=self._keyfile)
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

    def reverse_tunnel(self, local_host, local_port, remote_host='127.0.0.1', remote_port=0):
        transport = self._host_client.get_transport()
        self.remote_port = transport.request_port_forward(remote_host, remote_port)

        def reverse(lhost, lport):
            while True:
                chan = transport.accept(1000)
                if chan is not None:
                    Connection._reverse_handler(chan, lhost, lport)

        self._reversing_thread = threading.Thread(target=reverse, args=(local_host, local_port), daemon=True)
        self._reversing_thread.start()
        return self.remote_port

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
    def _forward_tunnel(local_port, remote_host, remote_port, transport):
        # this is a little convoluted, but lets me configure things for the Handler
        # object.  (SocketServer doesn't give Handlers any way to access the outer
        # server normally.)
        class SubHandler(Connection._ForwardHandler):
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
                chan = self.ssh_transport.open_channel(
                    "direct-tcpip",
                    (self.chain_host, self.chain_port),
                    self.request.getpeername(),
                )
            except Exception as e:
                logging.error(
                    "Incoming request to %s:%d failed: %s"
                    % (self.chain_host, self.chain_port, repr(e))
                )
                return
            if chan is None:
                logging.error(
                    "Incoming request to %s:%d was rejected by the SSH server."
                    % (self.chain_host, self.chain_port)
                )
                return

            logging.info(
                "Connected! Tunnel open %r -> %r -> %r"
                % (
                    self.request.getsockname(),
                    chan.getpeername(),
                    (self.chain_host, self.chain_port),
                )
            )
            while True:
                r, w, x = select.select([self.request, chan], [], [])
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

    @staticmethod
    def _reverse_handler(chan, host, port):
        sock = socket.socket()
        try:
            sock.connect((host, port))
        except Exception as e:
            logging.error("Reverse forwarding request to %s:%d failed: %r" % (host, port, e))
            return

        logging.info(
            "Connected! Reverse tunnel open %r <- %r <- %r"
            % ((host, port), chan.getpeername(), chan.origin_addr)
        )
        while True:
            r, w, x = select.select([sock, chan], [], [])
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
