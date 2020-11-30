import socket
import argparse
import select
import time

from .ssh import Connection


def main():
    parser = argparse.ArgumentParser('test socket connectivity')
    parser.add_argument('--port', '-p', type=int, help="port")
    parser.add_argument('--host', '-H', default='localhost', help="host")
    parser.add_argument('--username', '-u', help="username for tunneling")
    parser.add_argument('--keyfile', '-f', help="keyfile for tunneling")
    parser.add_argument('--msg', '-m', default=r'HELO\n', help="message")
    parser.add_argument('--timeout', '-t', type=int, default=3, help="timeout")
    parser.add_argument('--listen', action='store_true', help="listen to a port instead")
    parser.add_argument('--keep', '-k', action='store_true', help="don't let go of connections")
    parser.add_argument('--local', '-l', type=int, help="local port for tunneling")
    parser.add_argument('--fw', '-L', action='store_true', help="forward tunnel from local to remote")
    parser.add_argument('--rev', '-R', action='store_true', help="reverse tunnel from remote to local")
    args = parser.parse_args()

    if args.fw or args.rev:
        ssh = Connection(args.host, args.username or None, args.keyfile or None)
        if args.fw:
            assert not args.rev, "can't give both --fw and --rev"
            ssh.tunnel(args.local or args.port, args.port)
        else:
            ssh.reverse_tunnel('127.0.0.1', args.local or args.port, '127.0.0.1', args.port)
        time.sleep(3600)
        return

    addr = (args.host, args.port)
    msg = args.msg.replace(r'\n', '\n').encode('utf-8')
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.setblocking(False)

    try:
        if args.listen:
            s.settimeout(args.timeout)
            s.bind(addr)
            s.listen()
            try:
                while True:
                    (conn, (client_host, client_port)) = s.accept()
                    data = conn.recv(4096)
                    print('from %s:%s received "%s"' % (client_host, client_port, _decode(data)))
                    conn.send(msg)
                    if not args.keep:
                        break
            except socket.timeout:
                print('timeout')
        else:
            s.connect(addr)
            s.send(msg)
            ready = select.select([s], [], [], args.timeout)
            if ready[0]:
                data = s.recv(4096)
                print('success: %s' % _decode(data))
                if args.keep:
                    time.sleep(3600)
            else:
                print('timeout')

    except Exception as e:
        print('fail: %s' % e)

    finally:
        s.close()


def _decode(data):
    try:
        string = data.decode('utf-8')
    except UnicodeDecodeError:
        string = str(data)
    return string


if __name__ == '__main__':
    main()
