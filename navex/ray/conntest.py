import socket
import argparse
import select


def main():
    parser = argparse.ArgumentParser('test socket connectivity')
    parser.add_argument('--port', '-p', type=int, help="port")
    parser.add_argument('--host', '-H', default='localhost', help="host")
    parser.add_argument('--msg', '-m', default='HELO', help="message")
    parser.add_argument('--timeout', '-t', type=int, default=3, help="timeout")
    args = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.setblocking(False)
    try:
        s.connect((args.host, args.port))
        s.send(args.msg.encode('utf-8'))

        ready = select.select([s], [], [], args.timeout)
        if ready[0]:
            data = s.recv(4096)
            try:
                string = data.decode('utf-8')
            except UnicodeDecodeError:
                string = str(data)
            print('success: %s' % string)
        else:
            print('timeout')

    except Exception as e:
        print('fail: %s' % e)

    finally:
        s.close()


if __name__ == '__main__':
    main()
