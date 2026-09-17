#!/usr/bin/env bash
# Serve games/ on a free localhost port; writes port and pid to /tmp/games_http.{port,pid}
cd "$(dirname "$0")/../games" || exit 1
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
setsid python3 -m http.server "$PORT" --bind 127.0.0.1 >/tmp/games_http.log 2>&1 < /dev/null &
echo $! > /tmp/games_http.pid; echo "$PORT" > /tmp/games_http.port
sleep 0.7; echo "serving on $PORT pid $(cat /tmp/games_http.pid)"
