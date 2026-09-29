#!/usr/bin/env bash
# Usage: demo.sh <switchyard-server binary> <label>
BIN=$1; LABEL=$2; cd "$(dirname "$0")"
echo "\$ grep -E '^(mode|response_format_type) ' custom-json-object.toml"
grep -E '^(mode|response_format_type) ' custom-json-object.toml
echo
python3 mock.py 18431 > mock.log 2>&1 & MOCK=$!
echo "\$ switchyard-server --config custom-json-object.toml --port 18432   # $LABEL"
$BIN --config custom-json-object.toml --host 127.0.0.1 --port 18432 > server.log 2>&1 & SRV=$!
for i in $(seq 1 50); do curl -sf -o /dev/null http://127.0.0.1:18432/v1/models && break; kill -0 $SRV 2>/dev/null || break; sleep 0.2; done
if ! kill -0 $SRV 2>/dev/null; then
  wait $SRV; echo "server exited with status $?:"; grep -iE "error|custom" server.log | sed 's/\x1b\[[0-9;]*m//g' | tail -3
  kill $MOCK; exit 0
fi
echo "server started"
echo
echo "\$ curl -s localhost:18432/v1/chat/completions -d '{\"model\":\"switchyard/custom\",...}' | jq -r '.model, .choices[0].message.content'"
curl -s http://127.0.0.1:18432/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"switchyard/custom","messages":[{"role":"user","content":"Refactor the scheduler."}]}' | jq -r '.model, .choices[0].message.content'
echo
echo "\$ cat mock.log   # what the upstream saw"
cat mock.log
kill $SRV $MOCK
