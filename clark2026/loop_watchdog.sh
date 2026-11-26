#!/bin/bash
# Stops a coding container whose recent tool calls have collapsed into a loop (>=40 recent calls, <12 distinct).
# Runs while any gen_code.py process is alive. The session is then discarded by gen_code.py.
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*gen_code.py*' }) { exit 0 } else { exit 1 }"; do
  for c in $(docker ps --format "{{.Names}}" | grep '^clark-'); do
    v=$(docker exec $c bash -c "tail -c 150000 /root/sessions/session.jsonl 2>/dev/null" | python -c "
import sys,json
calls=[]
for l in sys.stdin.read().splitlines()[1:]:
    try: e=json.loads(l)
    except Exception: continue
    for b in (e.get('message') or {}).get('content') or []:
        if isinstance(b,dict) and b.get('type')=='toolCall': calls.append(json.dumps(b.get('arguments')))
calls=calls[-60:]
print(len(calls), len(set(calls)))")
    set -- $v
    if [ "${1:-0}" -ge 40 ] && [ "${2:-99}" -lt 12 ]; then
      echo "$(date +%H:%M) LOOP $c: $1 recent calls, $2 distinct -> stopped"; docker rm -f $c >/dev/null
    fi
  done
  sleep 120
done
echo "$(date +%H:%M) watchdog: no coding containers left"
