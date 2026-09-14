#!/usr/bin/env bash
set -euo pipefail
parent=
maximum=-1
while read -r pid; do
  children=$(ps -eo ppid= | awk -v p="$pid" '$1==p {n++} END {print n+0}')
  if (( children > maximum )); then
    parent=$pid
    maximum=$children
  fi
done < <(pgrep -f '/python pack_data[.]py ')
test -n "$parent"
test "$maximum" -gt 0
echo "signaling pack parent=$parent children=$maximum"
kill -TERM "$parent"
