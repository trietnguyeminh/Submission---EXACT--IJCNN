#!/bin/bash

BRANCH="main"
SLEEP_SEC=2
COUNT=1

FILES=$(git ls-files --others --exclude-standard)

if [ -z "$FILES" ]; then
  echo "Không có file mới để upload."
  exit 0
fi

echo "Bắt đầu upload từng file một..."

while IFS= read -r file; do
  echo "[$COUNT] Đang upload: $file"

  git add -- "$file"

  if git diff --cached --quiet; then
    echo "Bỏ qua vì không có thay đổi: $file"
    continue
  fi

  git commit -m "Add file $COUNT: $file"
  git push -u origin "$BRANCH"

  COUNT=$((COUNT + 1))
  sleep "$SLEEP_SEC"
done <<< "$FILES"

echo "Hoàn tất."
