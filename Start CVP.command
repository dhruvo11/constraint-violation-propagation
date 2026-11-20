#!/bin/bash
# Double-click to start the CVP web app. It opens in your browser once it is ready;
# keep this window open while you use it, and close it (or press Ctrl+C) to stop.
# (The earlier Gradio demo still runs with: .venv/bin/python app.py)
cd "$(dirname "$0")"
PORT=${PORT:-7860}
URL="http://127.0.0.1:$PORT"
export GRADIO_ANALYTICS_ENABLED=False

if [ ! -x .venv/bin/python ]; then
  echo "The Python environment is missing. Run ./setup.sh first."
  read -n 1 -s -r -p "Press any key to close."; exit 1
fi

if curl -s -o /dev/null "$URL"; then
  echo "CVP is already running at $URL"
  [ -z "$CVP_NO_BROWSER" ] && open "$URL"
  exit 0
fi

echo "Starting CVP (the page opens in a few seconds; the models finish loading in the background) ..."
PORT=$PORT CVP_NO_BROWSER=1 .venv/bin/python -m webapp &
APP=$!
trap 'kill $APP 2>/dev/null' EXIT INT TERM
until curl -s -o /dev/null "$URL"; do
  if ! kill -0 $APP 2>/dev/null; then
    echo "The app stopped before it was ready; the messages above say why."
    read -n 1 -s -r -p "Press any key to close."; exit 1
  fi
  sleep 1
done
echo
echo "CVP is running at $URL"
[ -z "$CVP_NO_BROWSER" ] && open "$URL"
echo "Keep this window open while you use it. Close it or press Ctrl+C to stop."
wait $APP
