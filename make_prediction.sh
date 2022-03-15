#!/usr/bin/env bash

PORT=8000
#HOST=172.17.0.3
HOST=127.0.0.1

echo "Port: $PORT"

# POST method predict
curl -d '{"SL":0.07471338, "SW":0.09794497, "PL":0.02951407, "PW": 0.01150299
}'\
     -H "Content-Type: application/json" \
     -X POST http://$HOST:$PORT/predict
