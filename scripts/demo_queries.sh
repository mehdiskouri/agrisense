#!/usr/bin/env bash
# Demo curl commands for AgriSense API
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
AUTH_TOKEN="${AUTH_TOKEN:-}"
API_KEY="${API_KEY:-}"
FARM_ID="${FARM_ID:-}"

pretty_print() {
	python3 -m json.tool
}

auth_header=()
if [[ -n "$AUTH_TOKEN" ]]; then
	auth_header=( -H "Authorization: Bearer $AUTH_TOKEN" )
fi

echo "=== AgriSense API Demo Queries ==="
echo ""

# Health checks
echo "--- Health Check ---"
curl -s "$BASE_URL/health" | pretty_print
echo ""

echo "--- Readiness Check ---"
curl -s "$BASE_URL/health/ready" | pretty_print
echo ""

if [[ -z "$AUTH_TOKEN" ]]; then
	echo "Set AUTH_TOKEN to run authenticated endpoint demos."
	exit 0
fi

echo "--- List Farms ---"
curl -s "${auth_header[@]}" "$BASE_URL/api/v1/farms" | pretty_print
echo ""

if [[ -n "$FARM_ID" ]]; then
	echo "--- Farm Status ---"
	curl -s "${auth_header[@]}" "$BASE_URL/api/v1/analytics/$FARM_ID/status" | pretty_print
	echo ""

	echo "--- Ask Endpoint ---"
	curl -s "${auth_header[@]}" \
		-H "Content-Type: application/json" \
		-X POST "$BASE_URL/api/v1/ask/$FARM_ID" \
		-d '{"question":"What is the current irrigation outlook?","language":"en"}' | pretty_print
	echo ""

	if [[ -n "$API_KEY" ]]; then
		echo "--- Ingest Soil (API Key) ---"
		curl -s -H "x-api-key: $API_KEY" -H "Content-Type: application/json" -X POST "$BASE_URL/api/v1/ingest/soil" -d "{
			\"farm_id\": \"$FARM_ID\",
			\"readings\": [
				{
					\"sensor_id\": \"00000000-0000-0000-0000-000000000001\",
					\"timestamp\": \"2026-01-01T00:00:00Z\",
					\"moisture\": 0.31,
					\"temperature\": 22.5
				}
			]
		}" | pretty_print
		echo ""
	fi
else
	echo "Set FARM_ID to run farm-scoped endpoint demos."
fi
