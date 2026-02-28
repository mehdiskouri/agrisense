#!/usr/bin/env bash
# Demo curl commands for AgriSense API
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=== AgriSense API Demo Queries ==="
echo ""

# Health check
echo "--- Health Check ---"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""
