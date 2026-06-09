#!/usr/bin/env bash
set -euo pipefail
kubectl delete namespace gpu-inference --ignore-not-found
echo "Namespace gpu-inference deleted."
