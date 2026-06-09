#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="gpu-inference"

echo "Deploying GPU Inference Platform to Kubernetes..."

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/triton-deployment.yaml
kubectl apply -f k8s/triton-service.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/dcgm-exporter-daemonset.yaml
kubectl apply -f k8s/hpa-api.yaml

echo ""
echo "Deployed to namespace: $NAMESPACE"
echo ""
echo "To access services:"
echo "  kubectl port-forward svc/gpu-inference-api 8000:8000 -n $NAMESPACE"
echo "  kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE"
echo "  kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
echo ""
echo "Check pod status:"
echo "  kubectl get pods -n $NAMESPACE"
