# Kubernetes Deployment

## Prerequisites

- Kubernetes cluster with at least one GPU node
- [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) installed on GPU nodes
- `kubectl` configured to target the cluster
- Model artifacts pushed to a shared volume or container image

## Deploy

```bash
make k8s-deploy
# or manually:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/triton-deployment.yaml
kubectl apply -f k8s/triton-service.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/dcgm-exporter-daemonset.yaml
kubectl apply -f k8s/hpa-api.yaml
```

## Verify

```bash
kubectl get pods -n gpu-inference
kubectl describe pod -l app=triton -n gpu-inference
```

## Access services via port-forward

```bash
# FastAPI
kubectl port-forward svc/gpu-inference-api 8000:8000 -n gpu-inference

# Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n gpu-inference

# Grafana (admin/admin)
kubectl port-forward svc/grafana 3000:3000 -n gpu-inference
```

## GPU scheduling

Triton requests `nvidia.com/gpu: 1` in resource limits. The NVIDIA device plugin exposes this resource on GPU nodes. Pods are automatically scheduled to nodes with available GPUs.

## Horizontal scaling

`hpa-api.yaml` configures HPA for the FastAPI tier (CPU-based, 70% target). Triton horizontal scaling requires multiple GPU nodes or multiple GPUs per node — increase replicas in `triton-deployment.yaml` and ensure the model repository is on shared storage (e.g., PVC backed by NFS or object storage).

## Tear down

```bash
make k8s-clean
# or:
kubectl delete namespace gpu-inference
```
