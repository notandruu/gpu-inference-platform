.PHONY: setup test lint export-onnx validate-model build-tensorrt \
        run-triton run-api benchmark docker-up docker-down \
        k8s-deploy k8s-clean build-cuda clean help

PYTHON  := python
UVICORN := uvicorn
PYTEST  := pytest

help:
	@echo "GPU Inference Platform — available targets:"
	@echo ""
	@echo "  setup            Install Python deps into .venv"
	@echo "  test             Run test suite (CPU-only tests)"
	@echo "  lint             Run ruff linter"
	@echo ""
	@echo "  export-onnx      Export ResNet-50 to ONNX [GPU optional]"
	@echo "  validate-model   Compare PyTorch vs ONNX output"
	@echo "  build-tensorrt   Build TensorRT engine [GPU REQUIRED]"
	@echo ""
	@echo "  run-triton       Start Triton Inference Server locally [GPU REQUIRED]"
	@echo "  run-api          Start FastAPI ingress on port 8000"
	@echo ""
	@echo "  benchmark        Run backend comparison benchmark"
	@echo "  build-cuda       Build C++/CUDA preprocessing extension [GPU REQUIRED]"
	@echo ""
	@echo "  docker-up        Start full stack via Docker Compose"
	@echo "  docker-down      Stop Docker Compose stack"
	@echo ""
	@echo "  k8s-deploy       Deploy to Kubernetes cluster"
	@echo "  k8s-clean        Remove Kubernetes resources"
	@echo ""
	@echo "  clean            Remove build artifacts"

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Activate with: source .venv/bin/activate"

test:
	$(PYTEST) -q -m "not gpu and not triton"

lint:
	ruff check .

export-onnx:
	$(PYTHON) models/export_onnx.py \
		--output model_repository/resnet50_onnx/1/model.onnx \
		--opset 17

validate-model:
	$(PYTHON) models/validate_model.py \
		--onnx model_repository/resnet50_onnx/1/model.onnx

build-tensorrt:
	@echo "[GPU REQUIRED] Building TensorRT engine..."
	bash scripts/build_tensorrt.sh

run-triton:
	@echo "[GPU REQUIRED] Starting Triton Inference Server..."
	bash scripts/run_triton_local.sh

run-api:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	$(PYTHON) benchmarks/run_backend_comparison.py \
		--output benchmarks/results/backend_comparison.json
	$(PYTHON) benchmarks/analyze_results.py \
		--input benchmarks/results/backend_comparison.json

build-cuda:
	@echo "[GPU REQUIRED] Building CUDA preprocessing extension..."
	cd cuda_preprocess && $(PYTHON) setup.py build_ext --inplace

docker-up:
	docker compose up --build -d
	@echo "API:        http://localhost:8000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000 (admin/admin)"

docker-down:
	docker compose down

k8s-deploy:
	bash scripts/deploy_k8s.sh

k8s-clean:
	bash scripts/cleanup_k8s.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
	rm -rf cuda_preprocess/build cuda_preprocess/*.egg-info
