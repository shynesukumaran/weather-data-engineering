# --- Configuration ---
PYTHON_ENV := .venv 
PYTHON := $(PYTHON_ENV)/bin/python
PIP := $(PYTHON_ENV)/bin/pip

DASH_APP_MODULE := src.dashboard.app
PIPELINE_ORCHESTRATOR_MODULE := src.pipeline_orchestrator

DOCKER_IMAGE_NAME := weather-dashboard-app
DOCKER_PORT := 8050

include .env
export $(shell sed 's/=.*//' .env)


.PHONY: all install run-pipeline build-docker run-docker stop-docker clean help setup-env logs \
        run-pipeline-docker 

# --- Default target ---
# The 'all' target now orchestrates the Docker Compose workflow:
# 1. Builds the Docker images.
# 2. Ensures the postgres_db is up and healthy.
# 3. Runs the pipeline as a one-off job to populate the DB.
# 4. Starts the app service.
# 5. Tails the app logs.
all: build-docker docker-up-db run-pipeline-docker docker-up-app logs-app


help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets (Docker-centric):"
	@echo "  all                   - Builds images, runs pipeline once, then starts the dashboard and DB."
	@echo "  build-docker          - Builds the Docker image for the application and pipeline."
	@echo "  docker-up-db          - Starts only the PostgreSQL database service in detached mode."
	@echo "  run-pipeline-docker   - Runs the data pipeline once inside a Docker container (populates DB)."
	@echo "  docker-up-app         - Starts only the Dash application service in detached mode."
	@echo "  stop-docker           - Stops and removes all Docker Compose services, networks, and volumes."
	@echo "  logs                  - Displays logs for all running Docker services."
	@echo "  logs-app              - Displays logs for the Dash application container."
	@echo "  clean                 - Removes Docker artifacts (images, volumes) and local generated files."
	@echo ""
	@echo "Targets (Local Python Env - for development/testing outside Docker):"
	@echo "  setup-env             - Create and activate Python virtual environment."
	@echo "  install               - Install Python dependencies from requirements.txt into .venv."
	@echo "  run-pipeline          - Execute the data pipeline using local Python environment."
	@echo ""
	@echo "Before running 'all' or Docker targets, ensure:"
	@echo "  1. Your docker-compose.yml is correctly configured with 'app', 'postgres_db', and 'pipeline' services."
	@echo "  2. Your .env file at the project root is correctly configured with DB credentials."
	@echo "  3. The necessary DB tables are created (they are created automatically by the pipeline)."

# --- Environment Setup (for local development, not used by docker targets directly) ---
setup-env:
	@echo "Setting up Python virtual environment..."
	python3.11 -m venv $(PYTHON_ENV)
	@echo "Virtual environment created at $(PYTHON_ENV). Activate it with: source $(PYTHON_ENV)/bin/activate"

# --- Install Dependencies (for local development, not used by docker targets directly) ---
install: setup-env
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

# --- Run Data Pipeline (for local development, not used by docker targets directly) ---
run-pipeline: install
	@echo "Running the complete data pipeline locally..."
	$(PYTHON) -m $(PIPELINE_ORCHESTRATOR_MODULE)
	@echo "Data pipeline execution complete. Data should be in PostgreSQL."

# --- Docker Operations (Primary Workflow) ---
build-docker:
	@echo "Building Docker images for all services defined in docker-compose.yml..."
	docker compose build
	@echo "Docker images built."

docker-up-db:
	@echo "Starting PostgreSQL database service..."
	docker compose up -d postgres_db
	@echo "Waiting for DB healthcheck to pass..."
	@until [ "$$(docker inspect --format='{{.State.Health.Status}}' weather_postgres_db)" = "healthy" ]; do \
		echo "Still waiting for PostgreSQL to be healthy..."; \
		sleep 2; \
	done
	@echo "PostgreSQL is healthy and ready to accept connections."


run-pipeline-docker: docker-up-db
	@echo "Running the data pipeline inside a Docker container to populate the database..."
	docker compose run --rm pipeline # Runs the pipeline service once and removes the container after it exits
	@echo "Pipeline execution complete. Database should be populated."

docker-up-app:
	@echo "Starting the Dash application service..."
	docker compose up -d app
	@echo "Dash application service started. Access at http://localhost:$(DOCKER_PORT)"

stop-docker:
	@echo "Stopping and removing all Docker Compose services, networks, and volumes..."
	docker compose down -v --remove-orphans
	@echo "Docker services stopped and cleaned up."

logs:
	@echo "Displaying logs for all Docker Compose services. Press Ctrl+C to exit."
	docker compose logs -f

logs-app:
	@echo "Displaying logs for the Dash application container. Press Ctrl+C to exit."
	docker compose logs -f app

# --- Clean up ---
clean: stop-docker
	@echo "Removing Docker images and local generated files..."
	docker rmi $(DOCKER_IMAGE_NAME) || true
	docker rmi postgres:16-alpine || true
	docker volume prune -f || true
	rm -f data/*.csv
	rm -f pipeline.log
	rm -rf .cache/
	rm -rf $(PYTHON_ENV)
	find . -depth -name "__pycache__" -exec rm -rf {} \;
	@echo "Cleanup complete."