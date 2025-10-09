terraform {
  backend "gcs" {
    bucket = "qunasys-ai-dev-funsearch-tfstate"
    prefix = "terraform/state"
  }
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "cloud_run_api" {
  service = "run.googleapis.com"
}

resource "google_project_service" "artifact_registry_api" {
  service = "artifactregistry.googleapis.com"
}

resource "google_project_service" "cloud_build_api" {
  service = "cloudbuild.googleapis.com"
}

# Artifact Registry repository
resource "google_artifact_registry_repository" "funsearch" {
  repository_id = "funsearch"
  location      = var.region
  format        = "DOCKER"
  description   = "FunSearch Docker repository"

  depends_on = [google_project_service.artifact_registry_api]
}

# Cloud Run service
resource "google_cloud_run_service" "funsearch" {
  name     = var.service_name
  location = var.region

  template {
    metadata {
      annotations = {
        "run.googleapis.com/ingress" = "all"
      }
    }

    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/funsearch/funsearch:latest"

        ports {
          container_port = 7860
        }

        env {
          name  = "GOOGLE_CLOUD_API_KEY"
          value = var.google_cloud_api_key
        }

        env {
          name  = "SLACK_WEBHOOK_URL"
          value = var.slack_webhook_url
        }

      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [google_project_service.cloud_run_api]
}

# Cloud Build trigger for automatic deployment
resource "google_cloudbuild_trigger" "funsearch" {
  name        = "funsearch-trigger"
  description = "Build and deploy FunSearch on push to main"

  service_account = "projects/${var.project_id}/serviceAccounts/funsearch-build@qunasys-ai-dev.iam.gserviceaccount.com"

  github {
    owner = "Xo1otl"
    name  = "funsearch"
    push {
      branch = "^main$"
    }
  }

  included_files = ["src/funsearch/**", "ui/**", "Dockerfile", "pyproject.toml.docker"]

  build {
    options {
      logging = "CLOUD_LOGGING_ONLY" # これがないとログバケットの準備などが必要になる
    }

    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "--no-cache",
        "-t", "${var.region}-docker.pkg.dev/${var.project_id}/funsearch/funsearch:$COMMIT_SHA",
        "-t", "${var.region}-docker.pkg.dev/${var.project_id}/funsearch/funsearch:latest",
        "-f", "Dockerfile",
        "."
      ]
    }

    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "push",
        "--all-tags",
        "${var.region}-docker.pkg.dev/${var.project_id}/funsearch/funsearch"
      ]
    }

    step {
      name       = "gcr.io/google.com/cloudsdktool/cloud-sdk:slim"
      entrypoint = "gcloud"
      args = [
        "run", "services", "update", var.service_name,
        "--platform=managed",
        "--image=${var.region}-docker.pkg.dev/${var.project_id}/funsearch/funsearch:$COMMIT_SHA",
        "--region=${var.region}",
        "--quiet"
      ]
    }
  }

  depends_on = [google_project_service.cloud_build_api]
}