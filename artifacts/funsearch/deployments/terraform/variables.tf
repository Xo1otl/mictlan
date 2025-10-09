variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud region for deployment"
  type        = string
  default     = "asia-northeast1"
}

variable "service_name" {
  description = "Name of the Cloud Run service"
  type        = string
  default     = "funsearch"
}

variable "google_cloud_api_key" {
  description = "Google Cloud API key for Gemini"
  type        = string
  sensitive   = true
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  sensitive   = true
}

variable "github_owner" {
  description = "GitHub owner (username or organization) for the repository"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository name for the Cloud Build trigger"
  type        = string
}