# What the evaluation workflow needs on Google Cloud, in a project of its own so that its spend is counted apart:
# Vertex AI, a service account that may only call it, and a keyless trust from GitHub Actions to that account,
# limited to the repository's `eval` environment. The trust is the github-wif module of terraform-cloud-platform.

terraform {
  required_version = ">= 1.10"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.0"
    }
  }
  backend "gcs" {
    prefix = "consultor-tributario/ci" # terraform init -backend-config="bucket=<name>"
  }
}

variable "project_id" {
  type = string
}

variable "repository" {
  description = "OWNER/REPO whose eval environment may run the evaluation."
  type        = string
  default     = "0103juan/consultor-tributario"
}

variable "billing_account" {
  type = string
}

variable "budget" {
  description = "Monthly gross spend (credits left out) at which the billing account's admins get an email. It warns; it stops nothing."
  type        = number
}

variable "budget_currency" {
  type    = string
  default = "USD"
}

provider "google" {
  project = var.project_id
  region  = "us-central1"
}

data "google_project" "this" {}

resource "google_project_service" "this" {
  for_each           = toset(["aiplatform.googleapis.com", "billingbudgets.googleapis.com", "iamcredentials.googleapis.com", "sts.googleapis.com"])
  service            = each.value
  disable_on_destroy = false
}

module "eval" {
  source = "github.com/0103juan/terraform-cloud-platform//gcp/modules/github-wif?ref=23de741"

  project_id    = var.project_id
  name          = "consultor-eval"
  subjects      = ["repo:${var.repository}:environment:eval"]
  project_roles = ["roles/aiplatform.user"]

  depends_on = [google_project_service.this]
}

resource "google_billing_budget" "eval" {
  billing_account = var.billing_account
  display_name    = "consultor-tributario evaluation"
  budget_filter {
    projects               = ["projects/${data.google_project.this.number}"]
    credit_types_treatment = "EXCLUDE_ALL_CREDITS"
  }
  amount {
    specified_amount {
      currency_code = var.budget_currency
      units         = tostring(var.budget)
    }
  }
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 1.0
  }
  depends_on = [google_project_service.this]
}

output "workload_identity_provider" {
  value = module.eval.workload_identity_provider
}

output "service_account" {
  value = module.eval.service_account
}
