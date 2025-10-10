variable "cloudflare_account_id" {
  type      = string
  sensitive = true
}

variable "cloudflare_zone_id" {
  type      = string
  sensitive = true
}

variable "cloudflare_email" {
  type      = string
  sensitive = true
}

variable "domain" {
  description = "The domain name for the services"
  type        = string
  default     = "mictlan.cc"
}

variable "ingress_rules" {
  description = "List of ingress rules for the Cloudflare tunnel"
  type = list(object({
    subdomain = string
    port      = number
  }))
}
