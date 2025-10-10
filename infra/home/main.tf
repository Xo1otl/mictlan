terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = ">= 5.11.0"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.5.3"
    }
  }
}

provider "cloudflare" {
}

resource "cloudflare_zero_trust_access_policy" "allow_emails" {
  account_id = var.cloudflare_account_id
  decision   = "allow"
  name       = "Allow email addresses"
  include = [{
    email = {
      email = var.cloudflare_email
    }
  }]
}

resource "cloudflare_zero_trust_device_default_profile" "default_profile" {
  account_id = var.cloudflare_account_id
  include = [{
    address     = "100.96.0.0/12"
    description = "Include domains in the tunnel"
  }]
  tunnel_protocol = "wireguard"
}

resource "cloudflare_zero_trust_tunnel_cloudflared" "mictlan_tunnel" {
  account_id = var.cloudflare_account_id
  name       = "mictlan-devcontainer"
  config_src = "cloudflare"
}

resource "cloudflare_dns_record" "portfolio" {
  zone_id = var.cloudflare_zone_id
  name    = "portfolio"
  content = "${cloudflare_zero_trust_tunnel_cloudflared.mictlan_tunnel.id}.cfargotunnel.com"
  type    = "CNAME"
  ttl     = 1
  proxied = true
}

data "cloudflare_zero_trust_tunnel_cloudflared_token" "mictlan_tunnel_token" {
  account_id = var.cloudflare_account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.mictlan_tunnel.id
}

resource "cloudflare_zero_trust_tunnel_cloudflared_config" "mictlan_tunnel_config" {
  account_id = var.cloudflare_account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.mictlan_tunnel.id

  config = {
    ingress = [{
      hostname = "portfolio.mictlan.cc"
      service  = "http://devcontainer:1111"
      }, {
      service = "http_status:404"
    }]
  }
}

resource "local_file" "tunnel_token_file" {
  content  = "CLOUDFLARE_TUNNEL_TOKEN=${data.cloudflare_zero_trust_tunnel_cloudflared_token.mictlan_tunnel_token.token}"
  filename = "${path.module}/.env"
}
