terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5"
    }
  }
}

provider "cloudflare" {
}

resource "cloudflare_zero_trust_access_policy" "xolotl_zero_trust_access_policy" {
  account_id       = "f92413c21a2212429577cfbede5172dc"
  decision         = "allow"
  name             = "xolotl"
  session_duration = "730h"
  exclude          = []
  include = [{
    email = {
      email = "xolotl.mictl4n@gmail.com"
    }
  }]
  require = []
}
