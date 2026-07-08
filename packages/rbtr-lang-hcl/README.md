# rbtr-lang-hcl

HCL / Terraform support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[hcl]`.

## What it ingests

Top-level blocks become **config-key** chunks (HCL is configuration, not
code). A block is named by its type and labels combined.

## Chunks produced

```hcl
resource "aws_instance" "web" {}   # config_key "resource aws_instance web"
variable "region" {}               # config_key "variable region"
terraform {}                       # config_key "terraform"
```

## Embedded / injected chunks

None. HCL does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-hcl` grammar (`.hcl`, `.tf`). No dependency on other
language plugins.
