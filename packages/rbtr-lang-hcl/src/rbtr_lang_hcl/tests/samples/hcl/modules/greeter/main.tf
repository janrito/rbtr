# The module main.tf reads via `source = "./modules/greeter"`. A Terraform
# module is a directory, so the reference reaches every .tf file in it.
variable "name" {
  type    = string
  default = "world"
}

output "greeting" {
  value = "Hello, ${var.name}"
}
