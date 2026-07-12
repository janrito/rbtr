# Greeter service infrastructure.
#
# The HCL plugin extracts each top-level block as a doc section, named by its
# type and labels: a bare block by its type (terraform), a variable/output by
# its name, a resource by its type and name.

terraform {
  required_version = ">= 1.5"
}

variable "region" {
  description = "Deployment region for the greeter service."
  type        = string
  default     = "us-east-1"
}

resource "aws_instance" "greeter" {
  ami           = "ami-0abc123"
  instance_type = "t3.micro"

  tags = {
    Name = "greeter"
  }
}

output "greeter_ip" {
  description = "Public IP of the greeter instance."
  value       = aws_instance.greeter.public_ip
}
