"""HCL extraction test cases (top-level blocks -> config_key)."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]


@case(tags=["symbol"])
def case_hcl_splits_by_blocks() -> SymbolCase:
    """HCL splits by top-level blocks."""
    src = """\
resource "aws_instance" "web" {
  ami = "ami-12345"
}

variable "region" {
  default = "us-east-1"
}
"""
    return (
        "hcl",
        src,
        [
            ("config_key", "resource aws_instance web", ""),
            ("config_key", "variable region", ""),
        ],
    )
