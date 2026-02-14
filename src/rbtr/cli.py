"""CLI entry points for rbtr."""

import click


@click.group()
def main() -> None:
    """rbtr — Interactive LLM-powered PR review workbench."""


@main.command()
@click.argument("pr_url")
def review(pr_url: str) -> None:
    """Review a GitHub pull request interactively."""
    click.echo(f"TODO: launch TUI for {pr_url}")
