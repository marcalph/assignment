#!/usr/bin/env python3
# coding: utf-8
##########################################
# authors                                #
# marcalph - https://github.com/marcalph #
##########################################
"""code automation w/ nox
"""
import nox

locations = "noxfile.py", "src", "tests"


@nox.session(python="3.8.6")
def test(session):
    """test session"""
    session.run("pytest", "--cov=src", external=True)


@nox.session(python="3.8.6")
def lint(session):
    """lint session w/ flake8
    bandit      > code security
    black       > code format
    bugbear     > code design
    docstrings  > code doc
    """
    args = session.posargs or locations
    session.install(
        "flake8", "flake8-bandit", "flake8-black", "flake8-bugbear", "flake8-docstrings"
    )
    session.run("flake8", *args)


@nox.session(python="3.8.6")
def format(session):
    """code formatting session w/ black"""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python="3.8.6")
def secure(session):
    """check dependencies for known vulnerabilities"""
    session.install("safety")
    session.run("safety", "check", "--file=demo/requirements.txt")
