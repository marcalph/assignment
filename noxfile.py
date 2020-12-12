import nox
import tempfile

locations = "noxfile.py", "src", "tests"


@nox.session(python="3.8.6")
def test(session):
    session.run("pytest", "--cov=src", external=True)


@nox.session(python="3.8.6")
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-bandit", "flake8-black", "flake8-bugbear")
    session.run("flake8", *args)


@nox.session(python="3.8.6")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python="3.8.6")
def safety(session):
    """ check dependencies for known security vulnerabilities
    """
    session.install("safety")
    session.run("safety", "check", "--file=demo/requirements.txt")
