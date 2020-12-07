import nox

locations = "src", "tests", "noxfile.py"


@nox.session(python="3.8.6")
def test(session):
    session.run("pytest", "--cov=src", external=True)


@nox.session(python="3.8.6")
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-black")
    session.run("flake8", *args)


@nox.session(python="3.8.6")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)

