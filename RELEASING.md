# Releasing FDEint

The package version is defined in `setup.py`; PyPI reads its description from
`README.md`. The workflow `.github/workflows/publish.yml` builds distributions
and tests the installed wheel on pushes to `main`, pull requests, manual runs,
and published GitHub releases. Only a published GitHub release uploads to PyPI,
after all build and test jobs succeed.

## One-time Trusted Publishing setup

Create a GitHub environment named `pypi` under the repository's
**Settings → Environments**. In the FDEint project's PyPI settings, open
**Publishing** and add a GitHub publisher with these exact values:

| Field | Value |
| --- | --- |
| Repository owner | `zimmer-ing` |
| Repository name | `FDEint` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

The filename is `publish.yml`, without the `.github/workflows/` prefix.
Trusted Publishing uses the workflow's identity, so no PyPI API token or
password needs to be stored in GitHub. See the
[PyPI setup guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Publish version 0.1.3

1. Confirm that `setup.py` has version `0.1.3` and review the changelog.
2. Commit and push the release changes, including
   `.github/workflows/publish.yml`, to `main`.
3. Open the repository's **Actions** tab. Wait for **Tests and PyPI release**
   to finish successfully. The PyPI publishing job is skipped on a normal push.
4. Open **Releases → Draft a new release** on GitHub. Create a new tag
   `v0.1.3` targeting the updated `main` branch, give the release a title such as
   `FDEint 0.1.3`, and add release notes from the changelog.
5. Click **Publish release**. The workflow builds and tests that tagged source,
   checks that tag `v0.1.3` matches package version `0.1.3`, and uploads the
   tested wheel and source archive to PyPI.
6. Check the workflow result and the
   [PyPI project page](https://pypi.org/project/fdeint/). Verify installation
   with `python -m pip install 'FDEint==0.1.3'` in a fresh environment.

If the GitHub environment requires a reviewer, the publish job waits for that
review in the workflow run. A draft release does not publish to PyPI. The
**Run workflow** button performs the build and tests without uploading.

For later releases, update the version in `setup.py` and the changelog, push
the changes, and repeat the release steps with a matching `v<version>` tag.
The tag must point to a commit containing this workflow and the new version.
A published PyPI file cannot be replaced; corrections require a new version.

## What the workflow checks

- Build a source archive and a wheel from that source archive in a fresh build
  environment, then check their descriptions and metadata with Twine.
- Install the same built wheel on Python 3.8–3.15 and run the regression tests
  and external-input example outside the source checkout.
- Run Python 3.8 and 3.9 on Ubuntu 22.04, where GitHub provides those
  interpreters. Newer Python jobs use `ubuntu-latest`.
- Test PyTorch 2.2.1 on Python 3.8, and compatible current PyTorch versions on
  the other Python versions. CPU-only PyTorch packages keep CI independent of
  GPU hardware. CUDA verification still requires a compatible GPU.

`torch>=2.2.1` is a minimum, not a pin. Installers choose a compatible newer
version when one is available. The supported Python minimum remains 3.8.

## Local checks and manual fallback

Run these commands from the repository root in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,release]'
python -m unittest discover -s tests -v
python Examples/Example_4_ExternalInputs.py
python -m build --outdir dist/0.1.3
python -m twine check --strict dist/0.1.3/fdeint-0.1.3.tar.gz dist/0.1.3/fdeint-0.1.3-py3-none-any.whl
```

Review the wheel and source archive before uploading. The wheel should contain
the `FDEint` package and its distribution metadata; the source archive also
includes documentation, examples, and regression tests. Local environments,
build directories, and unrelated source directories are excluded. PyPI requires
the normalized lowercase distribution name `fdeint` in the filenames and wheel
metadata directory (`fdeint-0.1.3.dist-info`). The import remains `FDEint`.
The version-specific directory should contain just the source archive and wheel
for 0.1.3. Use the full
`python -m build` command above, which creates the wheel from the source archive
in a fresh build directory.

Also test installation from the wheel in a separate virtual environment and
run the external-input example with the working directory outside this
repository, so the installed package is exercised:

```bash
python -m venv /tmp/fdeint-wheel-check
/tmp/fdeint-wheel-check/bin/python -m pip install dist/0.1.3/fdeint-0.1.3-py3-none-any.whl
cd /tmp
/tmp/fdeint-wheel-check/bin/python /absolute/path/to/FDEint/Examples/Example_4_ExternalInputs.py
```

Replace `/absolute/path/to/FDEint` with the repository path. Use a new temporary
environment path if `/tmp/fdeint-wheel-check` already exists. CUDA verification
requires a CUDA-enabled PyTorch installation and a compatible GPU.

## Manual upload after review

Return to the repository root and activate the environment with the release
tools. Confirm that `setup.py`, `CHANGELOG.md`, and the artifact filenames all
refer to 0.1.3. Configure a PyPI API token through Twine's supported credential
mechanisms; keep credentials out of the repository.

If publishing manually instead of using GitHub Actions, explicitly run:

```bash
python -m twine upload dist/0.1.3/fdeint-0.1.3-py3-none-any.whl dist/0.1.3/fdeint-0.1.3.tar.gz
```

Avoid a `dist/*` upload: older artifacts may be present. A published PyPI
version cannot be replaced with different files; if a correction is needed
after publishing, increment the version and build new artifacts.

After uploading, check the PyPI project page and test
`python -m pip install FDEint==0.1.3` in a fresh environment. Choose either
the GitHub release upload or a manual upload for each version.
