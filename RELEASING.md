# Releasing FDEint

Version 0.1.2 adds support for external inputs to the solver. The
version is defined in `setup.py`; PyPI reads its description from `README.md`.
The commands below prepare and check a release locally. Upload is a separate
step to run deliberately after reviewing the resulting artifacts.

## Prepare and verify

Run these commands from the repository root in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test,release]'
python -m unittest discover -s tests -v
python Examples/Example_4_ExternalInputs.py
python -m build --outdir dist/0.1.2
python -m twine check --strict dist/0.1.2/fdeint-0.1.2.tar.gz dist/0.1.2/fdeint-0.1.2-py3-none-any.whl
```

Review the wheel and source archive before uploading. The wheel should contain
the `FDEint` package and its distribution metadata; the source archive also
includes documentation, examples, and regression tests. Local environments,
build directories, and unrelated source directories are excluded. PyPI requires
the normalized lowercase distribution name `fdeint` in the filenames and wheel
metadata directory (`fdeint-0.1.2.dist-info`). The import remains `FDEint`.
The version-specific directory should contain just the source archive and wheel
for 0.1.2. Use the full
`python -m build` command above, which creates the wheel from the source archive
in a fresh build directory.

Also test installation from the wheel in a separate virtual environment and
run the external-input example with the working directory outside this
repository, so the installed package is exercised:

```bash
python -m venv /tmp/fdeint-wheel-check
/tmp/fdeint-wheel-check/bin/python -m pip install dist/0.1.2/fdeint-0.1.2-py3-none-any.whl
cd /tmp
/tmp/fdeint-wheel-check/bin/python /absolute/path/to/FDEint/Examples/Example_4_ExternalInputs.py
```

Replace `/absolute/path/to/FDEint` with the repository path. Use a new temporary
environment path if `/tmp/fdeint-wheel-check` already exists. CUDA verification
requires a CUDA-enabled PyTorch installation and a compatible GPU.

## Upload after review

Return to the repository root and activate the environment with the release
tools. Confirm that `setup.py`, `CHANGELOG.md`, and the artifact filenames all
refer to 0.1.2. Configure a PyPI API token through Twine's supported credential
mechanisms; keep credentials out of the repository.

To publish the reviewed 0.1.2 artifacts to PyPI, explicitly run:

```bash
python -m twine upload dist/0.1.2/fdeint-0.1.2-py3-none-any.whl dist/0.1.2/fdeint-0.1.2.tar.gz
```

Avoid a `dist/*` upload: older artifacts may be present. A published PyPI
version cannot be replaced with different files; if a correction is needed
after publishing, increment the version and build new artifacts.

After uploading, check the PyPI project page and test
`python -m pip install FDEint==0.1.2` in a fresh environment. Commit and tag the reviewed release
according to the repository's normal workflow.
