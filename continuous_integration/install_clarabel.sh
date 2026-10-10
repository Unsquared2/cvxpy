#!/bin/bash
# Installs Clarabel from the Unsquared2 fork's GitHub release (bound and
# auxiliary-variable elimination, banded KKT solver) into the active
# environment.  Its wheels are not on PyPI, where "clarabel" is the upstream
# project; they are found through the release page.  Pythons without a
# matching wheel (free-threaded builds) build the release's sdist, which
# needs a Rust toolchain.  Keep the version in step with [tool.uv] in
# pyproject.toml.

CLARABEL_VERSION="0.11.1+unsquared.1"
uv pip install \
  --find-links "https://github.com/Unsquared2/Clarabel.rs/releases/expanded_assets/v${CLARABEL_VERSION}" \
  "clarabel==${CLARABEL_VERSION}"
python -c "import clarabel, sys; v = clarabel.__version__; print('clarabel', v); sys.exit(v != '${CLARABEL_VERSION}')"
