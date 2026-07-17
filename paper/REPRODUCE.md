# Command-Only Reproduction Guide

## Checkout and verify the frozen source

```sh
git checkout kepler-paper-v1
test "$(git rev-parse kepler-v3.6-confirm1)" = "f9fc22d85cd52863087afbb3f924466a63b30ba2"
uv sync --frozen
```

## Install the isolated paper-rendering overlay

```sh
command -v pdftoppm
uv run --with reportlab==4.4.9 python -c "import reportlab; assert reportlab.Version == '4.4.9'"
```

## Lightweight scientific verification without training

```sh
uv run python paper/scripts/verify_results.py
uv run python paper/scripts/verify_frozen_artifacts.py
```

## Rebuild every figure, table, index, and manuscript

```sh
uv run --with reportlab==4.4.9 python paper/scripts/build_all.py
```

## Test and hash verification

```sh
uv run --with reportlab==4.4.9 python paper/scripts/verify_package.py --rebuild --with-tests
```

## Optional LaTeX compilation

```sh
cd paper
tectonic manuscript.tex
cd ..
```
