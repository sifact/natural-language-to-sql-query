"""Tests for the concise module guides required by the package layout."""

from pathlib import Path

PACKAGES = ("data", "database", "agent", "evaluation")
SOURCE_ROOT = Path(__file__).parents[1] / "src" / "nl_sql"


def test_given_pipeline_modules_when_guides_are_read_then_each_has_five_bullets() -> None:
    for package_name in PACKAGES:
        package_root = SOURCE_ROOT / package_name
        modules = {
            path.stem
            for path in package_root.glob("*.py")
            if path.name != "__init__.py"
        }
        guide_root = package_root / "guide"
        guide_names = {path.stem for path in guide_root.glob("*.md")}

        assert guide_names == modules

        for module_name in modules:
            lines = (guide_root / f"{module_name}.md").read_text().splitlines()
            assert lines[0] == f"# {module_name}.py"
            assert lines[1] == ""
            assert lines[2].endswith(".")
            assert lines[3] == ""
            assert len(lines[4:]) == 5
            assert all(line.startswith("* ") for line in lines[4:])
