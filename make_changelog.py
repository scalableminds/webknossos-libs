import datetime
import re
import sys

# Read existing changelog
with open("Changelog.md") as f:
    changelog_lines = list(s.rstrip() for s in f)

# Determine new version
this_version = sys.argv[1]
today = datetime.date.today()
today_str = f"{today.strftime('%Y')}-{today.strftime('%m')}-{today.strftime('%d')}"

# Determine last version
matches = re.finditer(r"^## \[(.*)\].*$", "\n".join(changelog_lines), re.MULTILINE)
last_version = next(matches)[1]

# Stop the script if new version is already the latest
if last_version == this_version:
    print(f"Version {this_version} is already up-to-date.")
    sys.exit(0)

# Find line with "## Unreleased" heading
unreleased_idx = next(
    i for i, line in enumerate(changelog_lines) if line.startswith("## Unreleased")
)

# Find line with the last release (i.e. "## x.y.z" heading)
last_release_idx = next(
    i
    for i, line in enumerate(changelog_lines)
    if line.startswith("## ") and i > unreleased_idx
)

# Clean up unreleased notes (i.e. remove empty sections)
unreleased_notes = changelog_lines[(unreleased_idx + 2) : last_release_idx]

empty_unreleased_sections = [
    (i, line)
    for i, line in enumerate(unreleased_notes)
    if line.startswith("### ") and unreleased_notes[i + 1] == ""
]
for i, section in empty_unreleased_sections[::-1]:
    # Remove section heading
    del unreleased_notes[i]
    # Remove empty line
    del unreleased_notes[i]


# Update changelog
lines_to_insert = [
    "## Unreleased",
    f"[Commits](https://github.com/scalableminds/voxelytics/compare/{this_version}...HEAD)",
    "",
    "### Breaking Changes",
    "",
    "### Added",
    "",
    "### Changed",
    "",
    "### Fixed",
    "",
    "",
    f"## [{this_version}](https://github.com/scalableminds/voxelytics/releases/tag/{this_version}) - {today_str}",
    f"[Commits](https://github.com/scalableminds/voxelytics/compare/{last_version}...{this_version})",
]
changelog_lines = (
    changelog_lines[:unreleased_idx]
    + lines_to_insert
    + unreleased_notes
    + changelog_lines[(last_release_idx):]
    + [""]
)

with open("Changelog.md", "wt") as f:
    f.write("\n".join(changelog_lines))
