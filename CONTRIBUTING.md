# Contributing

## Changelog entries

Each change a user will notice gets a file in `changelog.d/`, named after the
pull request and the kind of change, such as `412.fixed.md`. Open the pull
request first, so you know its number. A release collects the files into the
[changelog](CHANGELOG.md), one entry each, and links each entry to its pull
request.

| kind | use it when |
| --- | --- |
| `breaking` | code that uses `brighteyes` has to change |
| `added` | something new: a class, a function, a keyword, an option |
| `changed` | something that already existed behaves differently |
| `deprecated` | something still works but warns, and goes in a later release |
| `removed` | something is gone |
| `fixed` | something now works as its documentation says |
| `security` | a vulnerability is closed |

`towncrier` writes the file for you. Keep the text in single quotes, so the
shell leaves its backticks alone:

```bash
uv run towncrier create 412.fixed.md --content 'A timeout given to `read` is applied.'
```

A second change of the same kind in one pull request goes in `412.fixed.2.md`.

An entry says what changed, in a line or two, and names what a user would
type to reach it: a name the package exports as it is, and any other with
its module. It doesn't say why: that belongs in the pull request.

```markdown
Good: Add `timeout=` to `read`, which gives up when no reply comes in time.
Bad:  Improve read.
```

A change to something not released yet gets no entry of its own. Edit the
entry that added it instead, so the release says what users get.

A pull request nobody using `brighteyes` would notice, such as tests, CI, a
refactor or docs alone, needs no entry. A maintainer gives it the
`skip-changelog` label, and CI accepts it without one.
