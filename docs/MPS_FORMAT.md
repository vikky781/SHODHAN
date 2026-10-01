# MPS conventions

This describes exactly what `read_mps_*` and `write_mps` do. Where MPS
dialects disagree, this is the choice SHODHAN makes.

## Sections

`NAME`, `OBJSENSE`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, `ENDATA`.
Section keywords are case-insensitive; names are case-sensitive. Lines whose
first character is `*`, and blank lines, are skipped. `NAME` may be omitted.
`ENDATA` is required.

Unsupported sections give an error naming the line: `QUADOBJ`, `QMATRIX`,
`QSECTION`, `QCMATRIX` ("QPS not yet supported"), `SOS`, `INDICATORS`,
`OBJNAME`. Any other keyword is an "unknown section".

## Fixed versus free format

Free format splits fields on whitespace. Fixed format uses character columns
(1-based): field 1 at 2-3, field 2 at 5-12, field 3 at 15-22, field 4 at 25-36,
field 5 at 40-47, field 6 at 50-61. Fixed format is the only way to have a
**space inside a name**.

The format is auto-detected (`MpsFormat::Auto`): a file is read as fixed iff
every data line in ROWS, COLUMNS, RHS, RANGES and BOUNDS follows the column
layout (with numeric fields where numbers belong) **and** at least one name
contains a space. Otherwise it is read as free format, which gives the same
result for fixed-layout files without spaces. `MpsFormat::Fixed` and
`MpsFormat::Free` force a format. In ROWS, a fixed-format name runs to the end
of the line. `MARKER` lines are always read by whitespace.

A data line may start in column 1 in free format. A line starting in column 1
is a section header only if it has a single token, or starts with `NAME` or
`OBJSENSE`.

## Numbers

Decimal and exponent forms are accepted, including `D`/`d` exponent letters
(`1.0D+2`) and a signed `Inf`/`Infinity`. Negative zero is read as `+0`.
NaN and hexadecimal floats are rejected. Parsing uses the C locale.
A matrix coefficient or cost must be finite. Bounds and right-hand sides of
magnitude 1e30 or more are treated as infinite and stored as `+/-kInf`.

## Objective

- The first `N` row is the objective. Further `N` rows are ignored with a
  warning, together with every entry, RHS and RANGES value that refers to them.
- `OBJSENSE` is accepted as a section (`OBJSENSE` then `MAX`/`MIN` on the next
  line) or inline (`OBJSENSE MAX`), also as `MAXIMIZE`/`MINIMIZE`. The default
  is minimize.
- An RHS entry on the objective row sets `offset = -value`.

## Rows, RHS, RANGES

| Row type | Without RANGES | RANGES value R                              |
|----------|----------------|---------------------------------------------|
| `L`      | [-inf, rhs]    | [rhs - abs(R), rhs]                         |
| `G`      | [rhs, +inf]    | [rhs, rhs + abs(R)]                         |
| `E`      | [rhs, rhs]     | R > 0: [rhs, rhs + R]; R < 0: [rhs - abs(R), rhs]; R = 0: [rhs, rhs] |

The right-hand side defaults to 0. RANGES and RHS may appear in either order.
A RANGES value on a row whose right-hand side is infinite is ignored.
Only the first RHS, RANGES and BOUNDS set is used; entries of other sets are
ignored with a warning. In free format the set name may be omitted.

## Columns and integer markers

Columns appear contiguously in `COLUMNS`. A repeated (row, column) entry is an
error that names the line. A column that reappears after other columns is an
error. Columns between `'MARKER' 'INTORG'` and `'MARKER' 'INTEND'` are
integer. An unclosed block ends at the end of `COLUMNS` with a warning.

## Bounds

The default bounds are [0, +inf]. An integer column inside a marker block with
no bound entries is [0, +inf] (it is **not** made binary).

| Type | Effect                                             |
|------|----------------------------------------------------|
| `UP` | upper = value; if value < 0 and lower is 0, lower becomes -inf (warning) |
| `LO` | lower = value                                      |
| `FX` | lower = upper = value                              |
| `FR` | lower = -inf, upper = +inf                         |
| `MI` | lower = -inf (upper unchanged)                     |
| `PL` | upper = +inf                                       |
| `BV` | binary: bounds [0, 1], type Binary                 |
| `LI` | lower = value, type Integer                        |
| `UI` | upper = value (same rule as `UP`), type Integer    |
| `SC` | rejected: "semi-continuous (SC) bounds are not supported" |

A Binary column whose bounds are changed away from [0, 1] by a later entry
becomes Integer. A column with lower > upper is read, with a warning.

## Errors

Every parse error is reported as `file:line: message: 'offending text'`. A
missing `ENDATA` is reported at the last non-blank line. Malformed input never
crashes the reader.

## Writer

`write_mps` emits free format with shortest round-trip number formatting
(`std::to_chars`), one entry per line, using the set names `RHS`, `RNG`, `BND`.
Names default to `R<i>` and `C<j>` when the model has none. Rules:

- Free rows are written as `L` rows with right-hand side 1e30.
- A column without entries gets one objective line (cost 0) so that it exists.
- Binary columns are written inside integer markers with a `BV` bound.
- Maximize is written as an `OBJSENSE` section.
- If any name contains a space, the file is written in fixed format instead;
  then names must be at most 8 characters and numbers at most 12 characters,
  otherwise writing fails with an error and nothing is written.
- A ranged row is stored as a right-hand side plus a width. Bounds that are not
  exactly representable that way (for arbitrary doubles) come back within a few
  units in the last place, not bit-identical. Bounds such as multiples of 1/8
  are exact.
- A model that fails `LpModel::validate()` (for example lower > upper) is not
  written.
