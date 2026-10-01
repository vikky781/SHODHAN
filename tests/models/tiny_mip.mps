* Toy MIP written for the SHODHAN tests (not a benchmark instance).
NAME          TINY_MIP
ROWS
 N  COST
 G  COVER1
 G  COVER2
 L  BUDGET
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    Y1        COST         4.0   COVER1       1.0
    Y1        BUDGET       3.0
    Y2        COST         6.0   COVER1       1.0
    Y2        COVER2       1.0   BUDGET       5.0
    Y3        COST         5.0   COVER2       1.0
    Y3        BUDGET       4.0
    MARKER                 'MARKER'                 'INTEND'
    S         COST         0.1   COVER1       0.5
    S         COVER2       0.5
RHS
    RHS       COVER1       1.0   COVER2       1.0
    RHS       BUDGET      10.0
BOUNDS
 BV BND       Y1
 UP BND       Y2           3.0
 UP BND       S            2.5
ENDATA
