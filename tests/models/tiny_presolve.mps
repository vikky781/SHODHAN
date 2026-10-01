* Toy LP written for the SHODHAN tests (not a benchmark instance).
* Contains: an empty row, an empty column, a fixed column, a singleton row,
* a doubleton equation and a few ordinary rows, so presolve has work to do.
NAME          TINY_PRESOLVE
ROWS
 N  COST
 L  EMPTYROW
 G  SINGLE
 E  DOUBLE
 G  MAIN1
 L  MAIN2
 G  MAIN3
COLUMNS
    FIXED     COST         3.0   MAIN1        1.0
    X2        COST         1.0   SINGLE       2.0
    X3        COST         2.0   DOUBLE       1.0
    X3        MAIN1        1.0   MAIN3        1.0
    X4        COST         1.0   DOUBLE       1.0
    X4        MAIN2        1.0
    X5        COST         4.0   MAIN1        2.0
    X5        MAIN2       -1.0   MAIN3        1.0
    X6        COST        -1.0   MAIN2        1.0
    X6        MAIN3        1.0
    EMPTYCOL  COST         2.0
RHS
    RHS       EMPTYROW     1.0   SINGLE       4.0
    RHS       DOUBLE      10.0   MAIN1        8.0
    RHS       MAIN2        9.0   MAIN3        3.0
BOUNDS
 FX BND       FIXED        2.0
 UP BND       X5           6.0
 UP BND       X6           5.0
 UP BND       EMPTYCOL     5.0
 LO BND       EMPTYCOL     1.0
ENDATA
