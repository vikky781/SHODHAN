* Toy LP written for the SHODHAN tests (not a benchmark instance).
NAME          TINY_LP
OBJSENSE
    MAX
ROWS
 N  PROFIT
 L  LABOR
 G  DEMAND
 E  BALANCE
 L  CAPACITY
COLUMNS
    X1        PROFIT       3.0   LABOR        2.0
    X1        DEMAND       1.0   BALANCE      1.0
    X2        PROFIT       5.0   LABOR        1.0
    X2        BALANCE     -1.0   CAPACITY     4.0
    X3        PROFIT      -0.5   DEMAND       1.0
    X3        CAPACITY     1.5
RHS
    RHS       LABOR       40.0   DEMAND       2.0
    RHS       BALANCE      0.0   CAPACITY    30.0
    RHS       PROFIT      -10.0
RANGES
    RNG       CAPACITY    12.0
BOUNDS
 UP BND       X1          10.0
 LO BND       X2           1.0
 FR BND       X3
ENDATA
