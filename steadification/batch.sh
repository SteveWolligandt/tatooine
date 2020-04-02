#!/usr/bin/fish
set neighbor_weights 1 1.2 1.5 1.7. 2.0 3.0
set penalties -0.1 -0.5 -1 -2 -3

for w in $neighbor_weights
  for p in $penalties
    ./steadification dg 0 -5 5 2 0.05 20 10 1 $w $p
    ./steadification dg 0 -5 5 2 0.05 40 20 1 $w $p
    ./steadification dg 0 -5 5 2 0.05 100 50 1 $w $p
  end
end

for w in $neighbor_weights
  for p in $penalties
    ./steadification sc 0 -3.14149 3.14149 2 0.05 20 20 0.95 $w $p
    ./steadification sc 0 -3.14149 3.14149 2 0.05 40 40 0.95 $w $p
    ./steadification sc 0 -3.14149 3.14149 2 0.05 100 100 0.95 $w $p
  end
end

for w in $neighbor_weights
  for p in $penalties
    ./steadification bou 15 -5 5 2 0.05 20 60 0.95 $w $p
    ./steadification bou 10 -5 5 2 0.05 20 60 0.95 $w $p
    ./steadification bou 5 -5 5 2 0.05 20 60 0.95 $w $p
    ./steadification bou 10 -10 10 2 0.05 20 60 0.95 $w $p
  end
end
