#!/usr/bin/fish

set n 1
set coverage 0.99
set seedline_resolution 2
set stepwidth 0.1
#set num_iterations 100
set num_iterations 1000 2000 5000
set steadification ./steadification

for num_its in $num_iterations
  set tail $num_its $seedline_resolution $stepwidth $coverage
  for x in (seq $n)
    ###############################################################################
    ## doublegyre
    #echo $steadification dg 0 -5 5 $tail
    #$steadification dg 0 -5 5 $tail
    #echo $steadification dg 0 -10 10 $tail
    #$steadification dg 0 -10 10 $tail
    ##echo $steadification dg 0 -20 20 $tail
    ##$steadification dg 0 -20 20 $tail

    ###############################################################################
    ## fixed time doublegyre
    #echo $steadification fdg 0 -5 5 $tail
    #$steadification fdg 0 -5 5 $tail
    #echo $steadification fdg 0 -10 10 $tail
    #$steadification fdg 0 -10 10 $tail
    ##echo $steadification fdg 0 -20 20 $tail
    ##$steadification dfg 0 -20 20 $tail

    ##############################################################################
    # RBC
    #$steadification rbc 2000 0 10 $tail
    #$steadification rbc 2000 0 20 $tail
    #$steadification rbc 2010 -10 10 $tail
    #$steadification rbc 2020 -10 0 $tail
    #$steadification rbc 2020 -20 0 $tail

    ##############################################################################
    # Cavity
    #$steadification cav 0 0 10 $tail
    $steadification cav 10 -5 5 $tail
    $steadification cav 20 -10 0 $tail
    $steadification cav 10 -10 10 $tail

    ##############################################################################
    # boussinesq
    echo $steadification bou 15 -5 5 $tail
    $steadification bou 15 -5 5 $tail
    echo $steadification bou 10 0 10 $tail
    $steadification bou 10 0 10 $tail
    echo $steadification bou 20 -10 0 $tail
    $steadification bou 20 -10 0 $tail
  end
end
