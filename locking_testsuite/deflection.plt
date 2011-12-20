set xlabel 'Refinement'
set ylabel 'Deflection'

#set terminal postscript eps color "Arial,20" linewidth 6
set terminal pdf color linewidth 6  

set title 'Deflection of a cantilever beam'
set output "cantilever_deflection.pdf"

plot  'average_deflection_3.8462e+05' using 1:4 with lines title "mu_c = mu",  \
      'average_deflection_0' using 1:4 with lines title "mu_c = 0"

set title 'Lateral deflection of a cantilever beam'
set output "cantilever_lateral_deflection.pdf"

plot  'average_deflection_3.8462e+05' using 1:3 with lines title "mu_c = mu",  \
      'average_deflection_0' using 1:3 with lines title "mu_c = 0"

