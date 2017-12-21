set xlabel 'Refinement'
set ylabel 'Deflection'

set key bottom

# Tweak the aspect ratio a little
set size 0.8,1

# Plot only integral tics (the '1' is the increment between tics)
set xtics 1

#set terminal postscript eps color "Arial,20" linewidth 6
set terminal pdf color linewidth 6  

#set title 'Deflection'
set output "cantilever_deflection.pdf"

plot  [][0:24] 'deflection.data' using 1:2 with linespoints title "1-1", \
      'deflection.data' using 1:3 with linespoints title "2-1", \
      'deflection.data' using 1:4 with linespoints title "2-2", \
      'deflection.data' using 1:5 with linespoints title "3-2", \
      'deflection.data' using 1:6 with linespoints title "3-3"


