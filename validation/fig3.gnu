set xlabel 'Similarity parameter'
set output '/data/fig3.png'
set ylabel 'Me'
set terminal pngcairo size 600,600
set xrange[0:15]
set yrange[0:12]
set grid
unset key
p "data/5-15.dat"  u 1:3 w l,    \
  "data/10-15.dat" u 1:3 w l,   \
  "data/15-15.dat" u 1:3 w l,   \
  "data/20-15.dat" u 1:3 w l     
