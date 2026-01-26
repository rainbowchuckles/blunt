set xlabel 'Similarity parameter'
set ylabel 'Me'
set xrange[0:15]
set yrange[0:12]
set grid
unset key
p "data/5-15.dat"  u 1:3 w l,    \
  "data/10-15.dat" u 1:3 w l,   \
  "data/15-15.dat" u 1:3 w l,   \
  "data/20-15.dat" u 1:3 w l     
