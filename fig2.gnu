set xlabel 'Similarity parameter'
set ylabel 'Me'
set xrange[0:3.2]
set yrange[0:10]
set grid
unset key
p "data/M8.txt"  u 1:3 w l,    \
  "data/M10.txt" u 1:3 w l,   \
  "data/M12.txt" u 1:3 w l,   \
  "data/M20.txt" u 1:3 w l,   \
  "data/valid-M8.dat" u 1:2,  \
  "data/valid-M10.dat" u 1:2, \
  "data/valid-M12.dat" u 1:2, \
  "data/valid-M20.dat" u 1:2
