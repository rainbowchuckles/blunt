set xlabel 'Similarity parameter'
set ylabel 'Me'
set xrange[0:3.2]
set yrange[0:10]
p "output.txt" u 1:3 w l, "data/valid-M10.dat" u 1:2
