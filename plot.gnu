set terminal pngcairo size 600,600
set output '/data/plot.png'
set xlabel 'Similarity parameter'
set ylabel 'Me'
grid on
set xrange[0:3.2]
set yrange[0:10]
p "output.txt" u 1:3 w l, "data/valid-M10.dat" u 1:2
