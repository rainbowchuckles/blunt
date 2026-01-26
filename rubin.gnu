set xrange[0:2.8]
set yrange[0:12]
p "rubin.txt" u 1:2 w l, "rubin-valid.txt" u 1:2
