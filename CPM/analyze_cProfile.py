import pstats
p = pstats.Stats('output.txt')
p.sort_stats('cumulative').print_stats(10)