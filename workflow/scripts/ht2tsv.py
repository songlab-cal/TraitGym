import hail as hl
import sys

ht = hl.read_table(sys.argv[1])
print(ht.describe())
ht.export(sys.argv[2])
