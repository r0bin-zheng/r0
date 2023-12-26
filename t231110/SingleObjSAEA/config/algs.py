# 算法配置以及代号

# r1
# surr: globe(kriging + rbf) + local(kriging)
# phase: global + local
# change: global -> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r1 = {}

# r2
# surr: globe(kpls + rbf) + local(kriging)
# phase: global + local
# change: global -> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r2 = {}

# r3
# surr: globe(kriging + rbf) + local(rbf)
# phase: global + local
# change: global -> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r3 = {}

# r4
# surr: globe(kpls + rbf) + local(rbf)
# phase: global + local
# change: global -> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r4 = {}

# r5
# surr: globe(kriging + rbf) + local(kriging)
# phase: global + local
# change: global <-> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r5 = {}

# r6
# surr: globe(kpls + rbf) + local(kriging)
# phase: global + local
# change: global <-> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r6 = {}

# r7
# surr: globe(kriging + rbf) + local(rbf)
# phase: global + local
# change: global <-> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r7 = {}

# r8
# surr: globe(kpls + rbf) + local(rbf)
# phase: global + local
# change: global <-> local
# select: global(pareto best from min of surr and mae of all surr) + local(min)
r8 = {}

# kriging1
# surr: kriging
# phase: global
# select: min
k1 = {}

# rbf1
# surr: rbf
# phase: global
# select: min
k1 = {}

# kpls1
# surr: kpls
# phase: global
# select: min
k1 = {}

# ego1
# surr: kriging
# phase: global
# select: ei