# Nearest-neighbour distances — LUCAS+LfU+LfL Bavaria sample set

_Source: `/home/valerian/SGTPublication/Data/LUCAS_LFU_Lfl_00to23_Bavaria_OC.xlsx`_

- Rows in source file: **30,451** (many revisited POINTIDs share identical coordinates).
- Unique sample locations used for NN distance: **5,148**.
- Distances computed with `sklearn.neighbors.BallTree(metric="haversine")` × Earth radius 6371008.8 m.

## Summary statistics (metres)

| stat | metres |
|------|--------|
| min | 1.00 |
| 5th percentile | 24.73 |
| 25th percentile | 55.03 |
| median | 816.02 |
| mean | 1730.46 |
| 75th percentile | 2826.45 |
| 95th percentile | 6001.67 |
| 99th percentile | 8934.09 |
| max | 22292.84 |
| std (ddof=1) | 2177.10 |

## Tail shares

| threshold | fraction of locations whose NN is closer |
|-----------|-----------------------------------------|
| < 100 m | 34.32% |
| < 500 m | 43.51% |
| < 1 km  | 53.05% |
| < 2 km  | 63.99% |

_Plot:_ `nn_distance_histogram.png`
