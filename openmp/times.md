## Lattice Boltzmann timings

***
#### __Initial code__
+ gcc -O0

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 200s          | 1622s          | 6663s          |

+ gcc -O3

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 58.8s          | 477s           | 2033s         |

+ gcc -Ofast

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 32.7s          | 265s           | 1151s          |

+ icc -O3

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 43.2s          | 350s           | 1523s           |

+ icc -Ofast

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 26.3s          | 212s           | 1010s           |

***
#### __Serial optimisations__

+ Merge `rebound()` and `collision()`

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 100s          | 200s           | 300s           |

+ Merge `propagate()`, `rebound()`, `collision()`, `av_vels()`
+ Pointer swap in `timestep()`

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 100s          | 200s           | 300s           |

+ Merge `propagate()`, `rebound()`, `collision()`, `av_vels()`
+ Pointer swap in `timestep()`
+ Store intermediate values in `const` vars -> 1 read + 1 write

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 100s          | 200s           | 300s           |

+ AoS -> SoA, aligned memory access, (some) `const`
+ Only `rebound()` and `collision()` are merged!

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 100s          | 200s           | 300s           |

+ AoS -> SoA, aligned data access, `restrict`, all `const`
+ Pointer swap, 1 loop to rule them all

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 100s          | 200s           | 300s           |

+ AoS -> SoA, `restrict`, all `const`, aligned data access
+ Pointer swap, 1 loop to rule them all
+ Save relaxation step results in `const`,
write to `tmp_cells` at the end

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 100s          | 200s           | 300s           |

***
#### __Parallel optimisations__

+ OpenMP `parallel for`, reductions

| Cores     |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **1**     | 100s          | 200s           | 300s           |
| **2**     | 100s          | 200s           | 300s           |
| **3**     | 100s          | 200s           | 300s           |
| **4**     | 100s          | 200s           | 300s           |
| **5**     | 100s          | 200s           | 300s           |
| **6**     | 100s          | 200s           | 300s           |
| **7**     | 100s          | 200s           | 300s           |
| **8**     | 100s          | 200s           | 300s           |
| **9**     | 100s          | 200s           | 300s           |
| **10**    | 100s          | 200s           | 300s           |
| **11**    | 100s          | 200s           | 300s           |
| **12**    | 100s          | 200s           | 300s           |
| **13**    | 100s          | 200s           | 300s           |
| **14**    | 100s          | 200s           | 300s           |
| **15**    | 100s          | 200s           | 300s           |
| **16**    | 100s          | 200s           | 300s           |

+ NUMA-aware implementation (parallel init)
+ set `OMP_PROC_BIND=true`

| Cores     |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **1**     | 100s          | 200s           | 300s           |
| **2**     | 100s          | 200s           | 300s           |
| **3**     | 100s          | 200s           | 300s           |
| **4**     | 100s          | 200s           | 300s           |
| **5**     | 100s          | 200s           | 300s           |
| **6**     | 100s          | 200s           | 300s           |
| **7**     | 100s          | 200s           | 300s           |
| **8**     | 100s          | 200s           | 300s           |
| **9**     | 100s          | 200s           | 300s           |
| **10**    | 100s          | 200s           | 300s           |
| **11**    | 100s          | 200s           | 300s           |
| **12**    | 100s          | 200s           | 300s           |
| **13**    | 100s          | 200s           | 300s           |
| **14**    | 100s          | 200s           | 300s           |
| **15**    | 100s          | 200s           | 300s           |
| **16**    | 100s          | 200s           | 300s           |

##### Experiments:
- `schedule()`
- `OMP_PLACES`
- `OMP_PROC_BIND`
- non-parallel `accelerate_flow()`

***
#### __Final times on 16 cores__

| Size      |  128 x 128    |  256 x 256     |  1024 x 1024   |
| --------- |:-------------:|:--------------:|:--------------:|
| **Time**  | 100s          | 200s           | 300s           |
