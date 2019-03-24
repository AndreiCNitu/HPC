## Lattice Boltzmann GPU timings

***
#### __Initial code__

+ `gcc -O3`
+ `-cl-opt-disable`

| Size      | 128 x 128 | 128 x 256 | 256 x 256 | 1024 x 1024 |
| --------- |:---------:|:---------:|:---------:|:-----------:|
| **Time**  | 76.4s     | 153.7s    | 617s      | 2419s       |

***
#### __Port rebound & collision to kernels__

+ `gcc -O3`
+ `-cl-opt-disable`

| Size      | 128 x 128 | 128 x 256 | 256 x 256 | 1024 x 1024 |
| --------- |:---------:|:---------:|:---------:|:-----------:|
| **Time**  | 48.0s     | 87s       | 300s      | 1200s       |

***
#### __Port av_vels to kernel, simple reduction in host__

+ `gcc -O3`
+ `-cl-opt-disable`

| Size      | 128 x 128 | 128 x 256 | 256 x 256 | 1024 x 1024 |
| --------- |:---------:|:---------:|:---------:|:-----------:|
| **Time**  | 56.4s     | 103s      | 418s      | 3097s       |
