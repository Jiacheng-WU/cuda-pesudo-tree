# GPU Parallelization over Pesudo Tree

## Setup & Run
```shell
# install pixi by
# $ curl -fsSL https://pixi.sh/install.sh | sh
# clone this repo and cd into this repo
pixi init
pixi install --all
pixi shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=x86_64-conda-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=x86_64-conda-linux-gnu-g++ \
      ..
make -j $(nproc)
./main
```
