## Overview

A minimal implementation of Monte Carlo Tree Search (MCTS) in C++17.

---

MCTS is one of the machine learning techniques at the heart of AlphaGo/Zero, the first computer program to beat the world champion in a game of Go.

It’s elegant and easy to understand, can be used with little or no domain knowledge, and has succeeded on difficult problems (not only in the game area).

## Documentation

[A presentation from C++Day 2019](https://github.com/morinim/documents/tree/master/mcts_intro) (Parma - Italy)

## Setting up the build

```shell
mkdir -p build
cd build/
cmake ..
```

To suggest a specific compiler you can write:

```shell
CXX=clang++ cmake ..
```

You're now ready to build using the traditional `make` system. All the output files will be stored in the `build/` directory (out of source build).

## License

[Mozilla Public License v2.0](https://www.mozilla.org/MPL/2.0/) (also available in the accompanying `LICENSE` file).
