#pragma once
#ifdef __INTELLISENSE__

//KERNEL_ARG2(grid, block) : <<< grid, block >>>
#define KERNEL_ARG2(grid, block)
//KERNEL_ARG3(grid, block, sh_mem) : <<< grid, block, sh_mem >>>
#define KERNEL_ARG3(grid, block, sh_mem)
//KERNEL_ARG4(grid, block, sh_mem, stream) : <<< grid, block, sh_mem, stream >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream)

#else

#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#endif