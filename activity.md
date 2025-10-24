## Activity -- insulating sphere

The file [relaxation3d.py](relaxation3d.py) implements the algorithm discussed in the lecture to calculate the potential for simple boxes in 3D space. Your task is to extend this program to support spheres with constant charge or constant potential by writing a function `make_sphere(...)` that mirrors `make_box3d(...)`, and to compute and plot the charge density and electric field everywhere in space for a box at a constant, nonzero potential inside a sphere of constant surface charge.

You may utlize the utility functions `graph_slices` and `vector_field_plot` to aide in visualizing these fields.

The pieces of the solution can be found in [solution.py](solution.py). The full runnable program is compiled in [solution_complete.py](solution_complete.py).