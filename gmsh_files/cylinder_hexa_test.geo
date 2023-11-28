// Gmsh project created on Thu Oct 12 11:44:14 2023
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 0, 0, 1.5, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 1, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Curve Loop(2) = {2};
//+
Plane Surface(1) = {1, 2};
//+
Recombine Surface{1};//+
Extrude {0, 0, 1} {
  Surface{1}; Layers {5}; Recombine;
}
