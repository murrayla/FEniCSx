// Gmsh project created on Tue Oct 17 13:40:13 2023
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 10};
//+
Point(2) = {1, 0, 0, 10};
//+
Point(3) = {1, 1, 0, 10};
//+
Point(4) = {0, 1, 0, 10};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Surface(1) = {1};
Recombine Surface{1};//+
Extrude {0, 0, 1} {
  Surface{1}; Layers {1}; Recombine;
}

