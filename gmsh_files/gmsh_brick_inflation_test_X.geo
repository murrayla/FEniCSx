// Gmsh project created on Wed Nov  1 10:08:07 2023
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {0, 1, 1, 1.0};
//+
Point(4) = {0, 0, 1, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Point(5) = {0, -0.5, -0.5, 1.0};
//+
Point(6) = {0, 1.5, -0.5, 1.0};
//+
Point(7) = {0, 1.5, 1.5, 1.0};
//+
Point(8) = {0, -0.5, 1.5, 1.0};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 5};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Curve Loop(2) = {7, 8, 5, 6};
//+
Curve Loop(3) = {3, 4, 1, 2};
//+
Plane Surface(1) = {2, 3};
Recombine Surface{1};//+
Extrude {1, 0, 0} {
  Surface{1}; Layers {1}; Recombine;
}
