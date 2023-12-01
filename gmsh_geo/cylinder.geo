// Gmsh project created on Tue May 18 16:34:55 2021
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 00, 00, 0.6, 0, 2*Pi};
//+
Circle(2) = {0, 00, 00, 0.4, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Curve Loop(2) = {2};
//+
Plane Surface(1) = {1, 2};
//+
Extrude {0, 0, 10} {
  Curve{2}; Curve{1}; Layers {1}; Recombine;
}
//+
Curve Loop(5) = {6};
//+
Curve Loop(6) = {4};
//+
Plane Surface(4) = {5, 6};
//+
Physical Surface("Brime", 7) = {2};
//+
Physical Surface("AirSide", 8) = {3};
//+
Physical Surface("Top", 9) = {1};
//+
Physical Surface("Bottom", 10) = {4};
//+
Surface Loop(1) = {2, 4, 3, 1};
//+
Volume(1) = {1};
//+
Physical Volume(11) = {1};