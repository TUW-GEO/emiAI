//2D mesh script for ResIPy (run the following in gmsh to generate a triangular mesh with topograpghy)
Mesh.Binary = 0;//specify we want ASCII format
cl=0.26;//define characteristic length
//Define surface points
Point(1) = {-4.97,0.00,272.40,cl};//topography point
Point(2) = {0.00,0.00,272.40,cl};//electrode
Point(3) = {1.00,0.00,272.40,cl};//electrode
Point(4) = {2.00,0.00,272.40,cl};//electrode
Point(5) = {3.00,0.00,272.40,cl};//electrode
Point(6) = {4.00,0.00,272.40,cl};//electrode
Point(7) = {5.00,0.00,272.30,cl};//electrode
Point(8) = {5.90,0.00,272.30,cl};//electrode
Point(9) = {7.00,0.00,272.30,cl};//electrode
Point(10) = {7.90,0.00,272.30,cl};//electrode
Point(11) = {8.90,0.00,272.30,cl};//electrode
Point(12) = {9.90,0.00,272.40,cl};//electrode
Point(13) = {10.90,0.00,272.40,cl};//electrode
Point(14) = {12.00,0.00,272.40,cl};//electrode
Point(15) = {13.00,0.00,272.40,cl};//electrode
Point(16) = {14.00,0.00,272.50,cl};//electrode
Point(17) = {15.00,0.00,272.40,cl};//electrode
Point(18) = {16.00,0.00,272.40,cl};//electrode
Point(19) = {17.00,0.00,272.40,cl};//electrode
Point(20) = {17.90,0.00,272.40,cl};//electrode
Point(21) = {18.90,0.00,272.40,cl};//electrode
Point(22) = {19.90,0.00,272.30,cl};//electrode
Point(23) = {20.90,0.00,272.30,cl};//electrode
Point(24) = {21.90,0.00,272.40,cl};//electrode
Point(25) = {22.80,0.00,272.50,cl};//electrode
Point(26) = {23.90,0.00,272.70,cl};//electrode
Point(27) = {24.90,0.00,272.80,cl};//electrode
Point(28) = {25.90,0.00,272.80,cl};//electrode
Point(29) = {26.70,0.00,272.90,cl};//electrode
Point(30) = {27.80,0.00,272.90,cl};//electrode
Point(31) = {28.60,0.00,273.00,cl};//electrode
Point(32) = {29.80,0.00,273.20,cl};//electrode
Point(33) = {30.80,0.00,273.40,cl};//electrode
Point(34) = {31.80,0.00,273.50,cl};//electrode
Point(35) = {32.90,0.00,273.60,cl};//electrode
Point(36) = {33.80,0.00,273.70,cl};//electrode
Point(37) = {34.70,0.00,273.80,cl};//electrode
Point(38) = {35.90,0.00,273.80,cl};//electrode
Point(39) = {36.90,0.00,273.90,cl};//electrode
Point(40) = {37.90,0.00,274.00,cl};//electrode
Point(41) = {38.90,0.00,274.10,cl};//electrode
Point(42) = {39.90,0.00,274.30,cl};//electrode
Point(43) = {40.80,0.00,274.40,cl};//electrode
Point(44) = {41.90,0.00,274.60,cl};//electrode
Point(45) = {42.80,0.00,274.70,cl};//electrode
Point(46) = {43.80,0.00,274.80,cl};//electrode
Point(47) = {44.80,0.00,274.90,cl};//electrode
Point(48) = {45.80,0.00,275.10,cl};//electrode
Point(49) = {46.80,0.00,275.20,cl};//electrode
Point(50) = {47.80,0.00,275.40,cl};//electrode
Point(51) = {48.80,0.00,275.50,cl};//electrode
Point(52) = {49.80,0.00,275.60,cl};//electrode
Point(53) = {50.80,0.00,275.70,cl};//electrode
Point(54) = {51.70,0.00,275.90,cl};//electrode
Point(55) = {52.70,0.00,276.00,cl};//electrode
Point(56) = {53.70,0.00,276.10,cl};//electrode
Point(57) = {54.70,0.00,276.20,cl};//electrode
Point(58) = {55.70,0.00,276.40,cl};//electrode
Point(59) = {56.70,0.00,276.50,cl};//electrode
Point(60) = {57.70,0.00,276.60,cl};//electrode
Point(61) = {58.70,0.00,276.80,cl};//electrode
Point(62) = {59.70,0.00,276.90,cl};//electrode
Point(63) = {60.60,0.00,277.00,cl};//electrode
Point(64) = {61.60,0.00,277.10,cl};//electrode
Point(65) = {62.60,0.00,277.30,cl};//electrode
Point(66) = {63.60,0.00,277.40,cl};//electrode
Point(67) = {64.60,0.00,277.50,cl};//electrode
Point(68) = {65.60,0.00,277.60,cl};//electrode
Point(69) = {66.60,0.00,277.80,cl};//electrode
Point(70) = {67.60,0.00,277.90,cl};//electrode
Point(71) = {68.50,0.00,278.00,cl};//electrode
Point(72) = {69.50,0.00,278.10,cl};//electrode
Point(73) = {70.60,0.00,278.20,cl};//electrode
Point(74) = {75.57,0.00,278.20,cl};//topography point
//construct lines between each surface point
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,15};
Line(15) = {15,16};
Line(16) = {16,17};
Line(17) = {17,18};
Line(18) = {18,19};
Line(19) = {19,20};
Line(20) = {20,21};
Line(21) = {21,22};
Line(22) = {22,23};
Line(23) = {23,24};
Line(24) = {24,25};
Line(25) = {25,26};
Line(26) = {26,27};
Line(27) = {27,28};
Line(28) = {28,29};
Line(29) = {29,30};
Line(30) = {30,31};
Line(31) = {31,32};
Line(32) = {32,33};
Line(33) = {33,34};
Line(34) = {34,35};
Line(35) = {35,36};
Line(36) = {36,37};
Line(37) = {37,38};
Line(38) = {38,39};
Line(39) = {39,40};
Line(40) = {40,41};
Line(41) = {41,42};
Line(42) = {42,43};
Line(43) = {43,44};
Line(44) = {44,45};
Line(45) = {45,46};
Line(46) = {46,47};
Line(47) = {47,48};
Line(48) = {48,49};
Line(49) = {49,50};
Line(50) = {50,51};
Line(51) = {51,52};
Line(52) = {52,53};
Line(53) = {53,54};
Line(54) = {54,55};
Line(55) = {55,56};
Line(56) = {56,57};
Line(57) = {57,58};
Line(58) = {58,59};
Line(59) = {59,60};
Line(60) = {60,61};
Line(61) = {61,62};
Line(62) = {62,63};
Line(63) = {63,64};
Line(64) = {64,65};
Line(65) = {65,66};
Line(66) = {66,67};
Line(67) = {67,68};
Line(68) = {68,69};
Line(69) = {69,70};
Line(70) = {70,71};
Line(71) = {71,72};
Line(72) = {72,73};
Line(73) = {73,74};
//add points below surface to make a fine mesh region
Point(75) = {-4.97,0.00,247.40,cl*5.00};//base of smoothed mesh region
Point(76) = {-1.47,0.00,247.40,cl*5.00};//base of smoothed mesh region
Point(77) = {2.03,0.00,247.40,cl*5.00};//base of smoothed mesh region
Point(78) = {5.53,0.00,247.30,cl*5.00};//base of smoothed mesh region
Point(79) = {9.04,0.00,247.31,cl*5.00};//base of smoothed mesh region
Point(80) = {12.54,0.00,247.40,cl*5.00};//base of smoothed mesh region
Point(81) = {16.04,0.00,247.40,cl*5.00};//base of smoothed mesh region
Point(82) = {19.54,0.00,247.34,cl*5.00};//base of smoothed mesh region
Point(83) = {23.04,0.00,247.54,cl*5.00};//base of smoothed mesh region
Point(84) = {26.55,0.00,247.88,cl*5.00};//base of smoothed mesh region
Point(85) = {30.05,0.00,248.25,cl*5.00};//base of smoothed mesh region
Point(86) = {33.55,0.00,248.67,cl*5.00};//base of smoothed mesh region
Point(87) = {37.05,0.00,248.92,cl*5.00};//base of smoothed mesh region
Point(88) = {40.55,0.00,249.37,cl*5.00};//base of smoothed mesh region
Point(89) = {44.05,0.00,249.83,cl*5.00};//base of smoothed mesh region
Point(90) = {47.56,0.00,250.35,cl*5.00};//base of smoothed mesh region
Point(91) = {51.06,0.00,250.76,cl*5.00};//base of smoothed mesh region
Point(92) = {54.56,0.00,251.19,cl*5.00};//base of smoothed mesh region
Point(93) = {58.06,0.00,251.67,cl*5.00};//base of smoothed mesh region
Point(94) = {61.56,0.00,252.10,cl*5.00};//base of smoothed mesh region
Point(95) = {65.07,0.00,252.55,cl*5.00};//base of smoothed mesh region
Point(96) = {68.57,0.00,253.01,cl*5.00};//base of smoothed mesh region
Point(97) = {72.07,0.00,253.20,cl*5.00};//base of smoothed mesh region
Point(98) = {75.57,0.00,253.20,cl*5.00};//base of smoothed mesh region
//make lines between base of fine mesh region points
Line(74) = {75,76};
Line(75) = {76,77};
Line(76) = {77,78};
Line(77) = {78,79};
Line(78) = {79,80};
Line(79) = {80,81};
Line(80) = {81,82};
Line(81) = {82,83};
Line(82) = {83,84};
Line(83) = {84,85};
Line(84) = {85,86};
Line(85) = {86,87};
Line(86) = {87,88};
Line(87) = {88,89};
Line(88) = {89,90};
Line(89) = {90,91};
Line(90) = {91,92};
Line(91) = {92,93};
Line(92) = {93,94};
Line(93) = {94,95};
Line(94) = {95,96};
Line(95) = {96,97};
Line(96) = {97,98};

//Adding boundaries
//end of boundaries.
//Add lines at leftmost side of fine mesh region.
Line(97) = {1,75};
//Add lines at rightmost side of fine mesh region.
Line(98) = {74,98};
//compile lines into a line loop for a mesh surface/region.
Line Loop(1) = {97, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, -98, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45, -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1};

//Background region (Neumann boundary) points
cln=32.81;//characteristic length for background region
Point(99) = {-357.97,0.00,272.40,cln};//far left upper point
Point(100) = {-357.97,0.00,-236.80,cln};//far left lower point
Point(101) = {428.57,0.00,278.20,cln};//far right upper point
Point(102) = {428.57,0.00,-236.80,cln};//far right lower point
//make lines encompassing all the background points - counter clock wise fashion
Line(99) = {1,99};
Line(100) = {99,100};
Line(101) = {100,102};
Line(102) = {102,101};
Line(103) = {101,74};
//Add line loops and plane surfaces for the Neumann region
Line Loop(2) = {99, 100, 101, 102, 103, 98, -96, -95, -94, -93, -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, -78, -77, -76, -75, -74, -97};
Plane Surface(1) = {1, 2};//Coarse mesh region surface

//Adding polygons
//end of polygons.
Plane Surface(2) = {1};//Fine mesh region surface

//Make a physical surface
Physical Surface(1) = {2, 1};

//End gmsh script
