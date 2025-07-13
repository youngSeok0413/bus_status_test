bus status test

cmake
<br>requirements : opencv 4.xx
<br>mkdir build
<br>cd build
<br>cmake ..

road or the other
<br>bgr image -> hsv image(hue(color), saturation(purity), value(light or darkness)) 
<br> -> bitmasking(target to road)

warp perspective and rectify
<br>point vector(user, lt -> rt -> rb -> lb) -> get target form -> get perspective matrix
<br> => warp perspective
