echo "build ORBSlam: cpu_num="
read cpu_num

echo "Configuring and building DBoW2 ..."
cd DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$cpu_num

echo "Configuring and building g2o ..."
cd ../../g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$cpu_num

echo "Configuring and building Sophus ..."
cd ../../Sophus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$cpu_num

echo "Configuring and building ORB_SLAM3 ..."
cd ../../ORB_SLAM3
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$cpu_num
