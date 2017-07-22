cmake_minimum_required(VERSION 2.8)
project( testApp )
find_package( OpenCV REQUIRED )


add_executable( test test.cpp )
target_link_libraries( test ${OpenCV_LIBS} )

add_executable( train train.cpp )
target_link_libraries( train ${OpenCV_LIBS} )


