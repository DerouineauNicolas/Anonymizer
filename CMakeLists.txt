cmake_minimum_required(VERSION 2.8)
project( Anonymizer )
find_package( OpenCV REQUIRED )
#FFMPEG and OpenCV is assumed to be in /usr/local/*
INCLUDE_DIRECTORIES(/usr/local/include/)
link_directories(/usr/local/lib/)
INCLUDE_DIRECTORIES(/usr/include/)
link_directories(/usr/lib/)

add_executable( Anonymizer main.cpp )
target_link_libraries( Anonymizer ${OpenCV_LIBS})
