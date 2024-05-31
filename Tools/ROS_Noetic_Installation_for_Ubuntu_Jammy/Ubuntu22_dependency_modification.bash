#!/bin/bash

## CMakeLists.txt
sed -i -e s/"-std=c++0x"/"-std=c++17"/g ./src/ddynamic_reconfigure/CMakeLists.txt
sed -i -e s/"COMPILER_SUPPORTS_CXX11"/"COMPILER_SUPPORTS_CXX17"/g ./src/geometry/tf/CMakeLists.txt
sed -i -e s/"c++11"/"c++17"/g ./src/geometry/tf/CMakeLists.txt
#sed -i -e s/"CMAKE_CXX_STANDARD 14"/"CMAKE_CXX_STANDARD 17"/g ./src/kdl_parser/kdl_parser/CMakeLists.txt
#sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./src/laser_geometry/CMakeLists.txt
#sed -i -e s/"c++11"/"c++17"/g ./src/resource_retriever/CMakeLists.txt
#sed -i -e s/"COMPILER_SUPPORTS_CXX11"/"COMPILER_SUPPORTS_CXX17"/g ./src/robot_state_publisher/CMakeLists.txt
#sed -i -e s/"c++11"/"c++17"/g ./src/robot_state_publisher/CMakeLists.txt
#sed -i -e s/"c++11"/"c++17"/g ./src/rqt_image_view/CMakeLists.txt
sed -i -e s/"CMAKE_CXX_STANDARD 14"/"CMAKE_CXX_STANDARD 17"/g ./src/urdf/urdf/CMakeLists.txt

sed -i -e s/"CMAKE_CXX_STANDARD 14"/"CMAKE_CXX_STANDARD 17"/g ./src/perception_pcl/pcl_ros/CMakeLists.txt
sed -i -e s/"c++14"/"c++17"/g ./src/perception_pcl/pcl_ros/CMakeLists.txt
#sed -i -e s/"CMAKE_CXX_STANDARD 11"/"CMAKE_CXX_STANDARD 17"/g ./src/laser_filters/CMakeLists.txt 

#sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./src/mavros/libmavconn/CMakeLists.txt
sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./src/mavros/mavros/CMakeLists.txt
sed -i -e s/"include(EnableCXX11)"/"add_compile_options(-std=c++17)"/g ./src/mavros/mavros_extras/CMakeLists.txt
sed -i -e s/"c++11"/"c++17"/g ./src/vrpn_client_ros/CMakeLists.txt

## rosconsole_log4cxx.cpp
sed -i '169c \ \ logger->addAppender(log4cxx::AppenderPtr(new ROSConsoleStdioAppender));' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '203c \ \ return &*log4cxx::Logger::getLogger(name);' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '219c \ \ auto repo = log4cxx::spi::LoggerRepositoryPtr(log4cxx::Logger::getLogger(ROSCONSOLE_ROOT_LOGGER_NAME)->getLoggerRepository());' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '355c log4cxx::AppenderPtr g_log4cxx_appender = {};' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '359c \ \ g_log4cxx_appender = log4cxx::AppenderPtr( new Log4cxxAppender(appender));' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '365c \ \ if(dynamic_cast<Log4cxxAppender*>(&*g_log4cxx_appender)->getAppender() == appender)' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '369,370c \ \ \ \ g_log4cxx_appender = log4cxx::AppenderPtr();' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '378c \ \ \ \ g_log4cxx_appender = log4cxx::AppenderPtr();' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp
sed -i '385c \ \ static_cast<log4cxx::spi::LoggerRepositoryPtr>(log4cxx::Logger::getRootLogger()->getLoggerRepository())->shutdown();' ./src/rosconsole/src/rosconsole/impl/rosconsole_log4cxx.cpp

## thread_test.cpp
sed -i '86i LOG4CXX_PTR_DEF(TestAppender);'  ./src/rosconsole/test/thread_test.cpp
sed -i '99c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/thread_test.cpp
sed -i '88i LOG4CXX_PTR_DEF(TestAppender);'  ./src/rosconsole/test/utest.cpp
sed -i '117i LOG4CXX_PTR_DEF(TestAppenderWithThrow);'  ./src/rosconsole/test/utest.cpp
sed -i '124c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '135c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '147c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '158c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '172c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '182c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '193c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '203c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '216c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '226c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '237c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '247c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '260c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '270c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '281c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '291c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '304c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '314c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '325c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '335c \    auto appender = TestAppenderPtr(new TestAppender); \\' ./src/rosconsole/test/utest.cpp
sed -i '359c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '580c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '600c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '634c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '657c \ \ auto appender = TestAppenderWithThrowPtr(new TestAppenderWithThrow);' ./src/rosconsole/test/utest.cpp
sed -i '682c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '702c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '733c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '770c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '790c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '821c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '852c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '869c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '905c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '924c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '954c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '971c \ \ auto appender = TestAppenderPtr(new TestAppender);' ./src/rosconsole/test/utest.cpp
sed -i '1027c \ \ \ \ &*log4cxx::Logger::getLogger(ROSCONSOLE_ROOT_LOGGER_NAME), level, str,' ./src/rosconsole/test/utest.cpp
sed -i '1042c \ \ \ \ &*log4cxx::Logger::getLogger(ROSCONSOLE_ROOT_LOGGER_NAME), level, str,' ./src/rosconsole/test/utest.cpp
sed -i '1057c \ \ \ \ &*log4cxx::Logger::getLogger(ROSCONSOLE_ROOT_LOGGER_NAME), level, str,' ./src/rosconsole/test/utest.cpp
sed -i '1072c \ \ \ \ &*log4cxx::Logger::getLogger(ROSCONSOLE_ROOT_LOGGER_NAME), level, str,' ./src/rosconsole/test/utest.cpp

## CMakeLists.txt
sed -i '10c find_package(Boost REQUIRED COMPONENTS regex system thread)' ./src/rosconsole/CMakeLists.txt
