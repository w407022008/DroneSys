#pragma once
#include <termios.h>
#include <sstream>
#include <vector>
#include "ros/ros.h"

# define U_KEY_NONE -1
# define U_KEY_SPACE 32
# define U_KEY_LEFT 37
# define U_KEY_UP 38
# define U_KEY_RIGHT 39
# define U_KEY_DOWN 40

# define U_KEY_0 48
# define U_KEY_1 49
# define U_KEY_2 50
# define U_KEY_3 51
# define U_KEY_4 52
# define U_KEY_5 53
# define U_KEY_6 54
# define U_KEY_7 55
# define U_KEY_8 56
# define U_KEY_9 57

# define U_KEY_A 97
# define U_KEY_D 100
# define U_KEY_E 101
# define U_KEY_F 102
# define U_KEY_H 104
# define U_KEY_K 107
# define U_KEY_L 108
# define U_KEY_M 109
# define U_KEY_O 111
# define U_KEY_P 112
# define U_KEY_Q 113
# define U_KEY_R 114
# define U_KEY_S 115
# define U_KEY_T 116
# define U_KEY_U 117
# define U_KEY_V 118
# define U_KEY_W 119
# define U_KEY_X 120
# define U_KEY_Y 121
# define U_KEY_Z 122

int  input_flag;
char buff;

using namespace std;

class KeyboardEvent{
    void getch()// get keyboard input. We need all instance to use the same function to catch the keyboard
    {
        input_flag = -1; //if not key is pressed, then return -1 for the keyboard_input. reset this flag every time
        fd_set set;
        struct timeval timeout;
        int rv;
        buff = 0;
        int len = 1;
        int filedesc = 0;
        FD_ZERO(&set);
        FD_SET(filedesc, &set);

        timeout.tv_sec = 0;
        //timeout.tv_usec = 1000;//
        timeout.tv_usec = 1000;
        rv = select(filedesc + 1, &set, NULL, NULL, &timeout);

        struct termios old = {0};
        if (tcgetattr(filedesc, &old) < 0)
                ROS_ERROR("tcsetattr()");
        old.c_lflag &= ~ICANON;
        old.c_lflag &= ~ECHO;
        old.c_cc[VMIN] = 1;
        old.c_cc[VTIME] = 0;
        if (tcsetattr(filedesc, TCSANOW, &old) < 0)
                ROS_ERROR("tcsetattr ICANON");

        if(rv == -1){
                ROS_ERROR("select");
        }
        else if(rv == 0)
        {
                //{ROS_INFO("-----");
        }
        else
        {read(filedesc, &buff, len);
        input_flag = 1;}

        old.c_lflag |= ICANON;
        old.c_lflag |= ECHO;
        if (tcsetattr(filedesc, TCSADRAIN, &old) < 0)
                ROS_ERROR ("tcsetattr ~ICANON");
    }
public:
    KeyboardEvent()
    {
        input_flag = -1;
        buff = 0;
    }
    char GetPressedKey()
    {
        char key;
        if(input_flag>-1)//if keyboard is pressed, then perform the callbacks
        {
            key = buff;
        }else{
        key = U_KEY_NONE;
        }
        return key;
    }
    void RosWhileLoopRun() // this should be placed inside the ros while loop
    {
        getch();
    }
};
