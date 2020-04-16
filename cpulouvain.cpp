#include <iostream>
#include <cstring>

using namespace std;


int main(int argc, char *argv[]) {
    int i = 1;
    bool show_assignment = false;
    float threshold = 0;
    string matrix_file;

    while (i < argc) {
        char* s = argv[i];
        if (strcmp(s, "-f")) {
            matrix_file = string(argv[i + 1]);
            i += 2;
        } else if (strcmp(s, "-g")) {
            threshold = strtof(argv[i + 1], NULL);
            i += 2;
        } else if (strcmp(s, "-v")) {
            show_assignment = true;
            i += 1;
        } else {
            exit(1);
        }
    }
}