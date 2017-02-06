// Author: John-Austen Francisco
// Date: 9 September 2015
//
// Preconditions: Appropriate C libraries
// Postconditions: Generates Segmentation Fault for
//                               signal handler self-hack

// Student name: Matthew Lee
// Ilab machine used: man.cs.rutgers.edu

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

void segment_fault_handler(int signum)
{
        printf("I am slain!\n");
        void * p = (void *) &signum; //Use the signum to construct a pointer to flag on stored stack
        p += 0x3c; //Increment pointer down to the stored PC
        *(int *)p += 0x6; //Increment value at pointer by length of bad instruction
}

int main()
{
        int r2 = 0;

        signal(SIGSEGV, segment_fault_handler);

        r2 = *( (int *) 0 );

        printf("I live again!\n");

        return 0;
}
