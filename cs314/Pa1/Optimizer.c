/*
 *********************************************
 *  314 Principles of Programming Languages  *
 *  Spring 2017                              *
 *  Author: Ulrich Kremer                    *
 *  Student Version                          *
 *********************************************
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "InstrUtils.h"
#include "Utils.h"

int main()
{
	Instruction *head, *curr;

	head = ReadInstructionList(stdin);
	if (!head) {
		WARNING("No instructions\n");
		exit(EXIT_FAILURE);
	}

    curr = head;
    findUsed(current);
    prune(current);

    while(curr){
      Instruction * next = current->next;
      if(!curr->critical){
        curr->next->prev = curr->prev;
        curr->prev->next = curr->next;
      }
      curr = next;
    }

	if (head) 
		PrintInstructionList(stdout, head);
	
	return EXIT_SUCCESS;
}

void findUsed(Instruction * curr){
  while(curr && !found_instr){
    switch(curr->opcode)
      {
        case OUTPUTAI: 
          curr->critical = 1;
          Instruction *prev = curr;
          int reg = curr->field1;
          int loc = curr->field2;
          markLoad(curr, reg);
          markStore(curr, reg, loc);
          break;
        default;
      }
      curr = curr->next;
    }
  }
}

void markStore(Instruction * curr, int reg, int loc){
  int found_instr = 0;


}

void markLoad(){


}

void markArithmetic(){


}

