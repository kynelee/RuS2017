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

void findUsed(Instruction * curr);
void markStore(Instruction * curr, int reg, int loc);
void markLoad(Instruction * curr, int reg);
void markArithmetic(Instruction * curr, int reg);


int main()
{
	Instruction *head, *curr;

	head = ReadInstructionList(stdin);
	if (!head) {
		WARNING("No instructions\n");
		exit(EXIT_FAILURE);
	}

    curr = head;

    findUsed(curr);

    while(curr){
      Instruction * next = curr->next;
      if(!(curr->critical == '1')){
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
  int found_instr = 0;
  while(curr && !found_instr){
    switch(curr->opcode)
      {
        case OUTPUTAI: 
          curr->critical = '1';
          // Instruction *prev = curr;
          int reg = curr->field1;
          int loc = curr->field2;
          markLoad(curr, reg);
          markStore(curr, reg, loc);
          break;
        default:
          break;
      }
      curr = curr->next;
    }
}

void markStore(Instruction * curr, int reg, int loc){
  int found_instr = 0;
  while(!found_instr && curr){
    switch(curr->opcode){
      case STOREAI:
        if(curr->field2 == reg && curr->field3==loc && !(curr->critical == '1')){
          curr->critical = '1';
          markArithmetic(curr, curr->field1);
          return;
        }
        break;
      default:
        break;
    }
    curr = curr->prev;
  }
}

void markLoad(Instruction * curr, int reg){
  int found_instr = 0;
  while(!found_instr && curr){
    switch(curr->opcode){
      case LOADI:
        if(curr->field2 == reg && !(curr->critical == '1')){
          curr->critical = '1';
          return;
        }
      default:
        break;
    }
    curr = curr->prev;
  }
}

void markArithmetic(Instruction * curr, int reg){
  int found_instr = 0;
  while(!found_instr && curr){
    switch(curr->opcode){
      case LOADI:
        if(curr->field2 == reg){
          curr->critical = '1';
          return;
        }
      case LOADAI:
        if(curr->field3 == reg){
          curr->critical = '1';
          markStore(curr, curr->field1, curr->field2);
          return;
        }
      case ADD:
      case DIV:
      case SUB:
      case MUL:
        if(curr->field3 == reg && !(curr->critical == '1')){
          curr->critical = '1';
          markArithmetic(curr, curr->field1);
          markArithmetic(curr, curr->field2);
          return;
        }
      default:
        break;
    }
    curr = curr->prev;
  }

}

