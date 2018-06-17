#include <stdio.h>
#include <signal.h>

int main(void) {
  printf("helloo\n");
  raise(9);
  return 0;
}

