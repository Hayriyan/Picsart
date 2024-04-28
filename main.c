#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include "vector.h"

#define SIZE  4096

int main(void)
{
	char command[255];
	Vector vec = {.x = 10, .y = 20};
	sprintf(command, "gcc -c a.c -DADDRESS=%p", &vec);
	 //printf("%s\n", command);
	if (system(command))
	{
		perror("System");
		exit(EXIT_FAILURE);
	}
	if (system("chmod 777 a.o "))
	{
		perror("System");
		exit(EXIT_FAILURE);
	}

	int fd = open("a.o", O_RDONLY);
	//int size = 4096;
	void *m = mmap(NULL, SIZE, PROT_READ | PROT_EXEC, MAP_SHARED, -1, 0);

	if (m == (void *)-1)
	{
		perror("Can't allocate !");
		exit(1);
	}
	char *cast = (char *)m;
	int i = 0;
	while (i < SIZE)
	{
		if (*cast == 85)
		{
			double (*func)(void) = (double (*)(void))cast;
			printf("%f\n", func());
			break;
		}
		cast++;
		i++;
	}
	//close(fd);
}
