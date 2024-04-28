#ifndef VECTOR_H
#define VECTOR_H

typedef struct Vector
{
	int x;
	int y;
	double (*mod)( void );
}Vector;

#endif
