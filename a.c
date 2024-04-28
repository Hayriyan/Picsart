#include <stdio.h>
#include "vector.h"

double mod( void )
{
	Vector *vec = (Vector*)ADDRESS;
	return (vec->x + vec->y);
}
