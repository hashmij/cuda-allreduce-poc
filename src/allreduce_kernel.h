#ifndef _ALLREDUCE_KERNEL_H_
#define _ALLREDUCE_KERNEL_H_

// All supported types
typedef int     CHAR_TYPE;
typedef int     INT_TYPE;
typedef float   FLOAT_TYPE;
typedef double  DOUBLE_TYPE;

// Currently used type
//typedef INT_TYPE TYPE;
typedef FLOAT_TYPE TYPE;

enum reduction_op {
    SUM=0,
    MUL=1
}reduction_op;


#if 0
enum TYPES {
    CHAR,
    INT,
    FLOAT,
    DOUBLE
};

#define typename(x) _Generic((x),               \
       char: CHAR,                               \
        int: INT,                                \
      float: FLOAT,                              \
     double: DOUBLE,                             \
    default: INT)
#endif




#endif
