#pragma once
#include <Eigen/Dense>

typedef double tinytype;  


const int StateShape = 3;
const int ControlShape = 2;
const double T = 1;
const int horizon = 3;

using Eigen::Matrix;



    typedef Matrix<tinytype, StateShape , 1> state;
    typedef Matrix<tinytype, ControlShape , 1> control;

    typedef Matrix<tinytype, ControlShape + StateShape , 1> convergence;
    // typedef Matrix<tinytype, StateShape, 1> initial;
    typedef Matrix<tinytype, StateShape, 1> final_state;

    typedef Matrix<tinytype, (StateShape * 2 + ControlShape), 1> varible;
    typedef Matrix<tinytype, StateShape + ControlShape, StateShape + ControlShape> Hessian;
    typedef Matrix<tinytype, StateShape + ControlShape, 1> gradient;
    typedef Matrix<tinytype, StateShape, 1> equality;

    typedef Matrix<tinytype,  StateShape, StateShape + ControlShape> JB; 
    typedef Matrix<tinytype,  StateShape, StateShape > StateJB;   
    typedef Matrix<tinytype,  StateShape, ControlShape > ControlJB;   


    typedef Matrix<tinytype,  StateShape, StateShape > QCost;  

    typedef Matrix<tinytype,  ControlShape, ControlShape> RCost;  

    typedef Matrix<tinytype,  StateShape * 2 + ControlShape,   StateShape * 2 + ControlShape > FinalMatrix;   
 
    typedef Matrix<tinytype,   StateShape * 2 + ControlShape , 1 > FinalColumn;   
    

    // typedef Matrix<tinytype,  horizon*StateShape , horizon * StateShape > SharedMatrix;

    typedef Matrix<tinytype,  horizon*StateShape , 1 > SharedFirstTemp;  


    typedef Matrix<tinytype, StateShape, StateShape > temp; 

    typedef Matrix<tinytype, StateShape + ControlShape, 1 > FirstPhaseVarible;

    typedef Matrix<tinytype, StateShape, 1 > FirstPhaseDual;


    typedef Matrix<tinytype,  horizon * StateShape, 1 > TempBigDual;


    typedef Matrix<tinytype,   StateShape, StateShape  > UpperDualCache;

    // typedef Matrix<tinytype,   StateShape * 2, StateShape * 2  > TestUpperDualCache;

     typedef Matrix<tinytype,   StateShape *StateShape * 3 , 1 > SparseUpperDualCache;

    typedef  Matrix<int,   StateShape + 1, 1> RowIndices;

    typedef  Matrix<int,    StateShape * StateShape * 3, 1> ColIndices;  

    // typedef Matrix<int , horizon + 1 , 1> INDEX;

////// for debug:
    // typedef Matrix <double , horizon * StateShape * StateShape * 3 , 1>
    // DebugSparse; 

    typedef  Matrix<double,    StateShape ,  StateShape * 3> BigCache;  



    typedef struct
    {
        state state1;
        control control;

        gradient gradient;
        Hessian Hessian ;


        JB JB1;
        JB JB2;
        JB LowerLeftDown1;
        JB LowerLeftDown2;

   
        FirstPhaseVarible FirstVarible;
        FirstPhaseDual FirstDual;

    
        // TestUpperDualCache cache;

        SparseUpperDualCache SparseCache;

        RowIndices h_A_RowIndices;
        ColIndices h_A_ColIndices;

        int nnz = 0;

///// for debug
        equality equality;

//////
///// some new try
    UpperDualCache cache1;
    UpperDualCache cache2;
    UpperDualCache cache3;

    BigCache bigcache;


    } TinyCache;




