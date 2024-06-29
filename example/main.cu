#include "src/head.cuh"
using namespace std;

TinyCache cache[horizon + 2];

// SharedMatrix shared;

// TempBigDual tempGibDual;
void debug1(double * shared)
{
    std::cout<<"shared\n";
    for(int i = 0 ; i < horizon * StateShape ; i ++)
    {
        for(int j = 0 ; j < horizon * StateShape ; j ++)
        {
            std::cout<<shared[i * horizon * StateShape + j]<<" ";
        }
        std::cout<<std::endl;
    }
}

void debug2(double * tempGibDual)
{
    std::cout<<"tempGibDual\n";
    for (int i = 0 ; i <  horizon * StateShape ; i ++)
    {
        std::cout<<tempGibDual[i]<<" ";
    }
}

int main()
{
    // shared.setZero();
    QCost Qtemp;
    RCost Rtemp;
    Qtemp << 1, 0, 0,
        0, 5, 0,
        0, 0, 0.1;
    Rtemp << 0.5, 0,
        0, 0.05;
    state final_state;
    final_state << 1.5, 1.5, 0;
    state init_state;
    init_state << 0 , 0 , 0;

    state temp ;
    temp<<1 , 1 , 1;
    control init_control;
    init_control<<1 , 1;

    cache[0].state1<<init_state;

    cache[0].FirstVarible.setZero();
    cache[0].FirstVarible.setZero();

    cache[0].LowerLeftDown1.setZero();
    cache[0].LowerLeftDown2.setZero();

    for (int i = 1; i < horizon + 1; i++)
    {
        cache[i].state1<<temp;
        cache[i].control<<init_control;
        cache[i].cache1.setZero();
        cache[i].cache2.setZero();
        cache[i].cache3.setZero();

        cache[i].SparseCache.setZero();

        // cache[i].cache.setZero();
        cache[i].h_A_RowIndices.setZero();
        cache[i].h_A_ColIndices.setZero();

        cache[i].LowerLeftDown1.setZero();
        cache[i].LowerLeftDown2.setZero();
    }
    cache[horizon + 1].FirstDual.setZero();


    tinytype *shared = (tinytype*)malloc(StateShape * horizon * StateShape * horizon * sizeof(tinytype));
    tinytype *tempbigDual = (tinytype *)malloc(StateShape * horizon * sizeof(tinytype));
    
    tiny_solve_cuda(cache  , tempbigDual , Qtemp , Rtemp , init_state , final_state);

    // debug1(shared);
    // debug2(tempbigDual);

  
}