#include "iostream"
#include "math.h"
using namespace std;


/* 学习率 */
double learning_rate = 1e-4;

/* 允许误差 */
double eps = 1e-8;


/* 将多元线性回归的过程装在黑匣子（类 MulLiReg）中 */
class MulLiReg {
public:
    /* 初始化设计矩阵
     * u个训练对象，v个特征 */
    MulLiReg(int u,int v):
    n(u),m(v){

        /* 初始化设计矩阵 */
        X = new double*[n];
        for(int i=0;i<n;i++){
            X[i] = new double[m+1];
            /* 设计矩阵中，每个训练对象的第0个特征值为1 */
            X[i][0] = 1;
        }

        /* 初始化向量 Y */
        Y = new double[n];

        /* 记录m个特征的最大值和最小值，
         * m+1 是为了使角标一致 */
        Max_X = new double[m + 1];
        Min_X = new double[m + 1];

        /*初始化偏回归向量*/
        theta = new double[m + 1];
    }

    /* 依次输入训练集 */
    void input(){
        for(int i=0;i<n;i++){
            for(int j=1;j<=m;j++)
                cin>>X[i][j];
            cin>>Y[i];
        }

    }

    /* 查看矩阵 (X|Y) */
    void disp_XY(){
        for(int k=0;k<m+1;k++)
            cout<<"X"<<k<<"\t";
        cout<<"Y"<<endl;
        for(int i=0;i<n;i++){
            for(int j=0;j<=m;j++)
                cout<<X[i][j]<<"\t";
            cout<<Y[i];
            cout<<endl;
        }
        cout<<endl;
    }

    /* 极差归一化 X 中的数据，使得X中的数据的范围都在[0,1]
     * 以此减少梯度下降的迭代次数 */
    void normalize(){

        /* 归一化第j特征所有的特征值,
         * x_ij = (x_ij-min) / (max-min)
         * 第0列全为1，无需归一化，故从第1列开始归一化 */
        for(int j=1;j<=m;j++){

            /*寻找第j列向量的最小值和最大值*/
            double min_x = X[0][j],max_x = X[0][j];
            for(int i=1;i<n;i++){
                if(X[i][j] > max_x) max_x = X[i][j];
                if(X[i][j] < min_x) min_x = X[i][j];
            }

            /* 记录第j列向量的最大值最小值，也即归一化的参数，
             * 便于"逆归一化"将数据变为原来的形式 */
            Min_X[j] = min_x;
            Max_X[j] = max_x;

            /* 开始归一化 X */
            for(int i=0;i<n;i++)
                X[i][j] = (X[i][j] - min_x) / (max_x - min_x);

        }

        /*归一化Y*/
        /*寻找Y向量中的最小值和最大值*/
        double min_y = Y[0],max_y = Y[0];
        for(int i=1;i<n;i++){
            if(Y[i]>max_y) max_y = Y[i];
            if(Y[i]<min_y) min_y = Y[i];
        }
        /* 开始归一化 Y */
        for(int i=0;i<n;i++)
            Y[i] = (Y[i]-min_y) / (max_y - min_y);


        /* 记录Y的最大值和最小值 */
        Min_Y = min_y;
        Max_Y = max_y;

    }


    /* 第i个训练对象假设函数的值 */
    double hypo_f(double *t,int i){
        double h = 0;
        for(int j=0;j<=m;j++){
            h += theta[j] * X[i][j];
        }
        return h;
    }

    /* 第j特征梯度函数的值 */
    double grad(double *t,int j){
        double g = 0;
        for(int i=0;i<n;i++){
            g += (hypo_f(t,i)-Y[i])*X[i][j];
        }
        g /= (n);
        return g;
    }


    /* 梯度下降 */
    void process(){


        /* 临时数组，用于记录上一次迭代的theta值 */
        double *t_l = new double[m + 1];

        /* 记录迭代是否完成 */
        bool f = true;

        /* 从 theta = o开始梯度下降 */
        for(int j=0;j<=m;j++){
            theta[j] = 0;
            t_l[j] = theta[j];
        }


        /* 开始梯度下降迭代 */
        do{
            /* theta 同步梯度下降 */
            for(int j=0;j<=m;j++)
                theta[j] -= learning_rate * grad(t_l,j);

            /* 检验是否收敛 */
            for(int j=0,cnt=0;j<=m;j++){
                if(abs(t_l[j]-theta[j])<=eps) cnt++;
                if(cnt==m+1) f = false;
            }

            /* 记录上一次迭代的theta */
            for(int j=0;j<=m;j++)
                t_l[j] = theta[j];

        }while(f);


    }

    /* 展示theta偏回归系数 */
    void disp_theta(){
        cout<<"θ:"<<endl;
        for(int j=0; j <= m; j++)
            cout << theta[j] << endl;
    }

    /* “逆归一化”，将theta转化为适用于源数据的形式 */
    void trans_theta(){

        /* 转化theta_0 */

        /* t为临时变量 */
        double t=0;
        for(int j=1;j<=m;j++){
            t += Min_X[j] * theta[j] / (Max_X[j] - Min_X[j]);
        }
        theta[0] = (Max_Y-Min_Y)*(theta[0] - t) + Min_Y;

        /* 转化theta_j */

        t = 0;
        for(int j=1;j<=m;j++){
            theta[j] = (Max_Y-Min_Y)*theta[j] / (Max_X[j]-Min_X[j]);
        }
    }

private:

    /* n个训练对象，m个特征 */
    int n,m;

    /* 设计矩阵 X */
    double **X;

    /* 向量 Y */
    double *Y;


    /* 记录X第j个特征的最大值(Max_X[j])和最小值(Min_X[j])
     * (为了角标一致，规定j>=1) */
    double *Max_X,*Min_X;

    /* 记录Y的最大值和最小值 */
    double Max_Y,Min_Y;

    /* 偏回归系数 */
    double *theta;
};

int main(){
    /* n个训练对象，m个特征 */
    int n,m;
    cin>>n>>m;

    MulLiReg *p_MulLiReg = new MulLiReg(n, m);

    /* 输入数据 */
    p_MulLiReg->input();

    /* 查看 矩阵(X|Y) */
    cout<<"数据(归一化前):"<<endl;
    p_MulLiReg->disp_XY();

    /* 归一化 */
    p_MulLiReg->normalize();

    /* 查看 矩阵(X|Y) */
    cout<<"数据（归一化后）："<<endl;
    p_MulLiReg->disp_XY();

    /* 开始梯度下降迭代 */
    p_MulLiReg->process();

    /* 查看得到的的偏回归系数(于归一化数据) */
    cout<<"偏回归系数向量(于归一化处理后的数据):"<<endl;
    p_MulLiReg->disp_theta();

    p_MulLiReg->trans_theta();

    /* 查看得到的的偏回归系数(于源数据) */
    cout<<"偏回归系数向量(于源数据):"<<endl;
    p_MulLiReg->disp_theta();

    return 0;
}
