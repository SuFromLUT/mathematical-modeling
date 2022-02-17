/*EWM*/
#include "iostream"
#include "algorithm"
#include "math.h"
using namespace std;


void disp(double **matrix,int n,int m){
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
            cout<<matrix[i][j]<<" ";
        cout<<endl;
    }
    cout<<endl;
}

/*data normalization*/
double **normalize(double **x,bool *is_p,int n,int m){
    double **norm = new double*[n];
    for(int i=0;i<n;i++)
        norm[i] = new double[m];
    for(int j=0;j<m;j++){
        double maxv = x[0][j],minv = x[0][j];
        for(int i=1;i<n;i++){
            maxv = max(maxv,x[i][j]);
            minv = min(minv,x[i][j]);
        }
        bool f = is_p[j];
        for(int i=0;i<n;i++){
            if(f)
                norm[i][j] = 0.998*(x[i][j]-minv)/(maxv-minv) + 0.002;
            else
                norm[i][j] = 0.998*(maxv-x[i][j])/(maxv-minv) + 0.002;
        }
    }
    return norm;
}

/* proportion */
double **get_prop(double **norm,int n,int m){
    double **prop = new double*[n];
    for(int i=0;i<n;i++)
        prop[i] = new double[m];
    for(int j=0;j<m;j++){
        double s = 0;
        for(int i=0;i<n;i++)
            s += norm[i][j];
        for(int i=0;i<n;i++)
            prop[i][j] = norm[i][j] / s;
    }
    return prop;
}


/* entropy */
double *get_e(double **prop,int n,int m){
    double k = -1/log(n);
    double *e = new double[m];
    for(int j=0;j<m;j++){
        double s = 0;
        for(int i=0;i<n;i++)
            s += prop[i][j]*log(prop[i][j]);
        e[j] = k * s;
    }
    return e;
}

/* Information entropy redundancy */
double *get_d(double *e,int m){
    double *d = new double[m];
    for(int j=0;j<m;j++)
        d[j] = 1-e[j];
    return d;
}

/* weight of indicators */
double *get_w(double **x,bool *is_p,int n,int m){
    double *w = new double[m];
    /* S1, data normalization */
    double **norm = normalize(x,is_p,n,m);

    /* S2, proportion */
    double **prop = get_prop(norm,n,m);

    /* S3, entropy */
    double *e = get_e(prop,n,m);

    /* S4, Information entropy redundancy */
    double *d = get_d(e,m);

    /* S5, weight of indicators */
    double d_sum = 0;
    for(int j=0;j<m;j++)
        d_sum += d[j];
    for(int j=0;j<m;j++){
        w[j] = d[j] / d_sum;
    }
    return w;
}

int main(){
    int n,m;
    cin>>n>>m;
    double **x = new double*[n];
    bool *is_p = new bool[m];

    /*S0,Input Data*/
    for(int i=0;i<n;i++)
        x[i] = new double[m];
    for(int j=0;j<m;j++)
        cin>>is_p[j];
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            cin>>x[i][j];

    /* weight vector */
    double *w = get_w(x,is_p,n,m);
    cout<<"w="<<endl;
    for(int j=0;j<m;j++){
        cout<<w[j]<<endl;
    }

    /* score */
    double **norm = normalize(x,is_p,n,m);
    cout<<"s="<<endl;
    for(int i=0;i<n;i++){
        double s = 0;
        for(int j=0;j<m;j++){
            s += w[j] * norm[i][j];
        }
        cout<<s<<endl;
    }
    return 0;
}
