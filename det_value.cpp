double det_value(double **det, int n) {
    double v = 0;
    int *ps = new int[n];
    for(int i=0;i<n;i++)
        ps[i] = i;
    do {
        int cnt = 0;
        for (int i = n - 1; i > 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                if (ps[j] > ps[i])
                    cnt++;
            }
        }
        double pro = 1;
        for (int i = 0; i < n; i++)
            pro *= det[i][ps[i]];
        if (cnt % 2 == 1)
            pro = -pro;
        v += pro;

    } while(next_permutation(ps,ps+n));
    return v;
}
