int main1() {
    vector<double> c = {1.0, 2.0, 3.0};
    double x = 0.5;
    double result = legval(x, c);
    cout << "Result of legval: " << result << endl;
    return 0;
}

int main2() {
    MatrixXd c(4, 3);
    c << 1, 2, 3,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12;
    int cnt = 1;
    MatrixXd result = legder(c, cnt);
    cout << "Result of legder:\n" << result << endl;
    return 0;
}