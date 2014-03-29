// trnlsp.cpp test trust region non-linear solver
#include "trnlsp.h"

void
extendet_powell (int n, const double *x, double *f)
{
    MKL_INT i;

    for (i = 0; i < n / 4; i++)
    {
        f[4 * i] = x[4 * i] + 10.0 * x[4 * i + 1];
        f[4 * i + 1] = 2.2360679774998 * (x[4 * i + 2] - x[4 * i + 3]);
        f[4 * i + 2] = (x[4 * i + 1] - 2.0 * x[4 * i + 2]) * (x[4 * i + 1] - 2.0 * x[4 * i + 2]);
        f[4 * i + 3] = 3.1622776601684 * (x[4 * i] - x[4 * i + 3]) * (x[4 * i] - x[4 * i + 3]);
    }
    return;
}

void test_jacobi(void)
{
	double eps = 1e-13;
	std::vector<double> x(2,1.);

	auto f = [](const double* x, double*y) { 
		y[0] = x[0]*x[0]; 
		y[1] = x[0]*x[1]; 
	};
	mkl::jacobi dF(2, 2, f, eps);
	std::vector<double> df(4);
	dF(&x[0], &df[0]);
	size_t iter;
	iter = dF.iterations();

	ensure (fabs(df[0] - 2) < eps);
	ensure (fabs(df[1] - 1) < eps);
	ensure (fabs(df[2] - 1) < eps);
	ensure (fabs(df[3] - 0) < eps);
}

int main()
{
	test_jacobi();

	std::vector<double> x(4);

	mkl::trnlsp powell(4, 4, &x[0]);
	
	powell
	.absolute(1e-5)
	.function([](const double* x, double* f) {
			extendet_powell(4, x, f);
		});

	powell.init(&x[0]);
	while (powell.step())
		;

	return 0;
}