#include <iostream>
#include <ctime>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

void My_SVD(const Matrix2f &F, Matrix2f &U, Matrix2f &sigma, Matrix2f &V) {
	Matrix2f C, A;
	float c, s, t, d; // For Jacobi and Givens

	C.noalias() = F.transpose()*F;

	// Jacobi (C,V,sigma1,sigma2)
	if (C(1, 0) != 0) {
		float tau = (C(1, 1) - C(0, 0)) / (2 * C(1, 0));
		if (tau > 0) {
			t = tau - hypotf(1, tau);
		}
		else {
			t = tau + hypotf(1, tau);
		} // In the case tau = 0, this sets t = 1
		c = 1 / hypotf(1, t);
		s = t*c;
	}
	else {
		c = 0;
		s = 1;
	} 
	V(0, 0) = c;
	V(0, 1) = -s;
	V(1, 0) = s;
	V(1, 1) = c;

	// Take square root of eigenvalues from previous step
	sigma(0, 0) = sqrt(pow(c, 2)*C(0, 0) + pow(s, 2)*C(1, 1) + 2 * c*s*C(1, 0));
	sigma(1, 1) = sqrt(pow(c, 2)*C(1, 1) + pow(s, 2)*C(0, 0) - 2 * c*s*C(1, 0));
	sigma(1, 0) = sigma(0, 1) = 0;

	// Order eigenvalues
	if (sigma(1, 1) > sigma(0, 0)) {
		float ph = sigma(0, 0);
		sigma(0, 0) = sigma(1, 1);
		sigma(1, 1) = ph;

		//Switch columns of V
		ph = V(0, 0);
		V(0, 0) = V(0, 1);
		V(0, 1) = ph;

		ph = V(1, 0);
		V(1, 0) = V(1, 1);
		V(1, 1) = ph;
	}

	// A = FV
	A.noalias() = F*V;

	// Givens rotation
	d = hypotf(A(0, 0), A(1, 0));
	c = 1;
	s = 0;
	if (d != 0) {
		t = 1 / d;
		c = A(0,0)*t;
		s = -A(1, 0)*t;
	}
	U(0, 0) = c;
	U(0, 1) = s;
	U(1, 0) = -s;
	U(1, 1) = c;

	// Fixes sign of sigma
	sigma(1, 1) = A(0, 1)*s + A(1, 1)*c; 

	// Enforce sign conventions
	if (V.determinant() < 0) {
		V(0, 1) = -V(0, 1);
		V(1, 1) = -V(1, 1);
		sigma(1, 1) = -sigma(1, 1);
	}
}

void My_Polar(const Matrix3f &F, Matrix3f &R, Matrix3f &S) {
	Matrix3f G;
	float c, s, d;

	R = Matrix3f::Identity();
	S = F;

	float tol = .000001f; // Maximum error tolerance
	int max_it = 10; // Maximum iteration number (if error is below tol)

	float error = max(max(abs(S(1, 0) - S(0, 1)), abs(S(2, 0) - S(0, 2))), abs(S(2, 1) - S(1, 2)));

	int it = 0;
	while ((it < max_it) || (error > tol)) {
		for (int i = 0; i < 2; i++) {
			for (int j = i + 1; j <= 2; j++) {
				// Givens rotation
				d = hypotf(S(i, i) + S(j, j), S(j, i) - S(i, j));
				c = (S(i, i) + S(j, j)) / d;
				s = (S(j, i) - S(i, j)) / d;

				G = Matrix3f::Identity();
				G(i, i) = G(j, j) = c;
				G(i, j) = -s;
				G(j, i) = s;
				
				// Update R and S
				R = R*G;
				S = G.transpose()*S;
			}
		}
		it = it + 1;
		error = max(max(abs(S(1, 0) - S(0, 1)), abs(S(2, 0) - S(0, 2))), abs(S(2, 1) - S(1, 2)));
	}
}

int  main()
{
	srand((unsigned int) time(0));
	Matrix2f F = Matrix2f::Random();
	Matrix2f U = Matrix2f::Random();
	Matrix2f V = Matrix2f::Random();
	Matrix2f sigma = Matrix2f::Random();

	Matrix3f F2 = Matrix3f::Random();
	Matrix3f R = Matrix3f::Random();
	Matrix3f S = Matrix3f::Random();

	My_Polar(F2, R, S);
	My_SVD(F, U, sigma, V);
}

/*
cout << "S: " << endl << S << endl;
cout << "R: " << endl << R << endl;
cout << "Det R: " << R.determinant() << endl;
cout << "RTR: " << endl << R.transpose()*R << endl;
cout << "Check: " << endl << F2 - R*S << endl;
*/

/*
cout << "Det U: " << U.determinant() << endl;
cout << "UTU: " << endl << U*U.transpose() << endl;
cout << "Det V: " << V.determinant() << endl;
cout << "VTV: " << endl << V*V.transpose() << endl;
cout << "Sigma: " << endl << sigma << endl;
cout << "UEVT" << endl << U*sigma*V.transpose() << endl;
cout << "Check Difference: " << endl << F - U*sigma*V.transpose() << endl;
*/

/*
sigma(0, 0) = 7.12;
sigma(1, 1) = -7.11;
sigma(1, 0) = sigma(0, 1) = 0;

float theta1 = ((float)rand() / (RAND_MAX + 1)) * 2 * 3.1415926535;
float theta2 = ((float)rand() / (RAND_MAX + 1)) * 2 * 3.1415926535;

U(0, 0) = U(1, 1) = cos(theta1);
U(0, 1) = -sin(theta1);
U(1, 0) = sin(theta1);

V(0, 0) = V(1, 1) = cos(theta2);
V(0, 1) = -sin(theta2);
V(1, 0) = sin(theta2);

F.noalias() = U*sigma*V.transpose();

cout << "Sigma: " << endl << sigma << endl;
cout << "F: " << endl << F << endl;

My_SVD(F, U, sigma, V);

cout << "Sigma: " << endl << sigma << endl;
cout << "F: " << endl << F << endl;
cout << "Det U: " << endl << U.determinant() << endl;
cout << "Det V: " << endl << V.determinant() << endl;
cout << "UTU: " << endl << U.transpose()*U << endl;
cout << "VTV: " << endl << V.transpose()*V << endl;
*/