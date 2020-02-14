// This is a C implementation of the trueanomaly function to increase speed.

#include <math.h>
#include <stdlib.h>
#include "trueanomaly.h"
#include <stdio.h>

int trueanomaly(double *M, int n, double ecc, double *nu, int niterationmax, double tol)
{
    // Set upper limit for eccentricity
    if (ecc > 0.99)
        ecc = 0.99;

    // Newton-Raphson method
    for (int i = 0; i < n; i++)
    {
        double E = M[i];
        double E0 = M[i];
        int niteration = 0;

        while (fabs(E - E0) > tol || niteration == 0)
        {
            E0 = E;

            double ff = E - ecc * sin(E) - M[i];
            double dff = 1 - ecc * cos(E);

            // Use Newton method
            E = E0 - ff / dff;

            niteration += 1;
            if (niteration >= niterationmax)
                return -1;
        }
        // Compute true anomaly
        nu[i] = 2. * atan(sqrt((1. + ecc) / (1. - ecc)) * tan(E / 2.));
        // printf("Index=%d , Value=%f\n", i, nu[i]);
    }

    return 0;
}