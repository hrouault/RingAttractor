/*
 * Ring attractor is a software which simulates ring attractor models of
 * various connectivity profiles.
 *
 * Copyright © 2016 Howard Hughes Medical Institute
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the organization nor the
 * names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Howard Hughes Medical Institute ''AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL Howard Hughes Medical Institute BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#include <cmath>
#include <errno.h>
#include <cstring>
#include <sys/time.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "dyna.hpp"

gsl_rng* rng;

void simu(double* curstate, double* con, double* input, double* buf,
          fft_t* fft, parasw_t* par);

void simu(double* curstate, double* con, double* input, double* buf,
          fft_t* fft, parasw_t* par)
{
    uint16_t dim = par->nbpts;
    uint16_t dimc = par->nbpts / 2 + 1;
    for (size_t i = 0; i < dim; ++i) {
        curstate[i] = 0;
        input[i] = 0;
    }
    curstate[0] = 10.0;
    curstate[1] = 10.0;
    curstate[dim - 1] = 10.0;

    /* init_connect_delta(con, par); */
    /* init_connect_wta(con, par); */
    /* init_connect_cosine(con, par); */
    init_connect(con, par);

    /* FILE* fconnect = fopen("connectivity.dat", "w"); */
    /* if (!fconnect) { */
    /*     printf("cannot open file: connectivity.dat\n"); */
    /*     exit(EXIT_FAILURE); */
    /* } */
    /* for (size_t i = 0; i < dim; ++i) { */
    /*     fprintf(fconnect, "%f\n", con[i]); */
    /* } */
    /* fclose(fconnect); */

    for (size_t i = 0; i < dim; ++i) {
        fft->in[i] = con[i];
    }
    fftw_execute(fft->pland);
    for (size_t i = 0; i < dimc; ++i) {
        fft->fftcon[i][0] = fft->out[i][0];
        fft->fftcon[i][1] = fft->out[i][1];
    }
    fft->connect = con;

    init_sw(curstate, input, buf, fft, par);
    if (curstate[0] > 1e6) return;
    double max = 0;
    double min = 1e6;
    for (size_t i = 0; i < dim; ++i) {
        if (curstate[i] > max) max = curstate[i];
        if (curstate[i] < min) min = curstate[i];
    }
    if (fabs(min - max) < 1e-4) return;
    /* FILE* finit = fopen("init.dat", "w"); */
    /* if (!finit) { */
    /*     printf("cannot open file: init.dat\n"); */
    /*     exit(EXIT_FAILURE); */
    /* } */
    /* for (size_t i = 0; i < dim; ++i) { */
    /*     fprintf(finit, "%f\n", curstate[i]); */
    /* } */
    /* fclose(finit); */

    /* uint16_t nbmax = check_1max(curstate, par); */
    /* if (nbmax > 1) return; */

    double bw = 2 * M_PI * bump_width(curstate, par) / par->nbpts;
    /* printf("Bump width: %f\n", bw); */
    if (bw > M_PI / 4) return;
    if (bw < M_PI / 16) return;

    /* init_input(input, &par); */

    jump_vs_flow_simu(60 * M_PI / 180.0, curstate, input, buf, fft, par);
    jump_vs_flow_simu(90 * M_PI / 180.0, curstate, input, buf, fft, par);
    jump_vs_flow_simu(120 * M_PI / 180.0, curstate, input, buf, fft, par);
    jump_vs_flow_simu(150 * M_PI / 180.0, curstate, input, buf, fft, par);

    /* noisy_simu(curstate, input, buf, fft, par, rng); */

    /* end_trial(curstate, &par); */

    /* jump_vs_flow(maxs); */
}

void input_diagram(double* curstate, double* con, double* input, double* buf,
                   fft_t* fft, parasw_t* par, char** argv)
{
    uint16_t dim = par->nbpts;
    uint16_t dimc = par->nbpts / 2 + 1;

    if (std::strcmp("delta", argv[1]) == 0) {
        par->alpha = 3.0;
        par->beta = 20.0;
        par->D = 0.10;

        init_connect_delta(con, par);
    } else if (strcmp("cosine", argv[1]) == 0) {
        par->J0 = -0.2;
        par->J1 = 0.15;

        init_connect_cosine(con, par);
    } else {
        printf("model not recognized\n");
        exit(EXIT_FAILURE);
    }

    /* init_connect(con, par); */
    FILE* fconnect = fopen("connect.dat", "w");
    if (!fconnect) {
        printf("cannot open file: connect.dat\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < dim; ++i) {
        fprintf(fconnect, "%f\n", con[i]);
    }
    fclose(fconnect);

    for (size_t i = 0; i < dim; ++i) {
        fft->in[i] = con[i];
    }
    fftw_execute(fft->pland);
    for (size_t i = 0; i < dimc; ++i) {
        fft->fftcon[i][0] = fft->out[i][0];
        fft->fftcon[i][1] = fft->out[i][1];
    }
    fft->connect = con;

    jump_vs_flow_input(curstate, input, buf, fft, par, argv);
}

int main(int argc, char *argv[])
{
    gsl_rng_env_setup();
    rng = gsl_rng_alloc (gsl_rng_default);
    /* struct timeval tv; */
    /* gettimeofday(&tv, 0); */
    /* unsigned long int seed = tv.tv_sec + tv.tv_usec; */
    /* gsl_rng_set(rng, seed); */
    parasw_t par;

    uint16_t dim = 256;
    uint16_t dimc = dim / 2 + 1;
    par.nbpts = dim;

    fft_t fft;
    fft.n = dim;
        
    par.dt = 1.0e-3;
    par.trelax = 10.0;
    par.dx = 2 * M_PI / dim;

    size_t sd = sizeof(double);
    size_t sc = sizeof(fftw_complex);

    /* Initialization of the Fourier transform */
    double* in = static_cast<double*>(fftw_malloc(dim * sd));
    fftw_complex* out = static_cast<fftw_complex*>(fftw_malloc(dimc * sc));
    fftw_plan fftr = fftw_plan_dft_r2c_1d(dim, in, out, FFTW_MEASURE);
    fftw_plan fftinv = fftw_plan_dft_c2r_1d(dim, out, in, FFTW_MEASURE);
    fft.in = in;
    fft.out = out;
    fft.pland = fftr;
    fft.plani = fftinv;

    double* curstate = static_cast<double*>(malloc(dim * sd));
    double* input = static_cast<double*>(malloc(dim * sd));
    double* con = static_cast<double*>(malloc(dim * sd));
    fftw_complex* confft = static_cast<fftw_complex*>(malloc(dimc * sc));
    double* buf = static_cast<double*>(malloc(8 * dim * sd));

    fft.fftcon = confft;

    /* input sweep */
    input_diagram(curstate, con, input, buf, &fft, &par, argv);

    fftw_destroy_plan(fftr);
    fftw_destroy_plan(fftinv);
    free(curstate);
    free(con);
    free(confft);
    free(buf);
    fftw_free(in);
    fftw_free(out);

    return 0;
}
