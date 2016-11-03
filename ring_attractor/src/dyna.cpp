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
#include <vector>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fftw3.h>

#include "dyna.hpp"

const size_t nbpoints = 1;

double vonmises(double x, double mu, double kappa);
void deriv_fft(double* array, double* darray, double* input,
               parasw_t* par, fft_t* fft);
void dynamics_derivative(double* array, double* darray, float tottime);
void dynamics_noise_fft(double* curstate, double* input, double* buffer,
                        fft_t* fft, parasw_t* par);

void deriv_fft(double* array, double* darray, double* input,
               parasw_t* par, fft_t* fft)
{
    /* Convolution */
    for (size_t i = 0; i < par->nbpts; ++i) {
        fft->in[i] = array[i];
    }
    fftw_execute(fft->pland);
    for (size_t i = 0; i < par->nbpts / 2 + 1; ++i) {
        double out1 = fft->out[i][0];
        double out2 = fft->out[i][1];
        double con1 = fft->fftcon[i][0];
        double con2 = fft->fftcon[i][1];
        fft->out[i][0] = out1 * con1 - out2 * con2;
        fft->out[i][1] = out1 * con2 + out2 * con1;
    }
    fftw_execute(fft->plani);

    for (size_t i = 0; i < par->nbpts; ++i) {
        darray[i] = fft->in[i] / par->nbpts + 1 + input[i];
        /* leakage term and thresholding */
        if (darray[i] < 0) darray[i] = 0;
        darray[i] -= array[i];
    }
}

double vonmises(double x, double mu, double kappa)
{
    if (kappa > 5) return exp(kappa * (cos(x - mu) - 1));
    return (exp(kappa * (cos(x - mu) + 1)) - 1) / (exp(2 * kappa) - 1);
}

void init_connect(double* connect, parasw_t* par)
{
    double k1 = 1.0 / par->w1 / par->w1;
    double k2 = 1.0 / par->w2 / par->w2;
    for (size_t i = 0; i < par->nbpts; ++i) {
        double x = 2 * M_PI * i / par->nbpts;
        connect[i] = par->offcon + par->a1 * vonmises(x, 0.0, k1)
                     - par->a2 * vonmises(x, 0.0, k2);
    }
    for (size_t i = 1; i < par->nbpts - 1; ++i) {
        if (connect[i - 1] + connect[i + 1] > 2 * connect[i]) {
            par->conwidth = i;
            break;
        }
    }
}

void init_connect_delta(double* connect, parasw_t* par)
{
    for (size_t i = 0; i < par->nbpts; ++i) {
        connect[i] = -par->beta * 2 * M_PI / par->nbpts;
    }
    connect[0] += par->alpha;
    double dx = 2 * M_PI / par->nbpts;
    connect[par->nbpts - 1] += par->D / dx / dx;
    connect[1] += par->D / dx / dx;
    connect[0] -= 2 * par->D / dx / dx;
}

void init_connect_cosine(double* connect, parasw_t* par)
{
    for (size_t i = 0; i < par->nbpts; ++i) {
        connect[i] = par->J0 + par->J1 * cos(i * 2 * M_PI / par->nbpts);
    }
}

void init_connect_wta(double* connect, parasw_t* par)
{
    for (size_t i = 1; i < par->nbpts; ++i) {
        connect[i] = -par->beta;
    }
    connect[0] = par->alpha;
}

void init_input(double* input, parasw_t* par)
{
    double kin = 1.0 / par->win / par->win;
    for (size_t i = 0; i < par->nbpts; ++i) {
        double x = 2 * M_PI * i / par->nbpts;
        input[i] = par->offin + par->ain * vonmises(x, 2 * M_PI / 3, kin);
    }
}

void init_sw(double* curstate, double* input, double* buf,
             fft_t* fft, parasw_t* par)
{
    for (size_t i = 0; i < 20.0 / par->dt; ++i) {
        dynamics_rk4step_fft(curstate, input, buf, fft, par);
        /* if (i > 0) { */
        /*     FILE* fstate = fopen("state.dat", "w"); */
        /*     if (fstate == NULL) { */
        /*         fprintf(stderr, "Cannot open file %s: %s\n", */
        /*                 "state.dat", strerror(errno)); */
        /*         exit(EXIT_FAILURE); */
        /*     } */
        /*     for (size_t j = 0; j < par->nbpts; ++j) { */
        /*         fprintf(fstate, "%f\n", curstate[j]); */
        /*     } */
        /*     fclose(fstate); */
        /*     exit(EXIT_SUCCESS); */
        /* } */
        /* printf("%f\n", curstate[0]); */
    }
}

uint16_t check_1max(double* curstate, parasw_t* par)
{
    uint16_t nbmax = 0;
    uint16_t last = par->nbpts - 1;
    if (curstate[last] < curstate[0] && curstate[1] < curstate[0])
        nbmax++;
    if (curstate[last - 1] < curstate[last] && curstate[0] < curstate[last])
        nbmax++;
    for (size_t i = 1; i < par->nbpts - 1; ++i) {
        if (curstate[i - 1] < curstate[i] && curstate[i+1] < curstate[i])
            nbmax++;
    }
    return nbmax;
}

static max_t max_state(double* curstate, uint16_t size)
{
    max_t max;
    max.max = 0;
    max.pos = 0;
    for (uint16_t i = 0; i < size; ++i) {
        if (curstate[i] > max.max) {
            max.max = curstate[i];
            max.pos = i;
        }
    }
    return max;
}

int16_t bump_width(double* curstate, parasw_t* par)
{
    for (int16_t i = 1; i < par->nbpts - 1; ++i) {
        if (curstate[i] < 1e-6) return i;
    }
    return -1;
}

static double analyse_jump_or_flow(double* curstate, double* input,
                                   double* buf, fft_t* fft, parasw_t* par)
{
    /* double* statejump = */
    /*     static_cast<double*>(malloc(par->nbpts * sizeof(double))); */
    for (size_t j = 0; j < 100.0 / par->dt; ++j) {
        dynamics_rk4step_fft(curstate, input, buf, fft, par);
    }
    double maxin = 0.0;
    for (size_t j = 0; j < par->nbpts; ++j) {
        /* statejump[j] = curstate[j]; */
        if (curstate[j] > maxin) {
            maxin = curstate[j];
        }
    }
    /* double* statein = */
    /*     static_cast<double*>(malloc(par->nbpts * sizeof(double))); */
    /* for (size_t j = 0; j < par->nbpts; ++j) { */
    /*     statein[j] = curstate[j]; */
    /* } */
    /* free(statein); */
    /* free(statejump); */
    return maxin;
}

void noisy_simu(double* curstate, double* input, double* buf,
                fft_t* fft, parasw_t* par, gsl_rng* rng)
{
    uint16_t dim = par->nbpts;
    for (size_t i = 0; i < dim; ++i) {
        curstate[i] = 0;
        input[i] = 0;
    }
    curstate[0] = 0.01;
    curstate[1] = 0.01;
    curstate[dim - 1] = 0.01;

    FILE* fprof = fopen("noisy.dat", "w");
    /* if (kin > 80) exit(EXIT_SUCCESS); */
    /* for (size_t i = 0; i < 200000.0 / par->dt; ++i) { */
    for (size_t i = 0; i < 400000.0 / par->dt; ++i) {
        for (size_t i = 0; i < dim; ++i) {
            /* input[i] = gsl_ran_gaussian(rng, 0.1); */
            input[i] = gsl_ran_gaussian(rng, 0.50);
        }
        dynamics_noise_fft(curstate, input, buf, fft, par);
        /* if (i % 20000 == 0) { */
        if (i % 20000 == 0) {
            for (size_t i = 0; i < dim; ++i) {
                fprintf(fprof, "%f ", curstate[i]);
            }
            fprintf(fprof, "\n");
        }
    }
    fclose(fprof);
}
void
jump_vs_flow_simu(double inpos, double* curstate, double* input, double* buf,
                  fft_t* fft, parasw_t* par)
{
    uint16_t dim = par->nbpts;
    for (size_t i = 0; i < dim; ++i) {
        curstate[i] = 0;
        input[i] = 0;
    }
    curstate[0] = 10.0;
    curstate[1] = 10.0;
    curstate[dim - 1] = 10.0;

    init_sw(curstate, input, buf, fft, par);
    double ampliini = curstate[0];
    /* FILE* finit = fopen("init-cos.dat", "w"); */
    /* for (size_t i = 0; i < dim; ++i) { */
    /*     fprintf(finit, "%f\n", curstate[i]); */
    /* } */
    /* fclose(finit); */
    /* exit(EXIT_SUCCESS); */

    double ampliin = 0;
    double ampliaft = 0;
    uint16_t checkjvf1 = 0;
    uint16_t check = 0;
    double kin = 1.0 / par->win / par->win;
    for (size_t a = 0; a < 60; ++a) {
        ampliin = 2e-4 * pow(2.0, a);
        /* FILE* fin = fopen("input.dat", "w"); */
        for (size_t i = 0; i < dim; ++i) {
            double x = 2 * M_PI * i / (double)dim;
            input[i] = par->offin + ampliin * vonmises(x, inpos, kin);
            /* fprintf(fin, "%f\n", input[i]); */
        }
        /* fclose(fin); */
        /* if (kin > 80) exit(EXIT_SUCCESS); */
        for (size_t i = 0; i < 40.0 / par->dt; ++i) {
            dynamics_rk4step_fft(curstate, input, buf, fft, par);
            max_t max = max_state(curstate, dim);
            if (max.pos > dim / 16 && max.pos < 7 * dim / 8 ) {
                /* Checking for jumps */
                if (max.pos > dim / 8) checkjvf1 = 1;
                ampliaft = analyse_jump_or_flow(curstate, input, buf, fft, par);
                check = 1;
                break;
            }
        }
        if (check) break;
    }
    if (check) {
        /* printf("%f %f %f %d\n", inpos, par->win, ampliin, checkjvf1); */
        printf("%f %f %f %f %f %f %f %f %d %d \n",
                ampliini, ampliaft,
                par->w1, par->w2, par->win, par->a1, par->a2, par->offcon,
                checkjvf1, par->conwidth);
    }
}

size_t input_inner_loop(double amp, double width, double* input, size_t dim,
                        double jump_dist, double* curstate,
                        double* curstateini, double* buf, double ampliini,
                        fft_t* fft, parasw_t* par)
{
    double kin = 1.0 / width / width;

    /* compute the real ampli */
    for (size_t k = 0; k < dim; ++k) {
        double x = 2 * M_PI * k / (double)dim;
        input[k] = amp * vonmises(x, 0.0, kin);
        curstate[k] = curstateini[k];
    }
    init_sw(curstate, input, buf, fft, par);

    double ampliaft = curstate[0];

    for (size_t k = 0; k < dim; ++k) {
        curstate[k] = curstateini[k];
        double x = 2 * M_PI * k / (double)dim;
        /* input[k] = amp * vonmises(x, 150.0 * M_PI / 180.0, kin); */
        /* input[k] = amp * vonmises(x, 60.0 * M_PI / 180.0, kin); */
        /* input[k] = amp * vonmises(x, 120.0 * M_PI / 180.0, kin); */

        /* input[k] = amp * vonmises(x, 90.0 * M_PI / 180.0, kin); */
        input[k] = amp * vonmises(x, jump_dist * M_PI / 180.0, kin);
    }

    /* if (width > M_PI - 0.5) { */
    /*     FILE* finput = fopen("input.dat", "w"); */
    /*     if (finput == NULL) { */
    /*         fprintf(stderr, "Cannot open file %s: %s\n", */
    /*                 "input.dat", strerror(errno)); */
    /*         exit(EXIT_FAILURE); */
    /*     } */
    /*     for (size_t k = 0; k < dim; ++k) { */
    /*         curstate[k] = curstateini[k]; */
    /*         double x = 2 * M_PI * k / (double)dim; */
    /*         fprintf(finput, "%f %f\n", x, input[k]); */
    /*     } */
    /*     fclose(finput); */
    /* } */

    uint16_t checkjvf1 = 0;
    uint16_t check = 0;

    for (size_t k = 0; k < 20.0 / par->dt; ++k) {
        dynamics_rk4step_fft(curstate, input, buf, fft, par);
        max_t max = max_state(curstate, dim);
        if (max.pos > dim / 16 && max.pos < 7 * dim / 8 ) {
            /* Checking for jumps */
            if (max.pos > dim / 8) checkjvf1 = 1;
            /* analyse_jump_or_flow(curstate, input, buf, fft, par); */
            check = 1;
            break;
        }
    }
    printf("%f %f %f %d %f\n", ampliini, ampliaft, width,
           check + checkjvf1, amp);

    return check + checkjvf1;
}

void
jump_vs_flow_input(double* curstate, double* input, double* buf, fft_t* fft,
                   parasw_t* par, char** argv)
{
    uint16_t dim = par->nbpts;
    uint16_t dimc = par->nbpts / 2 + 1;

    /* initial state properties */
    for (size_t i = 0; i < dim; ++i) {
        curstate[i] = 0;
        input[i] = 0;
    }
    curstate[0] = 10.0;
    curstate[1] = 10.0;
    curstate[dim - 1] = 10.0;
    init_sw(curstate, input, buf, fft, par);
    double ampliini = curstate[0];
    if (curstate[0] > 1e6) return;
    double max = 0;
    double min = 1e6;
    for (size_t i = 0; i < dim; ++i) {
        if (curstate[i] > max) max = curstate[i];
        if (curstate[i] < min) min = curstate[i];
    }
    if (fabs(min - max) < 1e-4) return;
    FILE* finit = fopen("init.dat", "w");
    if (!finit) {
        printf("cannot open file: init.dat\n");
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < dim; ++i) {
        fprintf(finit, "%f\n", curstate[i]);
    }
    fclose(finit);

    double bw = 2 * M_PI * bump_width(curstate, par) / par->nbpts;
    /* printf("Bump width: %f\n", bw * 180 / M_PI); */
    if (bw > M_PI / 4) return;
    if (bw < 3 * M_PI / 16) return;
    /* printf("width is good!\n"); */

    double* curstateini = static_cast<double*>(malloc(dim * sizeof(double)));
    for (size_t i = 0; i < dim; ++i) {
        curstateini[i] = curstate[i];
    }

    size_t nb_width = 160;
    size_t nb_amp = 40;
    size_t nb_amp_fine = 10;
    double sta_width = 0.01;
    double sto_width = 3 * M_PI / 4;
    double sta_amp = 0.01;
    double sto_amp = 5.0 * 3;
    /* double sto_amp = 0.2; */
    if (std::strcmp("delta", argv[1]) == 0) {
        sto_amp = 0.4 * 3;
    }

    double jump_dist = atof(argv[2]);

    for (size_t i = 0; i < nb_width; ++i) {
        size_t out = 0;
        double width = i * (sto_width - sta_width) / nb_width + sta_width;
        for (size_t j = 0; j < nb_amp; ++j) {
            double amp = j * (sto_amp - sta_amp) / nb_amp + sta_amp;
            size_t outcur = input_inner_loop(amp, width, input, dim, jump_dist,
                                             curstate, curstateini, buf,
                                             ampliini, fft, par); 
            if (outcur != out) {
                for (size_t k = 0; k < nb_amp_fine; ++k) {
                    amp = (j - 1 + double(k) / nb_amp_fine)
                           * (sto_amp - sta_amp) / nb_amp + sta_amp;
                    input_inner_loop(amp, width, input, dim, jump_dist,
                                     curstate, curstateini, buf, ampliini, fft,
                                     par); 
                }
            }
            out = outcur;
        }
    }

    free(curstateini);
}

void input_sw(double* curstate, double* input, double* buf,
              fft_t* fft, parasw_t* par)
{
    uint16_t dim = par->nbpts;
    double* inistate = static_cast<double*>(malloc(dim * sizeof(double)));
    for (size_t i = 0; i < dim; ++i) {
        inistate[i] = curstate[i];
    }
    double ampliini = curstate[0];
    printf("width input: %f\n", par->win);
    double kin = 1.0 / par->win / par->win;
    uint16_t check = 0;
    double ampli1 = 0;
    double ampli2 = 0;
    double ampliin = 0;
    uint16_t checkjvf1 = 0;
    for (size_t a = 0; a < 60; ++a) {
        ampliin = 2e-3 * pow(2.0, a);
        FILE* fin = fopen("input.dat", "w");
        for (size_t i = 0; i < dim; ++i) {
            double x = 2 * M_PI * i / (double)dim;
            input[i] = par->offin + ampliin * vonmises(x, 120.0 * M_PI / 180.0,
                                                       kin);
            fprintf(fin, "%f\n", input[i]);
        }
        fclose(fin);
        /* if (kin > 80) exit(EXIT_SUCCESS); */
        for (size_t i = 0; i < 40.0 / par->dt; ++i) {
            dynamics_rk4step_fft(curstate, input, buf, fft, par);
            max_t max = max_state(curstate, dim);
            if (max.pos > dim / 8 && max.pos < 7 * dim / 8 ) {
                /* Checking for jumps */
                if (max.pos > dim / 4) checkjvf1 = 1;
                ampli1 = analyse_jump_or_flow(curstate, input, buf, fft, par);
                check = 1;
                break;
            }
        }
        if (check) break;
    }
    printf("ampliin: %f\n", ampliin);
    if (check) {
        if (checkjvf1) printf("jump\n");
        else printf("flow\n");
    }
    checkjvf1 = 1;

    /* initializing again to the first state */
    for (size_t i = 0; i < dim; ++i) {
        curstate[i] = 0;
        input[i] = 0;
    }
    curstate[0] = 10.0;
    curstate[1] = 10.0;
    curstate[dim - 1] = 10.0;
    init_sw(curstate, input, buf, fft, par);

    FILE* fcur = fopen("fcurini.dat", "w");
    for (size_t j = 0; j < par->nbpts; ++j) {
        fprintf(fcur, "%f %f\n", curstate[j],
                                 fft->in[j]);
    }
    fclose(fcur);

    for (size_t i = 0; i < dim; ++i) {
        input[i] = 0;
    }
    input[par->nbpts / 2] = 2.4;
    for (size_t i = 0; i < 40.0 / par->dt; ++i) {
        dynamics_rk4step_fft(curstate, input, buf, fft, par);
    }
    fcur = fopen("fcur.dat", "w");
    for (size_t j = 0; j < par->nbpts; ++j) {
        fprintf(fcur, "%f %f\n", curstate[j],
                                 fft->in[j]);
    }
    fclose(fcur);
    /* exit(EXIT_SUCCESS); */

    check = 0;
    uint16_t checkjvf2 = 0;
    for (size_t a = 0; a < 60; ++a) {
        double ampli = 2e-3 * pow(2.0, a);
        for (size_t i = 0; i < dim; ++i) {
            double x = 2 * M_PI * i / (double)dim;
            input[i] = par->offin + ampli * vonmises(x, M_PI, kin);
        }
        for (size_t i = 0; i < 40.0 / par->dt; ++i) {
            dynamics_rk4step_fft(curstate, input, buf, fft, par);
            max_t max = max_state(curstate, dim);
            if (max.pos > dim / 8 && max.pos < 7 * dim / 8 ) {
                if (max.pos > dim / 4) checkjvf2 = 1;
                ampli2 = analyse_jump_or_flow(curstate, input, buf, fft, par);
                check = 1;
                break;
            }
        }
        if (check) break;
    }
    if (curstate[0] < 1e-6) {
        printf("%f %f %f %f %f %f %f %f %f %d %d %d %f %f\n",
                ampliini, ampli1, ampli2,
                par->w1, par->w2, par->win, par->a1, par->a2, par->offcon,
                checkjvf1, checkjvf2, par->conwidth,
                par->alpha, par->beta);
        /* if (par->w1 > 1 && ampli2 / ampliini < 2.0 && checkjvf1) { */
        /*     FILE* fcur = fopen("fcur.dat", "w"); */
        /*     for (size_t j = 0; j < par->nbpts; ++j) { */
        /*         fprintf(fcur, "%f %f %f %f\n", inistate[j], */
        /*                 input[j], */
        /*                 fft->connect[j], */
        /*                 fft->in[j]); */
        /*     } */
        /*     fclose(fcur); */
        /*     exit(EXIT_SUCCESS); */
        /* } */

        /* FILE* pcur_states = fopen("states.dat", "w"); */
        /* if (pcur_states == NULL) { */
        /*     fprintf(stderr, "Cannot open file %s: %s\n", */
        /*             "states.dat", strerror(errno)); */
        /*     exit(EXIT_FAILURE); */
        /* } */
        /* for (size_t j = 0; j < par->nbpts; ++j) { */
        /*     fprintf(pcur_states, "%f %f %f %f %f\n", inistate[j], */
        /*                                              statejump[j], */
        /*                                              statein[j], */
        /*                                              input[j], */
        /*                                              fft->connect[j]); */
        /* } */
        /* fclose(pcur_states); */
        /* if (maxin / ampliini < 2.0 && par->w1 > 1.5 && checkjvf) { */
        /*     printf("%f %f\n", maxin, ampliini); */
        /*     exit(EXIT_SUCCESS); */
        /* } */
    }
    free(inistate);
}

void dynamics_rk4step_fft(double* curstate, double* input, double* buffer,
                          fft_t* fft, parasw_t* par)
{
    double dt = par->dt;

    uint16_t n = par->nbpts;

    for (size_t i = 0; i < n; i++ ) {
        buffer[i] = curstate[i];
    }
    deriv_fft(&buffer[0], &buffer[n], input, par, fft);

    for (size_t i = 0; i < n; i++ ) {
        buffer[2 * n + i] = buffer[i] + dt * buffer[n + i] / 2.0;
    }
    deriv_fft(&buffer[2 * n], &buffer[3 * n], input, par, fft);

    for (size_t i = 0; i < n; i++ ) {
        buffer[4 * n + i] = buffer[i] + dt * buffer[3 * n + i] / 2.0;
    }
    deriv_fft(&buffer[4 * n], &buffer[5 * n], input, par, fft);

    for (size_t i = 0; i < n; i++ ) {
        buffer[6 * n + i] = buffer[i] + dt * buffer[5 * n + i];
    }
    deriv_fft(&buffer[6 * n], &buffer[7 * n], input, par, fft);

    for (size_t i = 0; i < n; i++ ) {
        curstate[i] = buffer[i] + dt * (buffer[n + i] + 2.0 * buffer[3 * n + i]
                      + 2.0 * buffer[5 * n + i] + buffer[7 * n + i] ) / 6.0;
    }
}

void dynamics_noise_fft(double* curstate, double* input, double* buffer,
                        fft_t* fft, parasw_t* par)
{
    double dt = par->dt;

    uint16_t n = par->nbpts;

    for (size_t i = 0; i < n; i++ ) {
        buffer[i] = curstate[i];
    }
    deriv_fft(&buffer[0], &buffer[n], input, par, fft);

    for (size_t i = 0; i < n; i++ ) {
        curstate[i] = buffer[i] + dt * buffer[n + i];
    }
}
