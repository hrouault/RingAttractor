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

#ifndef DYNA_HPP_1KJSUPBA
#define DYNA_HPP_1KJSUPBA

#include "gsl/gsl_rng.h"
#include <fftw3.h>

extern const size_t nbpoints;

typedef struct {
    double max;
    uint16_t pos;
} max_t;

typedef struct {
    size_t nbpts;
    double dx;
    double D;
    double alpha;
    double alpha_p;
    double beta;
    double asym;
} params_t;

typedef struct {
    uint16_t nbpts;
    double dx;
    double dt;
    double trelax;
    double offcon;
    double w1;
    double a1;
    double w2;
    double a2;
    double offin;
    double ain;
    double win;
    double D;
    double alpha;
    double beta;
    double J0;
    double J1;
    uint16_t conwidth;
} parasw_t;

typedef struct {
    uint16_t n;
    double* in;
    double* connect;
    fftw_complex* out;
    fftw_complex* fftcon;
    fftw_plan pland;
    fftw_plan plani;
} fft_t;

void dynamics_rk4step_fft(double* curstate, double* input, double* buffer,
                          fft_t* fft, parasw_t* par);
void init_connect(double* connect, parasw_t* par);
void init_connect_delta(double* connect, parasw_t* par);
void init_connect_cosine(double* connect, parasw_t* par);
void init_connect_wta(double* connect, parasw_t* par);
void init_sw(double* curstate, double* input, double* buf,
             fft_t* fft, parasw_t* par);
uint16_t check_1max(double* curstate, parasw_t* par);
void init_input(double* input, parasw_t* par);
void input_sw(double* curstate, double* input, double* buf,
              fft_t* fft, parasw_t* par);
void jump_vs_flow_simu(double inpos, double* curstate, double* input,
                       double* buf, fft_t* fft, parasw_t* par);
void noisy_simu(double* curstate, double* input, double* buf,
                fft_t* fft, parasw_t* par, gsl_rng* rng);
void end_trial(double* curstate, parasw_t* par);
void jump_vs_flow(double* maxs);
int16_t bump_width(double* curstate, parasw_t* par);
void jump_vs_flow_input(double* curstate, double* input, double* buf, fft_t*
                        fft, parasw_t* par, char** argv);

#endif /* end of include guard: DYNA_HPP_1KJSUPBA */
