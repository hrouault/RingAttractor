/*
 * bump_sampling is a software able to fit several bumps of activity on a
 * neural ring like structure. It uses a Monte-Carlo sampling method.
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

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf.h>

#include "quickselect.h"

static const double ampliMin = 2.0;
static const double sigmaDiff = 0.5;
#define nbBumpMax 4
static const size_t nbmoves = 20000;
#define nbroi 16
static const double kappaMean = 2.5;
static const double sig2 = 1.0;
#define sigcoup (2 * M_PI / nbroi)
static const double sigcoup2 = sigcoup * sigcoup;
static const double penbump = 4.0;
/* static const double Jc = 0.020; */
static const double Jc = 1.800;
static const double beta = 5.0;

#define fnamelen 120

typedef struct {
    size_t nbump;
    double pos[nbBumpMax];
    double ampli[nbBumpMax];
    double kappa[nbBumpMax];
    double logl;
} sitebump_t;

typedef struct {
    size_t nbt;
    size_t n;
    double* data;
    char fnameout[fnamelen + 1];
    double median;
} trial_t;

gsl_rng * g_r;

int diffuse(sitebump_t* bump)
{
    if (!bump->nbump) return 1;
    size_t ind = gsl_rng_uniform(g_r) * bump->nbump;
    bump->pos[ind] += gsl_ran_gaussian(g_r, sigmaDiff);
    bump->pos[ind] = fmod(bump->pos[ind], 2 * M_PI);
    if (bump->pos[ind] < 0) bump->pos[ind] += 2 * M_PI;
    return 0;
}

int changeAmpli(sitebump_t* bump)
{
    if (!bump->nbump) return 1;
    size_t ind = gsl_rng_uniform(g_r) * bump->nbump;
    double changeAmpli = gsl_ran_gaussian(g_r, 1.0);
    if (bump->ampli[ind] + changeAmpli < 0)
        return 1;
    bump->ampli[ind] += changeAmpli;
    return 0;
}

int changeWidth(sitebump_t* bump)
{
    if (!bump->nbump) return 1;
    size_t ind = gsl_rng_uniform(g_r) * bump->nbump;
    double changeWidth = bump->kappa[ind] + gsl_ran_gaussian(g_r, 0.5);
    if (changeWidth < 2.0 || changeWidth > 6.0) return 1;
    bump->kappa[ind] = changeWidth;
    return 0;
}

int createBump(sitebump_t* bump)
{
    if (bump->nbump == nbBumpMax) return 1;
    bump->pos[bump->nbump] = gsl_rng_uniform(g_r) * 2 * M_PI;
    bump->ampli[bump->nbump] = ampliMin;
    bump->kappa[bump->nbump] = kappaMean;
    bump->nbump++;
    return 0;
}

int delBump(sitebump_t* bump)
{
    if (!bump->nbump) return 1;
    size_t ind = gsl_rng_uniform(g_r) * bump->nbump;
    bump->pos[ind] = bump->pos[bump->nbump - 1];
    bump->ampli[ind] = bump->ampli[bump->nbump - 1];
    bump->kappa[ind] = bump->kappa[bump->nbump - 1];
    bump->nbump--;
    return 0;
}

double vonmises(double x, double mu, double kappa)
{
    return exp(kappa * cos(x - mu)) / (2 * M_PI * gsl_sf_bessel_I0(kappa));
}

double interf_logl(sitebump_t* bump1, sitebump_t* bump2)
{
    int check1[nbBumpMax];
    int check2[nbBumpMax];
    for (size_t i = 0; i < nbBumpMax; ++i) {
      check1[i] = 0;
      check2[i] = 0;
    }
    int ind1;
    int ind2;
    int check = 0;
    double logli = 0;
    size_t nblinkmax = nbBumpMax;
    if (nblinkmax > bump1->nbump) nblinkmax = bump1->nbump;
    if (nblinkmax > bump2->nbump) nblinkmax = bump2->nbump;
    for (size_t k = 0; k < nblinkmax; ++k) {
        double diffmin = 2 * M_PI;
        for (size_t i = 0; i < bump1->nbump; ++i) {
          if (!check1[i]) {
            for (size_t j = 0; j < bump2->nbump; ++j) {
              if (!check2[j]) {
                double dist = fabs(bump1->pos[i] - bump2->pos[j]);
                double diff = fmin(dist, 2 * M_PI - dist);
                if (diff < diffmin) {
                  diffmin = diff;
                  ind1 = i;
                  ind2 = j;
                  check = 1;
                }
              }
            }
          }
        }
        if (check) {
          check1[ind1] = 1;
          check2[ind2] = 1;
          logli += exp(-0.5 * diffmin * diffmin / sigcoup2);
        }
    }
    return beta * Jc * logli;
}

double site_logl(double* intens, sitebump_t* bump)
{
    double logl = 0;
    logl -= bump->nbump * penbump;
    for (size_t i = 0; i < nbroi; ++i) {
        double val = 0;
        double x = i * 2 * M_PI / nbroi;
        for (size_t j = 0; j < bump->nbump; ++j) {
            val += bump->ampli[j] * vonmises(x, bump->pos[j], bump->kappa[j]);
        }
        double diff = intens[i] - val;
        logl -= diff * diff / sig2 * 0.5;
    }
    return beta * logl;
}

double log_likelihood_whole(sitebump_t* bump, trial_t* trial, double* interfe)
{
    double logl = 0;
    for (size_t i = 0; i < trial->nbt; ++i) {
        double slogl = site_logl(trial->data + i * nbroi, bump + i);
        bump[i].logl = slogl;
        logl += slogl;
    }
    for (size_t i = 0; i < trial->nbt - 1; ++i) {
        double ilogl = interf_logl(bump + i, bump + i + 1);
        interfe[i] = ilogl;
        logl += ilogl;
    }
    return logl;
}

void mcmc(trial_t* trial)
{
    size_t ntime = trial->nbt;
    sitebump_t* bumps = malloc(ntime * sizeof(sitebump_t));
    for (size_t i = 0; i < ntime; ++i) {
        bumps[i].nbump = 0;
    }

    double * interfe = malloc((ntime - 1) * sizeof(double));

    log_likelihood_whole(bumps, trial, interfe);

    for (size_t i = 0; i < nbmoves; ++i) {
        for (size_t j = 0; j < ntime; ++j) {
            sitebump_t bumpt = bumps[j];
            double randn = gsl_rng_uniform(g_r);
            int check = 1;
            if (randn < 0.01) {
                /* Appearance / disappearance of a bump */
                check = createBump(&bumpt);
            } else if (randn < 0.01 * (1 + bumpt.nbump)) {
                check = delBump(&bumpt);
            } else if (randn < 0.3) {
                /* Diffusion of one of the bumps */
                check = diffuse(&bumpt);
            } else if (randn < 0.4) {
                /* Change of the bump width */
                check = changeWidth(&bumpt);
            } else {
                /* Change of the amplitude */
                check = changeAmpli(&bumpt);
            }
            if (!check) {
                double loglt = site_logl(trial->data + j * nbroi, &bumpt);
                double loglit1 = 0;
                double loglit1p = 0;
                double loglit2 = 0;
                double loglit2p = 0;
                if (j != 0 && j != ntime - 1) {
                    loglit1 = interf_logl(bumps + j - 1, &bumpt);
                    loglit1p = interfe[j - 1];
                    loglit2 = interf_logl(bumps + j + 1, &bumpt);
                    loglit2p = interfe[j];
                } else if (j == 0) {
                    loglit1 = interf_logl(bumps + 1, &bumpt);
                    loglit1p = interfe[0];
                } else if (j == ntime - 1) {
                    loglit1 = interf_logl(bumps + ntime - 2, &bumpt);
                    loglit1p = interfe[ntime - 2];
                }

                double difflogl = loglt - bumps[j].logl + loglit1 - loglit1p;
                if (j != 0 && j != ntime - 1) {
                    difflogl += loglit2 - loglit2p;
                }
                if (difflogl > 0 || gsl_rng_uniform(g_r) < exp(difflogl)) {
                    bumpt.logl = loglt;
                    bumps[j] = bumpt;
                    if (j != 0 && j != ntime - 1) {
                        interfe[j - 1] = loglit1;
                        interfe[j] = loglit2;
                    } else if (j == 0) {
                        interfe[0] = loglit1;
                    } else if (j == ntime - 1) {
                        interfe[ntime - 2] = loglit1;
                    }
                }
            }
        }
    }
    char fnamefits[fnamelen + 1];
    snprintf(fnamefits, fnamelen, "%s-fits.dat", trial->fnameout);
    FILE* ffits = fopen(fnamefits, "w");
    if (ffits == NULL) {
        fprintf(stderr, "Cannot open file %s: %s\n",
                fnamefits, strerror(errno));
        exit(EXIT_FAILURE);
    }
    char fnamenbump[fnamelen + 1];
    snprintf(fnamenbump, fnamelen, "%s-nbump.dat", trial->fnameout);
    FILE* fnbump = fopen(fnamenbump, "w");
    if (fnbump == NULL) {
        fprintf(stderr, "Cannot open file %s: %s\n",
                fnamenbump, strerror(errno));
        exit(EXIT_FAILURE);
    }
    char fnamecentrbump[fnamelen + 1];
    snprintf(fnamecentrbump, fnamelen, "%s-centrbump.dat", trial->fnameout);
    FILE* fcentrbump = fopen(fnamecentrbump, "w");
    if (fcentrbump == NULL) {
        fprintf(stderr, "Cannot open file %s: %s\n",
                fnamecentrbump, strerror(errno));
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < ntime; ++i) {
        for (size_t j = 0; j < bumps[i].nbump; ++j) {
            fprintf(ffits, "%ld %f %f %f\n", i, bumps[i].pos[j],
                    bumps[i].ampli[j], bumps[i].kappa[j]);
            for (size_t k = 0; k < nbroi; ++k) {
                double dist = bumps[i].pos[j] - (k) * 2 * M_PI / nbroi;
                if (dist < -M_PI) dist += 2 * M_PI;
                if (dist > M_PI) dist -= 2 * M_PI;
                double ampli = trial->data[i * nbroi + k] / bumps[i].ampli[j];

                fprintf(fcentrbump, "%f %f\n", dist, ampli);
            }
        }
        fprintf(fnbump, "%ld %ld", i, bumps[i].nbump);
        for (size_t j = 0; j < nbroi; ++j) {
          double val = 0;
          double x = j * 2 * M_PI / nbroi;
          for (size_t k = 0; k < bumps[i].nbump; ++k) {
            val += bumps[i].ampli[k] * vonmises(x, bumps[i].pos[k],
                                                bumps[i].kappa[k]);
          }
          fprintf(fnbump, " %f", val);
        }
        fprintf(fnbump, "\n");
    }
    fclose(ffits);
    fclose(fnbump);
}

void import_dataset(char* filename, trial_t* trial)
{
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    FILE* dataf = fopen(filename, "r");
    if (dataf == NULL) {
        fprintf(stderr, "Cannot open file %s: %s\n",
                filename, strerror(errno));
        exit(EXIT_FAILURE);
    }

    trial->nbt = 0;
    size_t nblinesmax = 1024;
    trial->data = malloc(nblinesmax * nbroi * sizeof(double));
    size_t nbv = 0;
    while ((read = getline(&line, &len, dataf)) != -1) {
        char* token;
        token = strtok(line, " ");
        while (token != NULL) {
            trial->data[nbv] = atof(token);
            token = strtok(NULL, " ");
            nbv++;
        }
        trial->nbt++;
        if (trial->nbt >= nblinesmax) {
            nblinesmax *= 2;
            trial->data = realloc(trial->data,
                                  nblinesmax * nbroi * sizeof(double));
        }

    }
    /* for (size_t i = 0; i < trial->nbt; ++i) { */
    /*     for (size_t j = 0; j < nbroi - 1; ++j) { */
    /*         printf("%f ", trial->data[i * nbroi + j]); */
    /*     } */
    /*     printf("%f ", trial->data[(i + 1) * nbroi - 1]); */
    /* } */
    /* printf("Nb of lines read: %ld\n", trial->nbt); */
    strrchr(filename, '.')[0] = '\0';
    snprintf(trial->fnameout, fnamelen, "%s", filename);

    free(line);
    fclose(dataf);

    trial->n = trial->nbt * nbroi;

    double* datacp = malloc(trial->n * sizeof(double));
    memcpy(datacp, trial->data, trial->n * sizeof(double));
    trial->median = quick_select(datacp, trial->n) * 0.7;
    free(datacp);

    printf("median of the trial: %f\n", trial->median);
    for (size_t i = 0; i < trial->n; ++i) {
      trial->data[i] /= trial->median;
      trial->data[i] -= 1.0;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: ./bumpsampling filename\n");
        exit(EXIT_FAILURE);
    }

    gsl_rng_env_setup();

    g_r = gsl_rng_alloc(gsl_rng_default);

    trial_t trial;
    import_dataset(argv[1], &trial);

    mcmc(&trial);

    free(trial.data);

    gsl_rng_free(g_r);

    return 0;
}
