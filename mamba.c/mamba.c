/* Inference for Mamba model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

// Function prototype for calculate_mse
float calculate_mse(const char *filename, int *tokens, int num_tokens, float *embedding_table, int embedding_dim);

// ----------------------------------------------------------------------------
// Fixed-point arithmetic

typedef int32_t fixed_t;

#define FIXED_FRACTIONAL_BITS 23
#define FIXED_ONE (1 << FIXED_FRACTIONAL_BITS)
#define FLOAT_TO_FIXED(x) ((fixed_t)((x) * FIXED_ONE))
#define FIXED_TO_FLOAT(x) (((float)(x)) / FIXED_ONE)
#define FIXED_MUL(x, y) (((x) * (y)) >> FIXED_FRACTIONAL_BITS)
#define FIXED_DIV(x, y) ((y == 0) ? 0 : ((fixed_t)(((int64_t)(x) << FIXED_FRACTIONAL_BITS) / (y))))
#define FIXED_ADD_INT(x, n) (FLOAT_TO_FIXED(FIXED_TO_FLOAT((x)) + (n)))

#define FIXED_EXP_LIMIT 10
#define FIXED_LOG_PRECISION FLOAT_TO_FIXED(1)

// Exponential function using Taylor series approximation
// fixed_t exp_fixed(fixed_t x)
// {
//     if (x < 0)
//     {
//         // e^-x = 1 / e^x
//         return FIXED_DIV(FIXED_ONE, exp_fixed(-x));
//     }
//     return FLOAT_TO_FIXED(expf(FIXED_TO_FLOAT(x)));
//     // if (x < 0)
//     // {
//     //     // e^-x = 1 / e^x
//     //     return FIXED_DIV(FIXED_ONE, exp_fixed(-x));
//     // }
//     // if (x > FLOAT_TO_FIXED(2))
//     // {
//     //     return FIXED_MUL(exp_fixed(FIXED_DIV(x, FLOAT_TO_FIXED(2))), exp_fixed(FIXED_DIV(x, FLOAT_TO_FIXED(2))));
//     // }

//     // fixed_t result = FIXED_ONE; // e^0 = 1
//     // fixed_t term = FIXED_ONE;   // Initial term x^0 / 0! = 1
//     // fixed_t i = FIXED_ONE;      // Start with x^1 / 1!

//     // for (int j = 1; j < FIXED_EXP_LIMIT; j++)
//     // {
//     //     term = FIXED_DIV(FIXED_MUL(term, x), i);
//     //     result += term;
//     //     i += FIXED_ONE;
//     // }

//     // if (FIXED_TO_FLOAT(result) < 0)
//     // {
//     //     fprintf(stderr, "exp_fixed: x = %f, result = %f\n", FIXED_TO_FLOAT(x), FIXED_TO_FLOAT(result));
//     // }
//     // return result;
// }

// // Fixed-point logarithm function
// fixed_t log_fixed(fixed_t x)
// {
//     return FLOAT_TO_FIXED(logf(FIXED_TO_FLOAT(x)));
//     // if (x <= 0)
//     // {
//     //     printf("Error: log(x) is undefined for x <= 0\n");
//     //     return 0; // Return negative infinity to indicate error
//     // }

//     // // Initial guess for log(x)
//     // fixed_t guess = x > FIXED_ONE ? x - FIXED_ONE : x;
//     // fixed_t difference;

//     // do
//     // {
//     //     fixed_t e_guess = exp_fixed(guess);
//     //     difference = FIXED_DIV((x - e_guess), e_guess);
//     //     guess += difference;
//     // } while (difference > FIXED_LOG_PRECISION || difference < -FIXED_LOG_PRECISION);

//     // return guess;
// }

void fixed_to_float_arr(fixed_t *arr, float *result, int n)
{
    for (int i = 0; i < n; i++)
    {
        result[i] = FIXED_TO_FLOAT(arr[i]);
    }
}

// ----------------------------------------------------------------------------
// Mamba model

typedef struct
{
    int n_layers;   // number of layers
    int vocab_size; // vocabulary size
    int dim;        // embedding dimension
    int d_inner;
    int dt_rank;
    int d_state;
    int d_conv;
    int shared_classifier;
    int rounded_vocab_size; // vocab_size rounded up to the nearest multiple of 8
} Config;

typedef struct
{
    // token embedding table
    float *token_embedding_table; // (rounded_vocab_size, dim)
    // weights for layers
    float *in_proj;        // (layer, 2*d_inner, dim)
    float *conv1d_weight;  // (layer, d_inner, 1, d_conv)
    float *conv1d_bias;    // (layer, d_inner)
    float *x_proj;         // (layer, dt_rank+2*d_state, d_inner)
    float *dt_proj_weight; // (layer, d_inner, dt_rank)
    float *dt_proj_bias;   // (layer, d_inner)
    float *A;              // (layer, d_inner, d_state)
    float *D;              // (layer, d_inner)
    float *out_proj;       // (layer, dim, d_inner)
    float *norm;           // (layer, dim)
    // final rmsnorm
    float *final_norm; // (dim)
    // (optional) classifier weights for the logits, on the last layer
    float *lm_head; // (rounded_vocab_size, dim)
} MambaWeightsFloat;

typedef struct
{
    // token embedding table
    fixed_t *token_embedding_table; // (rounded_vocab_size, dim)
    // weights for layers
    fixed_t *in_proj;        // (layer, 2*d_inner, dim)
    fixed_t *conv1d_weight;  // (layer, d_inner, 1, d_conv)
    fixed_t *conv1d_bias;    // (layer, d_inner)
    fixed_t *x_proj;         // (layer, dt_rank+2*d_state, d_inner)
    fixed_t *dt_proj_weight; // (layer, d_inner, dt_rank)
    fixed_t *dt_proj_bias;   // (layer, d_inner)
    fixed_t *A;              // (layer, d_inner, d_state)
    fixed_t *D;              // (layer, d_inner)
    fixed_t *out_proj;       // (layer, dim, d_inner)
    fixed_t *norm;           // (layer, dim)
    // final rmsnorm
    fixed_t *final_norm; // (dim)
    // (optional) classifier weights for the logits, on the last layer
    fixed_t *lm_head; // (rounded_vocab_size, dim)
} MambaWeights;

typedef struct
{
    // memory reused by all layers
    float *input;        // (dim)
    float *hidden_state; // (dim)
    float *xz;           // (2*d_inner)          x and z are pointers into this buffer
    float *x_db;         // (dt_rank+2*d_state)  dt, B, C are pointers into this buffer
    float *dt;           // (d_inner)            later, dt is a pointer to this buffer
    float *dA;           // (d_inner, d_state)
    float *dB;           // (d_inner, d_state)
    float *temp;         // (d_inner, d_state)
    float *y;            // (d_inner)
    float *logits;       // (rounded_vocab_size)
    // internal state, separate memory for each layer
    float *conv_state; // (n_layers, d_inner, d_conv)
    float *ssm_state;  // (n_layers, d_inner, d_state)
} RunStateFloat;

typedef struct
{
    // memory reused by all layers
    fixed_t *input;        // (dim)
    fixed_t *hidden_state; // (dim)
    fixed_t *xz;           // (2*d_inner)          x and z are pointers into this buffer
    fixed_t *x_db;         // (dt_rank+2*d_state)  dt, B, C are pointers into this buffer
    fixed_t *dt;           // (d_inner)            later, dt is a pointer to this buffer
    fixed_t *dA;           // (d_inner, d_state)
    fixed_t *dB;           // (d_inner, d_state)
    fixed_t *temp;         // (d_inner, d_state)
    fixed_t *y;            // (d_inner)
    fixed_t *logits;       // (rounded_vocab_size)
    // internal state, separate memory for each layer
    fixed_t *conv_state; // (n_layers, d_inner, d_conv)
    fixed_t *ssm_state;  // (n_layers, d_inner, d_state)
} RunState;

typedef struct
{
    Config config;                   // the hyperparameters of the architecture (the blueprint)
    MambaWeights weights;            // the weights of the model
    MambaWeightsFloat weights_float; // the weights of the model in float
    RunState state;                  // buffers for the "wave" of activations in the forward pass
    RunStateFloat state_float;       // buffers for the "wave" of activations in the forward pass in float
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;            // file descriptor for memory mapping
    float *data;       // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Mamba;

void malloc_run_state(RunState *s, RunStateFloat *sf, Config *p)
{
    // memory reused by all layers
    s->input = malloc(p->dim * sizeof(fixed_t));
    s->hidden_state = malloc(p->dim * sizeof(fixed_t));
    s->xz = malloc(2 * p->d_inner * sizeof(fixed_t));
    s->x_db = malloc((p->dt_rank + 2 * p->d_state) * sizeof(fixed_t));
    s->dt = malloc(p->d_inner * sizeof(fixed_t));
    s->dA = malloc(p->d_inner * p->d_state * sizeof(fixed_t));
    s->dB = malloc(p->d_inner * p->d_state * sizeof(fixed_t));
    s->temp = malloc(p->d_inner * p->d_state * sizeof(fixed_t));
    s->y = malloc(p->d_inner * sizeof(fixed_t));
    s->logits = malloc(p->rounded_vocab_size * sizeof(fixed_t));
    // internal state, separate memory for each layer
    s->conv_state = calloc(p->n_layers * p->d_inner * p->d_conv, sizeof(fixed_t));
    s->ssm_state = calloc(p->n_layers * p->d_inner * p->d_state, sizeof(fixed_t));
    // ensure all mallocs went fine
    if (!s->xz || !s->x_db || !s->dt || !s->dA || !s->dB || !s->temp || !s->y || !s->logits || !s->conv_state || !s->ssm_state)
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    sf->input = malloc(p->dim * sizeof(float));
    sf->hidden_state = malloc(p->dim * sizeof(float));
    sf->xz = malloc(2 * p->d_inner * sizeof(float));
    sf->x_db = malloc((p->dt_rank + 2 * p->d_state) * sizeof(float));
    sf->dt = malloc(p->d_inner * sizeof(float));
    sf->dA = malloc(p->d_inner * p->d_state * sizeof(float));
    sf->dB = malloc(p->d_inner * p->d_state * sizeof(float));
    sf->temp = malloc(p->d_inner * p->d_state * sizeof(float));
    sf->y = malloc(p->d_inner * sizeof(float));
    sf->logits = malloc(p->rounded_vocab_size * sizeof(float));
    // internal state, separate memory for each layer
    sf->conv_state = calloc(p->n_layers * p->d_inner * p->d_conv, sizeof(float));
    sf->ssm_state = calloc(p->n_layers * p->d_inner * p->d_state, sizeof(float));
    // ensure all mallocs went fine
    if (!sf->xz || !sf->x_db || !sf->dt || !sf->dA || !sf->dB || !sf->temp || !sf->y || !sf->logits || !sf->conv_state || !sf->ssm_state)
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void reset_internal_state(Mamba *mamba)
{
    // reset the internal state of the model
    RunState *s = &mamba->state;
    Config *p = &mamba->config;
    memset(s->conv_state, 0, p->n_layers * p->d_inner * p->d_conv * sizeof(float));
    memset(s->ssm_state, 0, p->n_layers * p->d_inner * p->d_state * sizeof(float));
}

char *get_internal_state(Mamba *mamba, int *state_size)
{
    // get the internal state of the model
    Config *p = &mamba->config;
    RunState *s = &mamba->state;
    unsigned int conv_state_size = p->n_layers * p->d_inner * p->d_conv * sizeof(float);
    unsigned int ssm_state_size = p->n_layers * p->d_inner * p->d_state * sizeof(float);
    unsigned int total_size = conv_state_size + ssm_state_size;
    char *state = malloc(total_size);
    if (state)
    {
        memcpy(state, s->conv_state, conv_state_size);
        memcpy(state + conv_state_size, s->ssm_state, ssm_state_size);
        *state_size = total_size;
    }
    return state;
}

void set_internal_state(Mamba *mamba, char *state, int state_size)
{
    // set the internal state of the model
    Config *p = &mamba->config;
    RunState *s = &mamba->state;
    unsigned int conv_state_size = p->n_layers * p->d_inner * p->d_conv * sizeof(float);
    unsigned int ssm_state_size = p->n_layers * p->d_inner * p->d_state * sizeof(float);
    if (state_size == conv_state_size + ssm_state_size)
    {
        memcpy(s->conv_state, state, conv_state_size);
        memcpy(s->ssm_state, state + conv_state_size, ssm_state_size);
    }
}

void free_run_state(RunState *s)
{
    free(s->input);
    free(s->hidden_state);
    free(s->xz);
    free(s->x_db);
    free(s->dt);
    free(s->dA);
    free(s->dB);
    free(s->temp);
    free(s->y);
    free(s->logits);
    free(s->conv_state);
    free(s->ssm_state);
}

void memory_map_weights(MambaWeightsFloat *w, Config *p, float *ptr)
{
    // the multiplications below are done in 64-bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    // get the pointers to the weights
    w->token_embedding_table = ptr;
    ptr += p->rounded_vocab_size * p->dim;
    w->in_proj = ptr;
    ptr += n_layers * (2 * p->d_inner) * p->dim;
    w->conv1d_weight = ptr;
    ptr += n_layers * p->d_inner * 1 * p->d_conv;
    w->conv1d_bias = ptr;
    ptr += n_layers * p->d_inner;
    w->x_proj = ptr;
    ptr += n_layers * (p->dt_rank + 2 * p->d_state) * p->d_inner;
    w->dt_proj_weight = ptr;
    ptr += n_layers * p->d_inner * p->dt_rank;
    w->dt_proj_bias = ptr;
    ptr += n_layers * p->d_inner;
    w->A = ptr;
    ptr += n_layers * p->d_inner * p->d_state;
    w->D = ptr;
    ptr += n_layers * p->d_inner;
    w->out_proj = ptr;
    ptr += n_layers * p->dim * p->d_inner;
    w->norm = ptr;
    ptr += n_layers * p->dim;
    w->final_norm = ptr;
    ptr += p->dim;
    // the classifier weights can be shared with the token embedding table
    w->lm_head = p->shared_classifier ? w->token_embedding_table : ptr;
}

void load_model_file(char *model_path, Config *config, MambaWeights *weights, MambaWeightsFloat *weights_float,
                     int *fd, float **data, ssize_t *file_size)
{
    FILE *file = fopen(model_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Couldn't open file %s\n", model_path);
        exit(EXIT_FAILURE);
    }
    // read the magic number
    unsigned int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    if (magic != 0x4d616d62)
    {
        fprintf(stderr, "Invalid magic number: %x\n", magic);
        exit(EXIT_FAILURE);
    }
    // read the version
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    if (version != 1)
    {
        fprintf(stderr, "Invalid version: %d\n", version);
        exit(EXIT_FAILURE);
    }
    // read the config
    if (fread(config, sizeof(Config), 1, file) != 1)
    {
        exit(EXIT_FAILURE);
    }
    if (config->vocab_size % 8 != 0)
    {
        config->rounded_vocab_size = config->vocab_size + (8 - (config->vocab_size % 8));
    }
    else
    {
        config->rounded_vocab_size = config->vocab_size;
    }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the model weights into the data pointer
    *fd = open(model_path, O_RDONLY); // open in read only mode
    if (*fd == -1)
    {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = (float *)malloc(*file_size);
    if (*data == NULL)
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
    if (read(*fd, *data, *file_size) != *file_size)
    {
        fprintf(stderr, "read failed!\n");
        exit(EXIT_FAILURE);
    }
    float *weights_float_ptr = *data + (256 / 4);
    memory_map_weights(weights_float, config, weights_float_ptr);

    // convert the weights to fixed-point
    weights->token_embedding_table = malloc(config->rounded_vocab_size * config->dim * sizeof(fixed_t));
    weights->in_proj = malloc(config->n_layers * 2 * config->d_inner * config->dim * sizeof(fixed_t));
    weights->conv1d_weight = malloc(config->n_layers * config->d_inner * 1 * config->d_conv * sizeof(fixed_t));
    weights->conv1d_bias = malloc(config->n_layers * config->d_inner * sizeof(fixed_t));
    weights->x_proj = malloc(config->n_layers * (config->dt_rank + 2 * config->d_state) * config->d_inner * sizeof(fixed_t));
    weights->dt_proj_weight = malloc(config->n_layers * config->d_inner * config->dt_rank * sizeof(fixed_t));
    weights->dt_proj_bias = malloc(config->n_layers * config->d_inner * sizeof(fixed_t));
    weights->A = malloc(config->n_layers * config->d_inner * config->d_state * sizeof(fixed_t));
    weights->D = malloc(config->n_layers * config->d_inner * sizeof(fixed_t));
    weights->out_proj = malloc(config->n_layers * config->dim * config->d_inner * sizeof(fixed_t));
    weights->norm = malloc(config->n_layers * config->dim * sizeof(fixed_t));
    weights->final_norm = malloc(config->dim * sizeof(fixed_t));
    weights->lm_head = malloc(config->rounded_vocab_size * config->dim * sizeof(fixed_t));

    for (int i = 0; i < config->rounded_vocab_size * config->dim; i++)
    {
        weights->token_embedding_table[i] = FLOAT_TO_FIXED(weights_float->token_embedding_table[i]);
    }
    for (int i = 0; i < config->n_layers * 2 * config->d_inner * config->dim; i++)
    {
        weights->in_proj[i] = FLOAT_TO_FIXED(weights_float->in_proj[i]);
    }
    for (int i = 0; i < config->n_layers * config->d_inner * 1 * config->d_conv; i++)
    {
        weights->conv1d_weight[i] = FLOAT_TO_FIXED(weights_float->conv1d_weight[i]);
    }
    for (int i = 0; i < config->n_layers * config->d_inner; i++)
    {
        weights->conv1d_bias[i] = FLOAT_TO_FIXED(weights_float->conv1d_bias[i]);
    }
    for (int i = 0; i < config->n_layers * (config->dt_rank + 2 * config->d_state) * config->d_inner; i++)
    {
        weights->x_proj[i] = FLOAT_TO_FIXED(weights_float->x_proj[i]);
    }
    for (int i = 0; i < config->n_layers * config->d_inner * config->dt_rank; i++)
    {
        weights->dt_proj_weight[i] = FLOAT_TO_FIXED(weights_float->dt_proj_weight[i]);
    }
    for (int i = 0; i < config->n_layers * config->d_inner; i++)
    {
        weights->dt_proj_bias[i] = FLOAT_TO_FIXED(weights_float->dt_proj_bias[i]);
    }
    for (int i = 0; i < config->n_layers * config->d_inner * config->d_state; i++)
    {
        weights->A[i] = FLOAT_TO_FIXED(weights_float->A[i]);
    }
    for (int i = 0; i < config->n_layers * config->d_inner; i++)
    {
        weights->D[i] = FLOAT_TO_FIXED(weights_float->D[i]);
    }
    for (int i = 0; i < config->n_layers * config->dim * config->d_inner; i++)
    {
        weights->out_proj[i] = FLOAT_TO_FIXED(weights_float->out_proj[i]);
    }
    for (int i = 0; i < config->n_layers * config->dim; i++)
    {
        weights->norm[i] = FLOAT_TO_FIXED(weights_float->norm[i]);
    }
    for (int i = 0; i < config->dim; i++)
    {
        weights->final_norm[i] = FLOAT_TO_FIXED(weights_float->final_norm[i]);
    }
    for (int i = 0; i < config->rounded_vocab_size * config->dim; i++)
    {
        weights->lm_head[i] = FLOAT_TO_FIXED(weights_float->lm_head[i]);
    }
}

void load_model(Mamba *m, char *model_path)
{
    // read the Config and the Weights from the model file
    load_model_file(model_path, &m->config, &m->weights, &m->weights_float, &m->fd, &m->data, &m->file_size);
    // allocate the RunState buffers
    malloc_run_state(&m->state, &m->state_float, &m->config);
}

void free_model(Mamba *m)
{
    // close the memory mapping
    free(m->data);
    if (m->fd != -1)
    {
        close(m->fd);
    }
    // free the RunState buffers
    free_run_state(&m->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the model

void rmsnorm(fixed_t *o, fixed_t *x, fixed_t *weight, int size)
{
    // calculate sum of squares
    // float ss = 0.0f;
    fixed_t ss = 0;
    for (int j = 0; j < size; j++)
    {
        // ss += x[j] * x[j];
        ss += FIXED_MUL(x[j], x[j]);
    }
    // ss /= size;
    ss = FIXED_DIV(ss, size);
    ss += 1e-5f;
    // ss = 1.0f / sqrtf(ss);
    ss = FIXED_DIV(FIXED_ONE, sqrtf(FIXED_TO_FLOAT(ss)));
    // normalize and scale
    for (int j = 0; j < size; j++)
    {
        // o[j] = x[j] * weight[j] * ss;
        o[j] = FIXED_MUL(FIXED_MUL(x[j], weight[j]), ss);
    }
}

void softmax(float *x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

fixed_t softplus(fixed_t x)
{
    return FLOAT_TO_FIXED(logf(1.0f + expf(FIXED_TO_FLOAT(x))));
}

fixed_t sigmoid(fixed_t x)
{
    return FLOAT_TO_FIXED(1.0f / (1.0f + expf(-FIXED_TO_FLOAT(x))));
}

fixed_t silu(fixed_t x)
{
    return x * sigmoid(x);
}

void shift_matrix_left(fixed_t *matrix, int rows, int cols)
{
    // #pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols - 1; j++)
        {
            matrix[i * cols + j] = matrix[i * cols + j + 1];
        }
    }
}

void update_last_column(fixed_t *matrix, fixed_t *x, int rows, int cols)
{
    // #pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        matrix[i * cols + cols - 1] = x[i];
    }
}

void rowwise_dot_product(fixed_t *out, fixed_t *matrix, fixed_t *weights, int rows, int cols)
{
    // matrix[rows,cols], weights[cols] -> out[rows]
    // this is a dot product of each row of the matrix with the weights
    // i.e. out[i] = matrix[i,:] @ weights
    // #pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        fixed_t val = 0;
        for (int j = 0; j < cols; j++)
        {
            // val += matrix[i * cols + j] * weights[j];
            val += FIXED_MUL(matrix[i * cols + j], weights[j]);
        }
        out[i] = val;
    }
}

void matmul(fixed_t *xout, fixed_t *x, fixed_t *w, int d, int n)
{
    // if (d <= 100)
    // {
    //     for (int i = 0; i < d; i++)
    //     {
    //         for (int j = 0; j < n; j++)
    //         {
    //             fprintf(stderr, "w[%d][%d] = %f\n", i, j, w[i * n + j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // w[d,n] @ x[n] -> xout[d]
    // #pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        fixed_t val = 0;
        for (int j = 0; j < n; j++)
        {
            // val += w[i * n + j] * x[j];
            val += FIXED_MUL(w[i * n + j], x[j]);
        }
        xout[i] = val;
    }
}

void linear(fixed_t *xout, fixed_t *x, fixed_t *w, fixed_t *b, int d, int n)
{
    // w[d,n] @ x[n] + b[d] -> xout[d]
    // #pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        fixed_t val = 0;
        for (int j = 0; j < n; j++)
        {
            // val += w[i * n + j] * x[j];
            val += FIXED_MUL(w[i * n + j], x[j]);
        }
        xout[i] = val + b[i];
    }
}

void broadcast_multiply(fixed_t *out, fixed_t *x, fixed_t *y, int d, int n)
{
    // x[d], y[d,n] -> out[d,n]
    // #pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int index = i * n + j;
            // out[index] = x[i] * y[index];
            out[index] = FIXED_MUL(x[i], y[index]);
            // out[i * n + j] = x[i] * y[i * n + j];
        }
    }
}

void elementwise_multiply(fixed_t *result, fixed_t *matrix1, fixed_t *matrix2, int total_elements)
{
    // #pragma omp parallel for
    for (int i = 0; i < total_elements; i++)
    {
        // result[i] = matrix1[i] * matrix2[i];
        result[i] = FIXED_MUL(matrix1[i], matrix2[i]);
    }
}

void elementwise_add(fixed_t *result, fixed_t *matrix1, fixed_t *matrix2, int total_elements)
{
    // #pragma omp parallel for
    for (int i = 0; i < total_elements; i++)
    {
        // result[i] = matrix1[i] + matrix2[i];
        result[i] = matrix1[i] + matrix2[i];
    }
}

void elementwise_multiply_and_add(fixed_t *result, fixed_t *matrix1, fixed_t *matrix2, fixed_t *matrix3, int total_elements)
{
    // #pragma omp parallel for
    for (int i = 0; i < total_elements; i++)
    {
        // result[i] = matrix1[i] * matrix2[i] + matrix3[i];
        result[i] = FIXED_MUL(matrix1[i], matrix2[i]) + matrix3[i];
    }
}

void outer_product(fixed_t *out, fixed_t *x, fixed_t *y, int d, int n)
{
    // x[d], y[n] -> out[d,n]
    // #pragma omp parallel for
    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < n; j++)
        {
            out[i * n + j] = FIXED_MUL(x[i], y[j]);
        }
    }
}

void sum_along_last_dim(fixed_t *result, fixed_t *matrix, int rows, int cols)
{
    // #pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        fixed_t val = 0;
        for (int j = 0; j < cols; j++)
        {
            // val += matrix[i * cols + j];
            val += matrix[i * cols + j];
        }
        result[i] = val;
    }
}

void forward_layer(Mamba *mamba, unsigned long long l, fixed_t *hidden_state)
{
    Config *p = &mamba->config;
    MambaWeights *w = &mamba->weights;
    RunState *s = &mamba->state;
    int dim = p->dim, d_inner = p->d_inner, d_conv = p->d_conv, d_state = p->d_state, dt_rank = p->dt_rank;
    fixed_t *dA = s->dA; // (d_inner, d_state)
    fixed_t *dB = s->dB; // (d_inner, d_state)
    fixed_t *y = s->y;   // (d_inner)

    // conv_state, ssm_state = self._get_states_from_cache(inference_params)
    fixed_t *conv_state = s->conv_state + l * d_inner * d_conv;
    fixed_t *ssm_state = s->ssm_state + l * d_inner * d_state;

    // xz = self.in_proj(hidden_states)  # hidden_states: (dim), in_proj (2*d_inner, dim), xz (2*d_inner)
    matmul(s->xz, hidden_state, w->in_proj + l * 2 * d_inner * dim, 2 * d_inner, dim);
    // x, z = xz.chunk(2, dim=-1)
    fixed_t *x = s->xz;           // x (d_inner)
    fixed_t *z = s->xz + d_inner; // z (d_inner)

    // Conv step

    // conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    shift_matrix_left(conv_state, d_inner, d_conv);
    // conv_state[:, -1] = x
    update_last_column(conv_state, x, d_inner, d_conv);
    // x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
    elementwise_multiply(s->temp, conv_state, w->conv1d_weight + l * d_inner * d_conv, d_inner * d_conv);
    sum_along_last_dim(x, s->temp, d_inner, d_conv);
    // x = x + self.conv1d.bias
    elementwise_add(x, x, w->conv1d_bias + l * d_inner, d_inner);
    // x = F.silu(x)
    for (int i = 0; i < d_inner; i++)
    {
        x[i] = silu(x[i]);
    }

    // SSM step

    // x_db = self.x_proj(x)   # x_db (dt_rank+2*d_state)
    matmul(s->x_db, x, w->x_proj + l * (dt_rank + 2 * d_state) * d_inner, dt_rank + 2 * d_state, d_inner);
    // dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    fixed_t *dt = s->x_db;                    // dt (dt_rank)
    fixed_t *B = s->x_db + dt_rank;           // B  (d_state)
    fixed_t *C = s->x_db + dt_rank + d_state; // C  (d_state)

    // dt = self.dt_proj(dt)   # dt (dt_rank), dt_proj_weight (d_inner, dt_rank), dt_proj_bias (d_inner)
    linear(s->dt, dt, w->dt_proj_weight + l * d_inner * dt_rank, w->dt_proj_bias + l * d_inner, d_inner, dt_rank);
    dt = s->dt; // NOTE: dt is now bigger: (d_inner) instead of (dt_rank)
    // dt = F.softplus(dt)
    for (int i = 0; i < d_inner; i++)
    {
        dt[i] = softplus(dt[i]);
    }

    //  Discretize A and B
    // dA = torch.exp(torch.einsum("d,dn->dn", dt, self.A))   # A (d_inner, d_state), dA (d_inner, d_state)
    broadcast_multiply(dA, dt, w->A + l * d_inner * d_state, d_inner, d_state);
    for (int i = 0; i < d_inner * d_state; i++)
    {
        dA[i] = FLOAT_TO_FIXED(expf(FIXED_TO_FLOAT(dA[i])));
    }
    // dB = torch.einsum("d,n->dn", dt, B)    # dt (d_inner), B (d_state), dB (d_inner, d_state)
    B = malloc(d_state * sizeof(fixed_t));
    outer_product(dB, dt, B, d_inner, d_state);

    //  Update ssm_state
    // ssm_state.copy_(ssm_state * dA + rearrange(x, "d -> d 1") * dB)
    broadcast_multiply(s->temp, x, dB, d_inner, d_state);
    elementwise_multiply_and_add(ssm_state, ssm_state, dA, s->temp, d_inner * d_state);

    //  Compute y
    // y = torch.einsum("dn,n->d", ssm_state, C) # ssm_state (d_inner, d_state), C (d_state), y (d_inner)
    rowwise_dot_product(y, ssm_state, C, d_inner, d_state);
    // y = y + self.D * x
    elementwise_multiply_and_add(y, w->D + l * d_inner, x, y, d_inner);
    // y = y * F.silu(z)  # (d_inner)
    for (int i = 0; i < d_inner; i++)
    {
        y[i] = y[i] * silu(z[i]);
    }

    // hidden_state = self.out_proj(y)  # out_proj (dim, d_inner), hidden_state (dim)
    matmul(hidden_state, y, w->out_proj + l * dim * d_inner, dim, d_inner);
}

float *forward(Mamba *mamba, int token)
{
    // a few convenience variables
    Config *p = &mamba->config;
    MambaWeights *w = &mamba->weights;
    MambaWeightsFloat *w_float = &mamba->weights_float;
    RunState *s = &mamba->state;
    RunStateFloat *sf = &mamba->state_float;
    int dim = p->dim;
    float *input_float = sf->input;
    fixed_t *input = s->input;
    fixed_t *hidden_state = s->hidden_state;

    // copy the token embedding into x
    float *content_row = w_float->token_embedding_table + token * dim;
    memcpy(input_float, content_row, dim * sizeof(float));

    // convert the input to fixed-point
    for (int i = 0; i < dim; i++)
    {
        input[i] = FLOAT_TO_FIXED(input_float[i]);
    }

    // forward all the layers
    for (unsigned long long l = 0; l < p->n_layers; l++)
    {
        // normalize the input
        rmsnorm(hidden_state, input, w->norm + l * dim, dim);
        // forward this layer
        forward_layer(mamba, l, hidden_state);
        // residual connection back into hidden_state
        for (int i = 0; i < dim; i++)
        {
            hidden_state[i] += input[i];
            // copy hidden_state back into input for the next layer
            input[i] = hidden_state[i];
        }
    }

    // final rmsnorm
    rmsnorm(hidden_state, hidden_state, w->final_norm, dim);

    // classifier into logits
    matmul(s->logits, hidden_state, w->lm_head, p->rounded_vocab_size, p->dim);
    fixed_to_float_arr(s->logits, sf->logits, p->rounded_vocab_size);
    return sf->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

#define BOS 0
#define EOS 0

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char **vocab;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int model_vocab_size)
{
    // initialize the byte_pieces array
    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    // read header magic
    unsigned int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    if (magic != 0x4d62546b)
    {
        fprintf(stderr, "invalid magic number: %x\n", magic);
        exit(EXIT_FAILURE);
    }
    // read version
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    if (version != 1)
    {
        fprintf(stderr, "invalid version: %d\n", version);
        exit(EXIT_FAILURE);
    }
    // read vocab_size
    int vocab_size;
    if (fread(&vocab_size, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    // read max_token_length
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    // malloc space for the vocab
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    if (!t->vocab)
    {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    t->sorted_vocab = NULL; // initialized lazily
    // read vocab
    int len;
    for (int i = 0; i < vocab_size; i++)
    {
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
    char *piece = t->vocab[token];
    // discard initial space if prev_token was EOS
    if (prev_token == EOS && piece[0] == ' ')
    {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
    {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece)
{
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL)
    {
        return;
    }
    if (piece[0] == '\0')
    {
        return;
    }
    if (piece[1] == '\0')
    {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val)))
        {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str}; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t add_bos, int8_t add_eos, int *tokens, int *n_tokens)
{
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // add_bos != 0 means prepend the BOS token, add_eos != 0 means append the EOS token
    if (text == NULL)
    {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL)
    {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++)
        {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS token, if desired
    if (add_bos)
        tokens[(*n_tokens)++] = BOS;

    // add_dummy_prefix is not used in Mamba, but it's here for reference
    // prepend a dummy prefix token to the input string, but only if text != ""
    int add_dummy_prefix = 0;
    if (add_dummy_prefix && text[0] != '\0')
    {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point â UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++)
    {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80)
        {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
        {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1)
        {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        }
        else
        {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++)
            {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration
    while (1)
    {
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++)
        {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1)
            {
                // this merge pair exists in vocab! record its position
                best_id = id;
                best_idx = i;
                break;
            }
        }

        if (best_idx == -1)
        {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
        {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS token, if desired
    if (add_eos)
        tokens[(*n_tokens)++] = EOS;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct
{
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct
{
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n)
{
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
    {
        if (probabilities[i] > max_p)
        {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float *probabilities, int n, float coin)
{
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += probabilities[i];
        if (coin < cdf)
        {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b)
{
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        if (probabilities[i] >= cutoff)
        {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++)
    {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp)
        {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++)
    {
        cdf += probindex[i].prob;
        if (r < cdf)
        {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed)
{
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler)
{
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state)
{
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state)
{ // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits)
{
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f)
    {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    }
    else
    {
        // apply the temperature to the logits
        for (int q = 0; q < sampler->vocab_size; q++)
        {
            logits[q] /= sampler->temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1)
        {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        }
        else
        {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

// long time_in_ms()
// {
//     // return time in milliseconds, for benchmarking the model speed
//     struct timespec time;
//     clock_gettime(CLOCK_REALTIME, &time);
//     return time.tv_sec * 1000 + time.tv_nsec / 1000000;
// }

// ----------------------------------------------------------------------------
// generation loop

float generate(Mamba *mamba, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
    char *empty_prompt = "";
    if (prompt == NULL)
    {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', BOS, EOS
    encode(tokenizer, prompt, 0, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1)
    {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // print the first token in the prompt
    if (num_prompt_tokens > 1)
    {
        char *piece = decode(tokenizer, EOS, prompt_tokens[0]);
        // safe_printf(piece);
        // fflush(stdout);
    }

    // Buffer to store all generated tokens
    const char *tokens_file = "generated_tokens.txt";
    int *generated_tokens = (int *)malloc(steps * sizeof(int));
    int generated_count = 0;

    // start the main loop
    long start = 0;               // used to time our code, only initialized after first iteration
    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence
    while (pos < steps)
    {

        // forward the model to get logits for the next token
        float *logits = forward(mamba, token);

        // advance the state machine
        if (pos < num_prompt_tokens - 1)
        {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        }
        else
        {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the EOS token delimits sequences
        if (next == EOS)
        {
            break;
        }

        // Store the generated token
        generated_tokens[generated_count++] = next;

        // print the token as string, decode it with the Tokenizer object
        char *piece = decode(tokenizer, token, next);
        // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        // fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        // if (start == 0)
        // {
        //     start = time_in_ms();
        // }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    // if (pos > 1)
    // {
    //     long end = time_in_ms();
    //     fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    // }

    free(prompt_tokens);

    // Compute and print the MSE
    float mse = calculate_mse(tokens_file, generated_tokens, generated_count, mamba->weights_float.token_embedding_table, mamba->config.dim);
    printf("MSE: %f\n", mse);

    return mse;
}

void read_stdin(const char *guide, char *buffer, size_t bufsize)
{
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL)
    {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n')
        {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Mamba *mamba, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps)
{

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;             // will store the next token in the sequence
    int token;            // stores the current token to feed into the model
    int prev_token;
    int pos = 0; // position in the sequence
    while (pos < steps)
    {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn)
        {
            // get the (optional) system prompt at position 0
            if (pos == 0)
            {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL)
                {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                }
                else
                {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL)
            {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            }
            else
            {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0')
            {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            }
            else
            {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 0, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the model next
        if (user_idx < num_prompt_tokens)
        {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        }
        else
        {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS token ends the Assistant turn
        if (token == EOS)
        {
            user_turn = 1;
        }

        // forward the model to get logits for the next token
        float *logits = forward(mamba, token);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != EOS)
        {
            // the Assistant is responding, so print its output
            char *piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == EOS)
        {
            printf("\n");
        }
    }
    printf("\n");
    free(prompt_tokens);
}

float calculate_mse(const char *filename, int *tokens, int num_tokens, float *embedding_table, int embedding_dim)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Couldn't open file %s for reading\n", filename);
        exit(EXIT_FAILURE);
    }

    int *stored_tokens = (int *)malloc(num_tokens * sizeof(int));
    if (stored_tokens == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    int count = 0;
    while (fscanf(file, "%d", &stored_tokens[count]) == 1)
    {
        count++;
        if (count > num_tokens)
        {
            fprintf(stderr, "Number of tokens in file exceeds expected count\n");
            // free(stored_tokens);
            // fclose(file);
            return -1;
            break;
        }
    }

    // if (count != num_tokens)
    // {
    //     fprintf(stderr, "Number of tokens in file does not match expected count\n");
    //     free(stored_tokens);
    //     fclose(file);
    //     exit(EXIT_FAILURE);
    // }

    fclose(file);

    float mse = 0.0f;
    for (int i = 0; i < num_tokens; i++)
    {
        float diff_sum = 0.0f;
        for (int j = 0; j < embedding_dim; j++)
        {
            float diff = embedding_table[tokens[i] * embedding_dim + j] - embedding_table[stored_tokens[i] * embedding_dim + j];
            diff_sum += diff * diff;
        }
        mse += diff_sum;
    }

    mse /= num_tokens;

    free(stored_tokens);
    return mse;
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage()
{
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    // fprintf(stderr, "e^20: %f\n", exp(20));
    // float a = 12;
    // fixed_t b = FLOAT_TO_FIXED(a);
    // fixed_t c = FIXED_DIV(b, FLOAT_TO_FIXED(3));
    // fixed_t d = exp_fixed(b);
    // fprintf(stderr, "a: %f, b: %f, c: %f, d: %f\n", a, FIXED_TO_FLOAT(b), FIXED_TO_FLOAT(c), FIXED_TO_FLOAT(d));
    // exit(EXIT_FAILURE);

    // default parameters
    char *model_path = NULL; // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.1f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;                 // number of steps to run for
    char *prompt = NULL;             // prompt string
    unsigned long long rng_seed = 1; // seed rng with time by default
    char *mode = "generate";         // generate|chat
    char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode

    model_path = "model.bin";
    steps = 50;
    prompt = "Customer Support should";
    temperature = 0.0;

    // poor man's C argparse so we can override the defaults above from the command line
    // if (argc >= 2)
    // {
    //     model_path = argv[1];
    // }
    // else
    // {
    //     error_usage();
    // }
    // for (int i = 2; i < argc; i += 2)
    // {
    //     // do some basic validation
    //     if (i + 1 >= argc)
    //     {
    //         error_usage();
    //     } // must have arg after flag
    //     if (argv[i][0] != '-')
    //     {
    //         error_usage();
    //     } // must start with dash
    //     if (strlen(argv[i]) != 2)
    //     {
    //         error_usage();
    //     } // must be -x (one dash, one letter)
    //     // read in the args
    //     if (argv[i][1] == 't')
    //     {
    //         temperature = atof(argv[i + 1]);
    //     }
    //     else if (argv[i][1] == 'p')
    //     {
    //         topp = atof(argv[i + 1]);
    //     }
    //     else if (argv[i][1] == 's')
    //     {
    //         rng_seed = atoi(argv[i + 1]);
    //     }
    //     else if (argv[i][1] == 'n')
    //     {
    //         steps = atoi(argv[i + 1]);
    //     }
    //     else if (argv[i][1] == 'i')
    //     {
    //         prompt = argv[i + 1];
    //     }
    //     else if (argv[i][1] == 'z')
    //     {
    //         tokenizer_path = argv[i + 1];
    //     }
    //     else if (argv[i][1] == 'm')
    //     {
    //         mode = argv[i + 1];
    //     }
    //     else if (argv[i][1] == 'y')
    //     {
    //         system_prompt = argv[i + 1];
    //     }
    //     else
    //     {
    //         error_usage();
    //     }
    // }

    // parameter validation/overrides
    if (rng_seed <= 0)
        rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0)
        temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp)
        topp = 0.9;
    if (steps < 0)
        steps = 0;

    // load the model using the model.bin file
    Mamba mamba;
    load_model(&mamba, model_path);

    // print the config
    fprintf(stderr, "config: vocab_size=%d (%d), n_layers=%d, dim=%d, d_inner=%d, dt_rank=%d, d_state=%d, d_conv=%d\n",
            mamba.config.vocab_size, mamba.config.rounded_vocab_size, mamba.config.n_layers, mamba.config.dim, mamba.config.d_inner, mamba.config.dt_rank, mamba.config.d_state, mamba.config.d_conv);

    if (steps == 0)
        steps = 256; // override to default len if 0

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, mamba.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, mamba.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0)
    {
        int steps = 50;
        float mse = 0.0f;
        for (int i = 0; i < steps; i++)
        {
            float current_mse = generate(&mamba, &tokenizer, &sampler, prompt, steps);
            if (current_mse > 0)
            {
                mse += current_mse;
            }
        }
        mse /= steps;
        printf("Average MSE: %f\n", mse);
    }
    else if (strcmp(mode, "chat") == 0)
    {
        chat(&mamba, &tokenizer, &sampler, prompt, system_prompt, steps);
    }
    else
    {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_model(&mamba);
    return 0;
}
#endif