#include "Python.h"
#include <stdlib.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "art.h"

// compile me with
// python ./setup.py build_ext --inplace

// Idea: Use a Trie data structure to store the subsequence histogram.

typedef unsigned int uint;
typedef unsigned char uint8;
typedef char int8;
typedef uint32_t uint32;
typedef uint64_t uint64;

int validate_seqmat(PyObject *o, int *pL)
{
    if (!PyArray_Check(o)) {
        PyErr_SetString(PyExc_ValueError, "seq must be 2d uint8 ndarray");
        return -1;
    }
    PyArrayObject *a = (PyArrayObject *)o;

    if ((PyArray_NDIM(a) != 2) || PyArray_TYPE(a) != NPY_UINT8 ) {
        PyErr_SetString(PyExc_ValueError, "seq must be 2d uint8 array");
        return -1;
    }

    if (!PyArray_ISCARRAY(a)) {
        PyErr_SetString(PyExc_ValueError, "seq must be C-contiguous");
        return -1;
    }

    int nseq = PyArray_DIM(a, 0);
    int L = PyArray_DIM(a, 1);

    if (nseq == 0) {
        PyErr_SetString(PyExc_ValueError, "0 sequences found");
        return -1;
    }
    
    if (*pL == -1) {
        *pL = L;
    }
    else if (L != *pL) {
        PyErr_SetString(PyExc_ValueError, "seqs have inconsistent L");
        return -1;
    }

    return 0;
}

int validate_wgt(PyObject *wgt, PyObject *msa)
{
    if (wgt == Py_None) {
        return 0;
    }

    if (!PyArray_Check(wgt)) {
        PyErr_SetString(PyExc_ValueError, "weight must be 1d float32 ndarray");
        return -1;
    }
    PyArrayObject *a = (PyArrayObject *)wgt;

    if ((PyArray_NDIM(a) != 1) || PyArray_TYPE(a) != NPY_FLOAT32 ) {
        PyErr_SetString(PyExc_ValueError, "weight must be 1d float32 ndarray");
        return -1;
    }

    if (!PyArray_ISCARRAY(a)) {
        PyErr_SetString(PyExc_ValueError, "weight must be C-contiguous");
        return -1;
    }

    if (PyArray_DIM(a, 0) != PyArray_DIM((PyArrayObject*)msa, 0)) {
        PyErr_SetString(PyExc_ValueError, "weight of wrong length");
        return -1;
    }
    return 0;
}

static PyObject *
highmarg(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *seq_arrays;
    PyObject *weight_arr = Py_None;
    int return_uniq = 0;
    static char *kwlist[] = {"seqmats", "weights", "return_uniq", NULL};

    // input should be a list of seq matrices, and an integer q
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", kwlist,
                                     &seq_arrays, &weight_arr, &return_uniq)) {
        return NULL;
    }
    
    // some preliminary validation of inputs
    if (!PyList_Check(seq_arrays)) {
        PyErr_SetString(PyExc_ValueError,
                        "seqmats must be a list of ndarrays");
        return NULL;
    }

    if (weight_arr != Py_None && !PyList_Check(weight_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "weights must be a list of ndarrays");
        return NULL;
    }

    int n_msas = PyList_Size(seq_arrays);
    if (n_msas < 0) {
        return NULL;
    }
    if (n_msas == 0) {
        PyErr_SetString(PyExc_ValueError, "empty list supplied");
        return NULL;
    }
    if (weight_arr != Py_None) {
        int n_weights = PyList_Size(seq_arrays);
        if (n_weights < 0) {
            return NULL;
        }
        if (n_weights != n_msas) {
            PyErr_SetString(PyExc_ValueError, "weights of wrong length");
            return NULL;
        }
    }
    
    // validate input arrays
    int L = -1;
    int tot_nseq = 0;
    for (int msa_i = 0; msa_i < n_msas; msa_i++) {
        PyObject *msa = PyList_GetItem(seq_arrays, msa_i);
        if (msa == NULL || validate_seqmat(msa, &L) < 0) {
            return NULL;
        }
        tot_nseq += PyArray_DIM((PyArrayObject *)msa, 0);

        if (weight_arr != Py_None) {
            PyObject *wgt = PyList_GetItem(weight_arr, msa_i);
            if (wgt == NULL || validate_wgt(wgt, msa) < 0) {
                return NULL;
            }
        }
    }

    // allocate a histogram for each input MSA, combined in one memory block.
    float *histdat = malloc(sizeof(*histdat)*tot_nseq*n_msas);
    if (histdat == NULL) {
        PyErr_SetString(PyExc_MemoryError, "failed malloc");
        return NULL;
    }
    memset(histdat, 0, sizeof(*histdat)*tot_nseq*n_msas);
    
    // allocate space to store unique sequences
    uint8 *uniqueseq = NULL;
    if (return_uniq) {
        uniqueseq = malloc(sizeof(uint8)*tot_nseq*L);
        if (uniqueseq == NULL) {
            PyErr_SetString(PyExc_MemoryError, "failed malloc");
            free(histdat);
            return NULL;
        }
    }
    
    // set up the trie
    art_tree t;
    if (art_tree_init(&t) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "libart failed");
        free(histdat);
        if (return_uniq) {
            free(uniqueseq);
        }
        return NULL;
    }
    
    // iterate all sequences, put into the trie, build histograms
    int n_seen = 0;
    for (int msa_i = 0; msa_i < n_msas; msa_i++) {
        PyObject *msa = PyList_GET_ITEM(seq_arrays, msa_i);
        uint8 *msa_dat = PyArray_DATA((PyArrayObject *)msa);
        int nseq = PyArray_DIM((PyArrayObject *)msa, 0);
        
        float *wgt_dat = NULL;
        if (weight_arr != Py_None) {
            PyObject *wgt = PyList_GetItem(weight_arr, msa_i);
            if (wgt != Py_None) {
                wgt_dat = PyArray_DATA((PyArrayObject *)wgt);
            }
        }

        for (int i = 0; i < nseq; i++) {
            float *new_val = &histdat[n_seen*n_msas];
            float *old_val = art_search_or_insert(&t, &msa_dat[i*L],
                                                   L, new_val);
            if (old_val == NULL) {
                new_val[msa_i] = (wgt_dat == NULL) ? 1 : wgt_dat[i];
                if (return_uniq) {
                    memcpy(&uniqueseq[n_seen*L], &msa_dat[i*L], L);
                }
                n_seen++;
            }
            else {
                old_val[msa_i] += (wgt_dat == NULL) ? 1 : wgt_dat[i];
            }
        }
    }
    
    art_tree_destroy(&t);

    // allocate output ndarray for hist based on size of trie and copy data in
    npy_intp ret_hist_dim[2] = {n_seen, n_msas};
    PyObject *ret_hists = PyArray_SimpleNew(2, ret_hist_dim, NPY_FLOAT);
    if (ret_hists == NULL) {
        free(histdat);
        if (return_uniq) {
            free(uniqueseq);
        }
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)ret_hists), histdat,
           PyArray_NBYTES((PyArrayObject*)ret_hists));
    free(histdat);
    
    // return if user did not request uniq sequences
    if (!return_uniq) {
        return ret_hists;
    }
    
    // allocate output ndarray for unique seqs
    npy_intp ret_uniq_dim[2] = {n_seen, L};
    PyObject *ret_uniq = PyArray_SimpleNew(2, ret_uniq_dim, NPY_UINT8);
    if (ret_uniq == NULL) {
        free(uniqueseq);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject*)ret_uniq), uniqueseq,
           PyArray_NBYTES((PyArrayObject*)ret_uniq));
    free(uniqueseq);

    return PyTuple_Pack(2, ret_hists, ret_uniq);
}

static PyObject *
countref(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *seq_arrays, *ref_seq;
    PyObject *weight_arr = Py_None;
    static char *kwlist[] = {"seqs", "seqmats", "weights", NULL};

    // input should be an msa, a list of msas, optional list of weights
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", kwlist,
                                     &ref_seq, &seq_arrays, &weight_arr)) {
        return NULL;
    }
    
    // some preliminary validation of inputs
    if (!PyList_Check(seq_arrays)) {
        PyErr_SetString(PyExc_ValueError,
                        "seqmats must be a list of ndarrays");
        return NULL;
    }

    if (weight_arr != Py_None && !PyList_Check(weight_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "weights must be a list of ndarrays");
        return NULL;
    }

    int n_msas = PyList_Size(seq_arrays);
    if (n_msas < 0) {
        return NULL;
    }
    if (n_msas == 0) {
        PyErr_SetString(PyExc_ValueError, "empty list supplied");
        return NULL;
    }
    if (weight_arr != Py_None) {
        int n_weights = PyList_Size(seq_arrays);
        if (n_weights < 0) {
            return NULL;
        }
        if (n_weights != n_msas) {
            PyErr_SetString(PyExc_ValueError, "weights of wrong length");
            return NULL;
        }
    }

    // validate input arrays
    int L = -1;
    if (validate_seqmat(ref_seq, &L) < 0) {
        return NULL;
    }
    int n_ref = PyArray_DIM((PyArrayObject *)ref_seq, 0);
    uint8 *ref_dat = PyArray_DATA((PyArrayObject *)ref_seq);

    for (int msa_i = 0; msa_i < n_msas; msa_i++) {
        PyObject *msa = PyList_GetItem(seq_arrays, msa_i);
        if (msa == NULL || validate_seqmat(msa, &L) < 0) {
            return NULL;
        }

        if (weight_arr != Py_None) {
            PyObject *wgt = PyList_GetItem(weight_arr, msa_i);
            if (wgt == NULL || validate_wgt(wgt, msa) < 0) {
                return NULL;
            }
        }
    }

    // allocate a histogram for each input MSA, combined in one memory block.
    npy_intp ret_hist_dim[2] = {n_ref, n_msas};
    PyObject *ret_hists = PyArray_SimpleNew(2, ret_hist_dim, NPY_FLOAT);
    if (ret_hists == NULL) {
        return NULL;
    }

    float *histdat = PyArray_DATA((PyArrayObject *)ret_hists);
    memset(histdat, 0, sizeof(*histdat)*n_ref*n_msas);
    
    // set up the trie
    art_tree t;
    if (art_tree_init(&t) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "libart failed");
        Py_DECREF(ret_hists);
        return NULL;
    }
    
    // put ref sequences into the trie
    for (int i = 0; i < n_ref; i++) {
        art_insert(&t, &ref_dat[i*L], L, &histdat[i*n_msas]);
    }

    // iterate all sequences, build histograms
    for (int msa_i = 0; msa_i < n_msas; msa_i++) {
        PyObject *msa = PyList_GET_ITEM(seq_arrays, msa_i);
        uint8 *msa_dat = PyArray_DATA((PyArrayObject *)msa);
        int nseq = PyArray_DIM((PyArrayObject *)msa, 0);
        
        float *wgt_dat = NULL;
        if (weight_arr != Py_None) {
            PyObject *wgt = PyList_GetItem(weight_arr, msa_i);
            if (wgt != Py_None) {
                wgt_dat = PyArray_DATA((PyArrayObject *)wgt);
            }
        }

        #define innerloop(val) \
            for (int i = 0; i < nseq; i++) { \
                float *h = art_search(&t, &msa_dat[i*L], L); \
                if (h != NULL) { \
                    h[msa_i] += val; \
                } \
            }

        if (wgt_dat == NULL) {
            innerloop(1)
        }
        else {
            innerloop(wgt_dat[i])
        }

        #undef innerloop
    }
    
    art_tree_destroy(&t);

    return ret_hists;
}


static PyMethodDef SeqtoolsMethods[] = {
    {"highmarg", (PyCFunction)(void(*)(void))highmarg,
            METH_VARARGS | METH_KEYWORDS,
            "compute number of similar sequences"},
    {"countref", (PyCFunction)(void(*)(void))countref,
            METH_VARARGS | METH_KEYWORDS,
            "compute occurrence of ref seqs in msas"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef highmargmodule = {
    PyModuleDef_HEAD_INIT,
    "highmarg",   /* name of module */
    NULL,         /* module documentation, may be NULL */
    -1,           /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
    SeqtoolsMethods
};

PyMODINIT_FUNC
PyInit_highmarg(void)
{
    import_array();
    return PyModule_Create(&highmargmodule);
}
