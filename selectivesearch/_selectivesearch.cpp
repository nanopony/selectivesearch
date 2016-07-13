#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "selectivesearch.hpp"

using namespace std;
using namespace cv;

static char module_docstring[] =
    "This module provides an interface for selectivesearch";

static PyObject*
selectivesearch_selectivesearch(PyObject* self, PyObject* args, PyObject* kwargs);

static PyMethodDef module_methods[] = {
    {"selectivesearch", (PyCFunction)selectivesearch_selectivesearch, METH_VARARGS|METH_KEYWORDS, "Perform selective search on image"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spammodule = {
   PyModuleDef_HEAD_INIT,
   "selectivesearch",   /* name of module */
   module_docstring, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   module_methods
};


PyMODINIT_FUNC PyInit__selectivesearch(void)
{
    /* Load `numpy` functionality. */
    import_array();
    return PyModule_Create(&spammodule);
}

static PyObject*
selectivesearch_selectivesearch(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject *input_img;
    float threshold = 2000.0f;
    int min_size = 200;
    static char *kwlist[] = {"img", "threshold", "min_size", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|fi", kwlist, &input_img, &threshold, &min_size))
        return NULL;

    PyObject *img = PyArray_FROM_OTF(input_img, NPY_FLOAT, NPY_IN_ARRAY);
    if (img == NULL) {
        Py_XDECREF(img);
        return NULL;
    }

    int ndims = PyArray_NDIM(img);
    npy_intp *dims = PyArray_DIMS(img);
    float *data_ptr = (float *)PyArray_DATA(img);

    std::vector<int> rectsOut;
    std::vector<int> initSeg;
    std::vector<float> histTexOut;
    std::vector<float> histColourOut;
    std::vector<int> sims;

    sims.push_back(vl::SIM_COLOUR |  vl::SIM_SIZE | vl::SIM_FILL);
    // sims.push_back(vl::SIM_TEXTURE | vl::SIM_SIZE | vl::SIM_FILL);

    vl::selectivesearch(rectsOut, initSeg, histTexOut, histColourOut,
                       data_ptr, dims[0], dims[1],
                       sims, threshold, min_size);

    PyObject *out_list = PyList_New(rectsOut.size()/4);

    for (unsigned i=0; i < rectsOut.size()/4; i++) {
	    PyList_SetItem(out_list, i,  Py_BuildValue("iiii", rectsOut[i*4], rectsOut[i*4+1], rectsOut[i*4+2], rectsOut[i*4+3] ));
    }

    Py_DECREF(img);
    return out_list;

}
