# This code comes from https://github.com/sigproc/python-sigprocge/blob/master/sigprocge/cl.py

from __future__ import print_function
import pyopencl as cl


def create_opencl_context():
    # Find CUDA platforms
    platforms = cl.get_platforms()
    if len(platforms) == 0:
        print('No OpenCL platforms found')
        return cl.create_some_context(interactive=False)
    print('Found {} OpenCL platform(s)'.format(len(platforms)))
    print(' {}'.format(platforms))

    # Find CUDA devices
    devices = []
    for p in platforms:
        devices.extend(p.get_devices())
    print('Found {} OpenCL device(s)'.format(len(devices)))
    print(' {}'.format(devices))
    if len(devices) != 0:
        return cl.Context(devices=[devices[0]])

    print('No OpenCL device specified. Using default PyOpenCL strategy')
    return cl.create_some_context(interactive=False)
