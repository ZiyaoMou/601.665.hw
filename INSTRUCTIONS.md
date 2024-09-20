# NLP Homework 2: Probability and Vector Exercises

## Downloading the Assignment Materials

We assume that you've made a local copy of
<http://cs.jhu.edu/~jason/465/hw-prob/> (for example, by downloading
and unpacking the zipfile there) and that you're currently in the
`code/` subdirectory.

## Environments and Miniconda

If you're working on the `ugrad` network, activate a "conda
environment" that has the Python packages you'll need:

    conda activate nlp-class

If this worked, then your prompt should be prefixed by the 
environment name, like this:

    (nlp-class) arya@ugradx:~/hw-prob/code$

This means that various third-party packages are now available for you
to "import" in your Python scripts.  You are also, for sure, using the
same versions of everything as the autograder is.  To return to the
default environment, do

    conda deactivate nlp-class

Alternatively, you can set this up on your own machine by installing
Miniconda, a tool for installing and managing Python environments.
Miniconda and its big sibling Anaconda (which has swallowed more
Python packages) are all the rage in NLP and deep learning. Install
Miniconda following your platform-specific instructions from [here](https://conda.io/projects/conda/en/latest/user-guide/install/).
Then create the environment with

    conda env create -f nlp-class.yml

after which you can activate the environment as explained above.

## Quick PyTorch Tutorial (author: Arya McCarthy)

HW2 and some subsequent assignments rely on a vector math library
called PyTorch, which has become popular in the deep learning
community. Because you may not have seen or used it before, this note
serves as a whirlwind intro. This intro assumes familiarity with
Python. It won’t teach you about neural networks, which this library
enables you to build more easily.

You can start by reading [PyTorch’s own tutorial](https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html).  For HW2, you can skip the section "Computation Graphs and Automatic Differentiation", although you'll need it for HW3.

PyTorch `Tensor` objects are more like the fixed-length arrays from
C++ or Java than the dynamically sized `ArrayList` and `list` classes
in Java and Python. It’s slow to append elements to a tensor
(e.g. using `torch.concat`), because this is a non-destructive
operation that allocates a new, larger tensor and copies all the old
elements over.  So, if you’re reading in an embedding matrix from a
file, either read it into a dynamic structure and then convert to a
tensor, or else pre-allocate a tensor of the final size and fill it
row-by-row.

Another thing to keep in mind (once you’ve looked at the tutorial) is
the relationship between vectorized operations (which are fast for
several reasons) and their looping counterparts (which are
slooooow). PyTorch knows how to apply functions to structures with
different shapes.

    import torch

	a = torch.tensor([0.1, 0.2, 0.3, 0.4])
	b = torch.tensor([11.2, 33.4, 55.6, 77.8])

	# Addition: The looping version
	c = torch.empty_like(a)  # same size as a
	for j in range(len(c)):
		c[j] = a[j] + b[j]

	# Addition: The vectorized version
	d = a + b

	# Sanity check
	assert (c == d).all()  # All entries match.

Another example, using a custom function:

	e = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
	def complicated_function(a):
		"""f(a) = 1 / (1 + e^(-a))"""
		return 1 / (1 + a.neg().exp())

	# Complicated function: The looping version
	f = torch.empty_like(e)
	for j in range(len(e)):
		for k in range(len(e[0])):  # Length of the next dimension
			f[j, k] = complicated_function(e[j, k])

	# Complicated function: The vectorized version
	g = complicated_function(e)

	# Sanity check
	assert (f == g).all()

(You should probably use more descriptive variable names in real code.)

Useful debugging tools:

    my_tensor.shape   # Is it the size I thought?
    my_tensor.dtype   # Did I accidentally store ints?
    type(my_variable) # Generic; works on all types.
	                  # Provides less info on Tensor objects, though ...
    breakpoint()      # Sets a breakpoint
    log.debug(f"{my_var=}")
	log.debug(f"{some_expression(involving_some+values)=}") 
        # Easiest way to construct a message; it’ll even
    	# include the variable or expression
		
----------

## QUESTION 8.

You'll complete the `findsim.py` starter code.

The `findsim.py` starter code imports PyTorch, as well as our
`Integerizer` class (which is implemented and documented
in `integerize.py`).
