# NLP Homework 3: Smoothed Language Modeling

## Downloading the Assignment Materials

We assume that you've made a local copy of
<http://www.cs.jhu.edu/~jason/465/hw-lm/> (for example, by downloading
and unpacking the zipfile there) and that you're currently in the
`code/` subdirectory.

## Environments and Miniconda

You can activate the same environment you created for Homework 2.

    conda activate nlp-class

You may want to look again at the PyTorch tutorial materials in the
[Homework 2 INSTRUCTIONS](http://cs.jhu.edu/~jason/465/hw-prob/INSTRUCTIONS.html#quick-pytorch-tutorial),
this time paying more attention to the documentation on automatic
differentiation.

## Wildcards in the Shell

The command lines below use wildcard notations such as
`../data/gen_spam/dev/gen/*` and `../data/gen_spam/train/{gen,spam}`.
These are supposed to expand automatically into lists of matching
files.

This will work fine if you are running a shell like `bash`, which is
standard on Linux and MacOS.  If you're using Windows, we recommend that you 
[install bash](https://itsfoss.com/install-bash-on-windows/) there, but 
you could alternatively [force Windows PowerShell to expand wildcards](https://stackoverflow.com/questions/43897242/powershell-wildcards-in-passing-filenames-as-arguments).

## QUESTION 1.

We provide a script `./build_vocab.py` for you to build a vocabulary
from some corpus.  Type `./build_vocab.py --help` to see
documentation.  Once you've familiarized yourself with the arguments,
try running it like this:

    ./build_vocab.py ../data/gen_spam/train/{gen,spam} --threshold 3 --output vocab-genspam.txt 

This creates `vocab-genspam.txt`, which you can look at: it's just a set of 
word types including `OOV` and `EOS`.

Once you've built a vocab file, you can use it to build one or more
smoothed language models.  If you are *comparing* two models, both
models should use the *same* vocab file, to make the probabilities
comparable (as explained in the homework handout).

We also provide a script `./train_lm.py` for you to build a smoothed
language model from a vocab file and a corpus.  (The code for actually
training and using models is in the `probs.py` module, which you will
extend later.)

Type `./train_lm.py --help` to see documentation.  Once you've
familiarized yourself with the arguments, try running it like this:

    ./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 ../data/gen_spam/train/gen 

Here `add_lambda` is the type of smoothing, and `--lambda` specifies
the hyperparameter λ=1.0.  While the documentation mentions additional
hyperparameters like `--l2_regularization`, they are not used by the
`add_lambda` smoothing technique, so specifying them will have no
effect on it.

Since the above command line doesn't specify an `--output` file to
save the model in, the script just makes up a long filename (ending in
`.model`) that mentions the choice of hyperparameters.  You may
sometimes want to use shorter filenames, or specific filenames that
are required by the submission instructions that we'll post on Piazza.

The file
`corpus=gen~vocab=vocab-genspam.txt~smoother=add_lambda~lambda=1.0.model`
now contains a
[pickled](https://docs.python.org/3/library/pickle.html) copy of a
trained Python `LanguageModel` object.  The object contains everything
you need to *use* the language model, including the type of language
model, the trained parameters, and a copy of the vocabulary.  Other
scripts can just load the model object from the file and query it to
get information like $p(z \mid xy)$ by calling its methods. They don't
need to know how the model works internally or how it was trained.

You can now use your trained models to assign probabilities to new
corpora using `./fileprob.py`.  Type `./fileprob.py --help` to see
documentation.  Once you've familiarized yourself with the arguments,
try running the script like this:

    ./fileprob.py [mymodel] ../data/gen_spam/dev/gen/*

where `[mymodel]` refers to the long filename above.  (You may not
have to type it all: try typing the start and hitting Tab, or type
`*.model` if it's the only model matching that pattern.)

*Note:* It may be convenient to use symbolic links (shortcuts) to
avoid typing long filenames or directory names.  For example,

    ln -sr corpus=gen~vocab=vocab-genspam.txt~smoother=add_lambda~lambda=1.0.model gen.model

will make `gen.model` be a shortcut for the long model filename, and

	ln -sr ../data/speech/train sptrain 

will make `sptrain` be a shortcut to that directory, so that `sptrain/switchboard` is now a shortcut to `../data/speech/train/switchboard`.

----------

## QUESTIONS 2-3.

Copy `fileprob.py` to `textcat.py`.

Modify `textcat.py` so that it does text categorization. `textcat.py`
should have almost the same command-line API as `./fileprob.py`,
except it should take *two* models instad of just one.

You could train your language models with lines like

    ./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 gen --output gen.model
    ./train_lm.py vocab-genspam.txt add_lambda --lambda 1.0 spam --output spam.model

which saves the trained models in a file but prints no output.  You should then
be able to categorize the development corpus files in question 3 like this:

    ./textcat.py gen.model spam.model 0.7 ../data/gen_spam/dev/{gen,spam}/*

Note that `LanguageModel` objects have a `vocab` attribute.  You
should do a sanity check in `textcat.py` that both language models
loaded for text categorization have the same vocabulary.  If not,
`raise` a exception (`ValueError` is appropriate for illegal or
inconsistent arguments), or if you don't like throwing uncaught
exceptions back to the user, print an error message (`log.critical`) 
and halt (`sys.exit(1)`).

(It's generally wise to include sanity checks in your code that will
immediately catch problems, so that you don't have to track down
mysterious behavior.  The `assert` statement is used to check
statements that should be correct if your code is *internally*
correct.  Once your code is correct, these assertions should *never*
fail.  Some people even turn assertion-checking off in the final
version, for speed.  But even correct code may encounter conditions
beyond its control; for those cases, you should `raise` an exception
to warn the caller that the code couldn't do what it was asked to do,
typically because the arguments were bad or the required resources
were unavailable.)

----------

## QUESTION 5.

You want to support the `add_lambda_backoff` argument to
`train_lm.py`.  This makes use of `BackoffAddLambdaLanguageModel`
class in `probs.py`.  You will have to implement the `prob()` method
in that class.

Make sure that for any bigram $xy$, you have $\sum_z p(z \mid xy) = 1$, where
$z$ ranges over the whole vocabulary including OOV and EOS.

As you are only adding a new model, the behavior of your old models such
as `AddLambdaLanguageModel` should not change.

----------------------------------------------------------------------

## QUESTION 6.

Now add the `sample()` method to `probs.py`.  Did your good
object-oriented programming principles suggest the best place to do
this?

To make `trigram_randsent.py`, start by copying `fileprob.py`.  As the
handout indicates, the graders should be able to call the script like
this:

    ./trigram_randsent.py [mymodel] 10 --max_length 20

to get 10 samples of length at most 20.

----------------------------------------------------------------------

## QUESTION 7.

You want to support the `log_linear` argument to `train_lm.py`.
This makes use of `EmbeddingLogLinearLanguageModel` in `probs.py`.
Complete that class.

For part (b), you'll need to complete the `train()` method in that class.

For part (d), you want to support `log_linear_improved`.  This makes
use of `ImprovedLogLinearLanguageModel`, which you should complete as
you see fit.  It is a subclass of the LOGLIN model, so you can inherit or
override methods as you like.

As you are only adding new models, the behavior of your old models
should not change.

### Using vector/matrix operations (crucial for speed!)

Training the log-linear model on `en.1K` can be done with simple "for" loops and
2D array representation of matrices.  However, you're encouraged to use
PyTorch's tensor operations, as discussed in the handout.  This will reduce 
training time and might simplify your code.

*TA's note:* "My original implementation took 22 hours per epoch. Careful
vectorization of certain operations, leveraging PyTorch, brought that
runtime down to 13 minutes per epoch."


Make sure to use the `torch.logsumexp` method for computing the log-denominator
in the log-probability.

### Improve the SGD training loop (optional)

The reading handout has a section with this title.

To recover Algorithm 1 (convergent SGD), you can use a modified
optimizer that we provide for you in `SGD_convergent.py`:

    from SGD_convergent import ConvergentSGD
    optimizer = ConvergentSGD(self.parameters(), gamma0=gamma0, lambda_=2*C/N)

To break the epoch model as suggested in the "Shuffling" subsection, 
check out the method `draw_trigrams_forever` in `probs.py`.

For mini-batching, you could modify either `read_trigrams` or `draw_trigrams_forever`.

### A note on type annotations

In the starter code for this class, we have generally tried to follow good Python practice by [annotating the type](https://www.infoworld.com/article/3630372/get-started-with-python-type-hints.html) of function arguments, function return values, and some variables.  This serves as documentation.  It also allows a type checker like [`mypy`](https://mypy.readthedocs.io/en/stable/getting_started.html) or `pylance` to report likely bugs -- places in your code where it can't prove that your function is being applied on arguments of the declared type, or that your variable is being assigned a value of the declared type.  You can run the type checker manually or configure your IDE to run it continually.  

Ordinarily Python doesn't check types at runtime, as this would slow down your code.  However, the `typeguard` module does allow you to request runtime checking for particular functions.

Runtime checking is especially helpful for tensors.  All PyTorch tensors have the same type -- namely `torch.Tensor` -- but that doesn't mean that they're interchangeable.  For example, trying to multiply a 4 x 7 matrix by a 10 x 7 matrix will raise a runtime exception, which will alert you that one of your matrices was wrong.  Perhaps you meant to transpose the 10 x 7 matrix into a 7 x 10 matrix.  Unfortunately, this exception only happens once that line of code is actually executed, and only because 10 happened to be different from 7; if the second matrix were 7 x 7, then the dimensions would coincidentally match and there would be no exception, just a wrong answer.

So it helps to use stronger typing than in standard Python.  The [`jaxtyping`](https://github.com/google/jaxtyping) module enables you to add finer-grained type annotations for a tensor's shape (including _named_ dimensions to distinguish the two 7's above), dtype, etc.  (This package is already in the `nlp-class.yml` environment.)

With `typeguard`, your tensors can be checked at runtime to ensure that their actual types match the declared types.  Without `typeguard`, `jaxtyping` is just for documentation purposes. 

```
# EXAMPLE

import torch
from jaxtyping import Float32
from typeguard import typechecked

@typechecked
def func(x: Float32[torch.Tensor, "batch", "num_features"],
         y: Float32[torch.Tensor, "num_features", "batch"]) -> Float32[torch.Tensor, "batch", "batch"]:
    return torch.matmul(x,y)

func(torch.rand((10,8)), torch.rand((8,10)))  # passes test
func(torch.rand((10,8)), torch.rand((10,8)))  # complains that dimension named "batch" has inconsistent sizes
```

----------------------------------------------------------------------

## QUESTION 9 (EXTRA CREDIT)

You can use the same language models as before, without changing
`probs.py` or `train_lm.py`.

In this question, however, you're back to using only one language
model as in `fileprob` (not two as in `textcat`).  So, initialize
`speechrec.py` to a copy of `fileprob.py`, and then edit it.

Modify `speechrec.py` so that, instead of evaluating the prior
probability of the entire test file, it separately evaluates the prior
probability of each candidate transcription in the file.  It can then
select the transcription with the highest *posterior* probability and
report its error rate, as required.

The `read_trigrams` function in `probs.py` is no longer useful, since a
speech dev or test file has a special format.  You don't want to
iterate over all the trigrams in such a file.  You may want to make an
"outer loop" utility function that iterates over the candidate
transcriptions in a given speech dev or test file, along with an
"inner loop" utility function that iterates over the trigrams in a
given candidate transcription.

(The outer loop is specialized to the speechrec format, so it probably
belongs in `speechrec.py`.  The inner loop is similar to
`read_trigrams` and might be more generally useful, so it probably
belongs in `probs.py`.)

----------------------------------------------------------------------

## Using Kaggle

This section is optional but is a good way to speed up your experiments,
as discussed in the reading handout.

(If you have your own GPU, you don't need Kaggle; just include the
command-line option `--device cuda` when running the starter code.
Or if you have a Mac, you may be able to use `--device mps`.)

### Create a Kaggle account

To get a Kaggle account (if you don't already have one), follow the instructions at <https://www.kaggle.com/account/login>. Kaggle allows one account per email address.

### Create a Kaggle Notebook for this homework

Visit <https://www.kaggle.com/code> and click on the "New Notebook" button.

Check the sharing settings of your notebook by opening the "File" menu and choosing "Share".  You can add your homework partner there as a collaborator.  Academic honesty is required in all work that you and others submit to be graded, so please do not make your homework public.

You can click on the notebook's title to rename it.  Now click on "Add Data" and add the dataset <https://www.kaggle.com/datasets/jhunlpclass/hw-lm-data>, which has the data for this homework.

### Get used to Notebooks

How do you work the interface?  If you hover over a cell, some useful buttons will appear nearby.  Others will appear if you click on the cell to select it, or right-click on it (Command-click on a Mac) to bring up a context menu.  You can also drag a cell.  There are more buttons and menus to explore at the top of the notebook.  Finally, there are lots of [keyboard shortcuts](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330) -- starting with very useful shortcuts like `Shift+Enter` for running the code you've just edited in the current cell.

Try adding and running some Code cells, like

    import math
    x=2
    math.log(x)

Try editing cells and re-running them.

It's also wise to create Markdown cells where you can keep notes on your computations.  That's why it's called a Notebook.  Try it out!

Also try other operations on cells: cut/paste, collapse/expand, delete, etc.

Try some shell commands.  Their effects on the filesystem will persist as
long as the kernel does.  However, each `!` line is run in its own temporary
bash shell, so the working directory and environment variables
will be forgotten at the end of the line.

    !cd /; ls                         # forgotten after this line
    !pwd                              # show name of current dir
    !ls -lF /kaggle/input/hw-lm-data  # look at the dataset you added

If you want to change the working directory persistently, use Python commands,
since the Python session lasts for as long as the kernel does:

	import os
    os.chdir('/kaggle/input/hw-lm-data/lexicons')
    !ls

### Upload your Python code to Kaggle

#### Upload your Python code via github

This is the first method sketched in the reading handout, and is preferred if you're comfortable with git.

Make a private github repository `hw-lm-code` for your homework code.  You'd like to clone it from your Notebook like this:

    !git clone https://github.com/<username>/hw-lm-code.git

Unfortunately, your repo is private and there's no way to enter your password from the Notebook (well, [maybe there is](https://gist.github.com/p-geon/75523f5de4a571cecf35b124aa319474)).

You might think you could include your password somewhere on the above command line, but github now disallows this since they don't want you to expose your password.  Instead, github requires you to create a *personal access token*.  Click on "Generate new token" [here](https://github.com/settings/tokens?type=beta) to get a fine-grained personal access token.  Configure your token as follows so that it gives access to what is necessary, and no more:

[comment]: # More instructions https://www.shanebart.com/clone-repo-using-token/ or https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token

* Repository access = "Only select repositories".  Select your `hw-lm-code` repository.
* Repository Positions: Allow "Contents" access of "Read and write".
* Now click the green "Generate token" button at the bottom of the page.

You should now be able to put the following in a Notebook cell and run it:

    !git clone https://oauth2:<token>@github.com/<github username>/hw-lm-code.git

where `<token>` is your token (about 100 chars) and `<github username>` is your github account username.  If the Notebook reports that it can't find `github.com`, then try again after turning "Internet on" in the "Notebook options" section of the "Notebook" pane.  (The "Notebook" pane appears to the right of the Notebook; you might have to close the "Add Data" pane first in order to see it.)

Congratulations!  You now have a directory `/kaggle/working/hw-lm-code`.  You can now run your code from your Notebook.  For example, here's a shell command:

    !hw-lm-code/fileprob.py ../input/hw-lm-data/data/gen_spam/train/{gen,spam} --threshold 3 --output vocab-genspam.txt

If you want to use your modules interactively, try

	pip install jaxtyping                           # needed dependency (we're not using nlp-class environment)
    sys.path.append('/kaggle/working/hw-lm-code')   # tells Python where to look for your modules (symlink)

    import probs      # import your own module
    tokens = probs.read_tokens("/kaggle/input/hw-lm-data/data/gen_spam/train/spam")
    list(tokens)[:20]

Whenever you push committed changes from your local machine to github, you can pull them down again into the Notebook environment by running this:

    !cd hw-lm-code; git pull

#### Upload your Python code as a Dataset

This is the second method sketched in the reading handout.  You'll upload your code files to Kaggle as a new  *private* Dataset; call it `hw-lm-code`.

One option is to do this manually:

#. Go to the web interface at \url{https://www.kaggle.com/datasets}, click on the "New Dataset" button, and upload your code files (individually or as a `.zip` file).

#. If you later change your code, go to your Dataset `https://www.kaggle.com/datasets/<kaggle username>/hw-lm-code` and click "New Version" to replace files.

<a id="api"></a>Another option is to upload from your local machine's command line.  This makes it easier to update your code, but it will require a little setup.

[comment]: # you can also choose to export your Kaggle username and token as environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY` 

#. Go to the "Account" tab of your Kaggle profile (`https://www.kaggle.com/<kaggle username>/account`) and select "Create API Token".  A file `kaggle.json` will be downloaded.

   Move this file to `$HOME/.kaggle/kaggle.json` if you're on MacOS/Linux, or to `C:\Users\<windows username>\.kaggle\kaggle.json` if you're on Windows.  

   You should make the file private, e.g., `chmod 600 ~/.kaggle/kaggle.json`.

#. Generate a metadata file in your code directory:

       kaggle datasets init -p <code directory> 

   In the generated file, `datapackage.json`, edit the `title` to your liking and set `id` to `<kaggle username>/hw-lm-code`.

#. You don't have to do `pip install kaggle` because you already activated the `nlp-class` conda environment, which includes that package.

#. Now you can create the Dataset with

       kaggle datasets create -r zip -p <code directory>

#. If you later change your code, update your Dataset to a new version with

       kaggle datasets version -r zip -p <code directory> -m'your comment here'

Once you've created the `hw-lm-code` Dataset (via either option), use your Notebook's "Add Data" button again to add it to your Notebook as a second Dataset.  Congratulations!  Your Notebook can now see your code under `/kaggle/input/hw-lm-code`.

Let's create a symlink from `/kaggle/working/hw-lm-code`, so we can find the code at the same location as with the `git clone` method:

    !ln -s /kaggle/input/hw-lm-code .

Unfortunately, everything under `/kaggle/input/` is read-only, since Kaggle thinks it's data.  So you can't make the `.py` files executable (or modify them in any other way).  But you can still execute the Python interpreter on them:

    !python hw-lm-code/fileprob.py ../input/hw-lm-data/data/gen_spam/train/{gen,spam} --threshold 3 --output vocab-genspam.txt

And you can still use the modules interactively just as before:

	pip install jaxtyping                           # needed dependency (we're not using nlp-class environment)
    sys.path.append('/kaggle/working/hw-lm-code')   # tells Python where to look for your modules (symlink)

    import probs      # import your own module
    tokens = probs.read_tokens("/kaggle/input/hw-lm-data/data/gen_spam/train/spam")
    list(tokens)[:20]

Whenever you update your Kaggle Dataset, you'll have to tell the Notebook to reload `/kaggle/input/hw-lm-code`.  Find your dataset in the "Notebook" pane (under "Datasets") and select "Check for updates" from its "⋮" menu.  (The "Notebook" pane appears to the right of the Notebook; you might have to close the "Add Data" pane first in order to see it.)

### Running and Saving the Notebook

A dashboard in the upper right of the notebook lets you see how long the associated kernel (background Python session) has been running and its CPU, GPU, RAM, and disk usage.

Your kernel will be terminated if you close or reload the webpage, or if you've been inactive for over 40 minutes.  Your Notebook will still be visible in the webpage.  However, if you resume using it, a new kernel will be started.  <a id="persistence"></a>You'll have to re-run the Notebook's cells (see the "Run All" command on the "Run" menu) to restore the state of your Python session and the contents of your working directory.  (Alternatively, they will be automatically restored at the start of the new session if you turned on the "Persistence" features under "Notebook options" in the "Notebook" pane.)

<a id="commit"></a>Fortunately, there is a way to run your Notebook in the background for up to 12 hours. Click "Save Version" in the upper-right corner of the Notebook webpage.  Make sure the version type is "Save and Run All (Commit)".  This will take a snapshot of the Notebook, start a new kernel, and run all of the Notebook's Code cells from top to bottom.

The resulting "committed" version of the Notebook will include the output, meaning the output of each cell and the contents of `/kaggle/working`.  To go look at it while it is still running, use the "[View Active Events](https://www.kaggle.com/discussions/product-feedback/193925)" button in the lower left corner of the Kaggle webpage.  This gives a list of running kernels.  Find the one you want, and select "Open in Viewer" or "Open Logs in Viewer" from its "..." menu.

"Save Version" can also do a "Quick Save" that saves the current notebook without re-running anything.  You can choose whether or not to save the output.

To see all of your saved versions -- both quick and committed -- click on the version number next to the "Save Version" button.

### Download your output

If your code creates `/kaggle/working/whatever.model`, how can you then download that file to your local machine so that you can submit it to Gradescope?

Your *interactively running* Notebook has a "Notebook" pane at the right.  (To see it, you might have to close another pane such as "Add Data".)  In the "Output" section of that pane, browse to your file and select "Download" from its "⋮" menu.

Each *saved version* of your Notebook has an Output tab.  You can click around there to download individual files from `/kaggle/working`.  That tab also provides a command you can run on your local machine to download all the output, which will work if you set up a `kaggle.json` file with an API token [as described earlier](#api).

Another option: If you cloned a git repo, the end of your Notebook could push the changes back up to github.  For example:

    # give git enough info for its commit log (this modifies hw-lm-code/.git/config)
    !cd hw-lm-code; git config --global user.email '<username>@jh.edu'
    !cd hw-lm-code; git config --global user.name '<Your Name> via Kaggle'

    # copy your models into your local working tree
    !cp *.model hw-lm-code

	# add the model files to the repo, commit them, and push
    !cd hw-lm-code; git add *.model; git commit -a -m'newly trained models'; git push

You can now `git pull` the changes down to your local machine.

### Hardware Acceleration

Now, why again did we bother to get all of that working?  Oh right!  It's time to get our speedup.

Include the option `--device cuda` when running `train_lm.py` (and perhaps also when running `fileprob.py` or `textcat.py`).  In our starter code, that flag tells PyTorch to create all tensors on the GPU by default.  (Every tensor has an [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device) attribute that specifies the device where it will be computed.)

However, creating a tensor will now throw a runtime exception if your kernel doesn't have access to a GPU.  To give it access, choose "Accelerator" from the "⋮" menu in the upper right of the Notebook, and change "None" to "GPU P100".

When you recompute your notebook in the background with "Save and Run All (commit)", ordinarily it uses whatever GPU acceleration is turned on for the Notebook.  However, you can change this under "Advanced options" when saving.

#### Conserving GPU usage

Kaggle's GPU usage is subject to limitations:

* one GPU at a time
* 9 hours per session
* 30 total hours per week

The weekly 30 hours rolls over every Friday night at 8pm EDT / 7pm EST. 

**Important:** Some tips about conserving GPU usage are [here](https://www.kaggle.com/docs/efficient-gpu-usage).  You may want to turn on [persistence](#persistence) so that you don't have to rerun training every time you *reload* your Notebook, and do Quick Saves rather than [commits](#commit) so that you don't rerun training every time you *save* your Notebook.

Be warned that when an interactive session with acceleration is active, it counts towards your 30 hours, even when no process is running.  To conserve your hours when you're not actually using the GPU, change the Accelerator back to None, or shut down the kernel altogether by clicking the power icon in the upper right or closing the webpage.

----------------------------------------------------------------------

## CREDITS

A version of this Python port for an earlier version of this
assignment was kindly provided by Eric Perlman, a previous student in
the NLP class.  Thanks to Prof. Jason Baldridge at U. of Texas for
updating the code and these instructions when he borrowed that
assignment.  They were subsequently modified for later versions of
the assignment by Xuchen Yao, Mozhi Zhang, Chu-Cheng Lin, Arya McCarthy,
Brian Lu, and Jason Eisner.

The Kaggle instructions were written by Camden Shultz and Jason Eisner.
Thanks to David Chiang for suggesting the classroom use of Kaggle and 
the idea of uploading code files as a Dataset.
