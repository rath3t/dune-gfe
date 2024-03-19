# dune-gfe

This module contains implementations of various geometric finite element methods (GFE).


## Automatic code formatting

The GitLab CI system of this module (in the file `.gitlab-ci.yml`) includes a job
that checks for proper code formatting.   The CI system
uses the tool [uncrustify](https://github.com/uncrustify/uncrustify) to enforce the rules, guided by
the configuration file `dune-uncrustify.cfg` in the `dune-gfe` module source directory.

CI testing of the code formatting means that any merge request will fail CI
testing if it introduces code changes that violate the formatting rules.
To check locally whether code changes that you have made comply with the rules,
run
```
uncrustify -l CPP -c dune-uncrustify.cfg --no-backup `find -name "*.cc" -o -name "*.hh"`
```
in your source directory.  If this does not modify the code, then your formatting
is correct.

When starting to use `uncrustify` to enforce proper code formatting, one large commit
had to be made that applied large amounts of white-space changes to pretty much
every source file.  If you are bothered by this when using `git blame`, you can
tell `git` to ignore this particular commit by saying
```
git blame --ignore-revs-file=.git-blame-ignore-revs <file>
```
