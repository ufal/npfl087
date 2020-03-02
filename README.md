# NPFL087 Statistical Machine Translation
Lab materials and projects for the ÚFAL Course NPFL087
The slides and other materials are available here:
  http://ufal.mff.cuni.cz/courses/npfl087

## How to Work with This Repository

Luckily, github supports both git and subversion (svn) access to this repo. Subversion makes much more sense for you, because you do not want to checkout everyone's projects.

### SVN Access

In order to be able to commit to this repository, you need to tell me your
github user name. I will add you as a collaborator.

You may need to set up github SSH keys. (Details welcome, please send a pull request.)

#### Once in Your Lifetime

Create your project repository. This is a little cumbersome since empty directories are not supported.

```
# Make sure to use the correct academic year and semester
ACADYEAR=$(($(date +%y)-1))$(date +%y)-summer
svn co https://github.com/ufal/npfl087/trunk/$ACADYEAR/projects
cd projects
svn mkdir MY_NAME_-_MY_PROJECT_NAME
echo "This is the directory for my project." > MY_NAME_-_MY_PROJECT_NAME/README.md
svn add MY_NAME_-_MY_PROJECT_NAME/README.md
svn ci -m "creating my project directory"
```

Then you can delete the whole local checkout of projects:
```
rm -rf projects
```

And you can checkout only your own project:
```
ACADYEAR=$(($(date +%y)-1))$(date +%y)-summer
svn co https://github.com/ufal/npfl087/trunk/$ACADYEAR/projects/MY_NAME_-_MY_PROJECT_NAME
```

Then work on your project as you are used to:

```
cd MY_NAME_-_MY_PROJECT_NAME
# edit what is needed
svn add file-to-commit
svn ci -m 'adding a new file' file-to-commit
```

### GIT Access

If SVN commands are too alien to you, you can fork this whole repo, commit to your fork and then live on pull-requests, i.e. request me to merge your pull requests.


## Your Corrections and Updates Always Welcome!

Aside from working on your projects (and commiting your scripts, results, ... [*but not large files*] here), please do not hesitate to provide improvements to the tutorials here.

Editing of these common files is better done via github pull requests, but feel free to directly commit the updates, too, if you already have the access.

## Very Important Warning

Github does not like large files, so *never ever* commit anything bigger than a
megabyte or two. For such big files, commit a placeholder, e.g.
`my-big-file.url` with the URL to download it.
