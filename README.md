# General AI designed for tackling climate change!

The goal of this package is to implementation general Artificial Intelligence (AI) techniques with a focus on tackling climate change.


## Getting started

* [data](https://github.com/Selber-AI/selberai/tree/master/selberai/data)

* [solver](https://github.com/Selber-AI/selberai/tree/master/selberai/solver)

## Collaborators

Run jupyter notebook inside Docker with mounted volume:
```
./run.sh -br notebook
```

Run unit and integration tests using Docker-compose:
```
./run.sh -br all_tests
```

Run only integration tests:
```
./run.sh -br integration_test
```

Run only unit tests:
```
./run.sh -br unit_test
```

We use two long-lived branches for this project, "develop" and "master".


```
------------------------------------develop-------------------------------------

-------------------------------------master-------------------------------------
```

In order to get started with contributing code, follow these steps:
  1. `git clone https://github.com/Selber-AI/selberai`
  or `git clone https://<your_personal_access_token>@github.com/Selber-AI/selberai`
  2. `git checkout develop`
  3. `git branch <your_personal_branch>`
  4. `git checkout <your_personal_branch>`
  
```
            ----------------<your_personal_branch>----------------

------------------------------------develop-------------------------------------

-------------------------------------master-------------------------------------
```

All changes that you make should be done to <your_personal_branch>. In a running workflow, where others from the team are contributing to the project simultaneously, you should always make sure that your code has no collision with the latest changes to the "develop" branch, before opening a pull request. For this, please follow these steps:
  1. `git checkout <your_personal_branch>`
  2. `git fetch origin develop:develop`
  3. `git merge develop` (resolve conflicts locally, if they occur)
  4. `git push -u origin <your_personal_branch>`
  5. On the remote host, create a pull request for <your_personal_branch> into the "develop" branch 
  
  
While making changes to <your_personal_branch>, you can create arbitrary short-lived branches named <your_feature_branch> originating from <your_personal_branch>. Make sure you test and merge/rebase these smaller feature branches before pushing <your_personal_branch> to the remote repository and openning a pull request.

```
                     -------<your_feature_branch>-------

            ----------------<your_personal_branch>----------------

------------------------------------develop-------------------------------------

-------------------------------------master-------------------------------------
```



NOTE:
Do NOT use Rebase on commits that you have already pushed to a remote repository! Instead, use Rebase only for cleaning up your local commit history before merging it into a shared team branch.



## Lead developer

Upload package to [the test PyPI platform](https://test.pypi.org/manage/projects/):
```
./run.sh -br pypi_test
```

Upload package to [the real PyPI platform](https://pypi.org/manage/projects/):
```
./run.sh -br pypi_real
```

Upload package Docker images to Dockerhub. Login to Dockerhub with your credentials. Build Dockerhub images with and without jupyter notebook. Tag the built images. Push built and tagged images to Dockerhub repository:
```
./run.sh -br docker_hub
./run.sh -br docker_hub_notebook
```

