# General AI designed for tackling climate change!

The goal of this package is to implementation general Artificial Intelligence (AI) techniques with a focus on tackling climate change.



## Contributions

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



Warning notice:

Do NOT use Rebase on commits that you have already pushed to a remote repository!

Instead, use Rebase only for cleaning up your local commit history before merging it into a shared team branch.
