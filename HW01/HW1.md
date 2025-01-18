# Homework I

## Problem I
1. false
2. false
3. true
4. false
5. false
6. true
7. false
8. false

## Problem II


## Problem III (Git and GitHub)
### Part I
see https://github.com/ION606/COGMOD-HWI

### Part II
see https://github.com/ION606/COGMOD-HWI/commit/260b0f4c3b9430a9d0ba2e29fd50d3f9c640c498

### Part III

#### `git restore`

**Purpose**: Used to restore files in your working directory to a particular state. It can undo changes either in the working directory or the staging area (index)

- **Undo local changes**: You can restore a file to its last committed state or a specific commit state
- **Undo staged changes**: You can also use it to unstage files
  
**Common use cases**:
- Discard local changes in the working directory:
  ```bash
  git restore <file>
  ```
- Unstage files (remove from staging area):
  ```bash
  git restore --staged <file>
  ```

#### `git checkout`

**Purpose**: `git checkout` was (and is) kind of a bundle of commands and could:
1. Switch branches
2. Restore files in the working directory or staging area
3. Checkout a specific commit or branch

However, they finally split it into `git restore` and `git switch` for restoring files and switching (respectivly). `git checkout` is still used for switching branches and checking out commits

**Common use cases**:
- Switch branches:
  ```bash
  git checkout <branch-name>
  ```
- Checkout a specific commit (detached HEAD state):
  ```bash
  git checkout <commit-hash>
  ```


#### `git reset`

**Purpose**: resets the current branch to a specific commit
**Affects**:
1. **The working directory** (with `--hard`).
2. **The staging area** (with `--soft` or `--mixed`).
3. **The commit history** (with any form of `git reset`).

**Types of `git reset`**:
- **`--soft`**: Only moves the HEAD to a previous commit, leaving both the staging area and working directory unchanged (useful for undoing single commits)
  ```bash
  git reset --soft <commit-hash>
  ```
- **`--mixed`** (default): Moves the HEAD and resets the staging area to the specified commit, but leaves the working directory unchanged. This is useful for un-staging files but keeping local changes
  ```bash
  git reset <commit-hash>
  ```
- **`--hard`**: Resets the HEAD, staging area, and working directory to match a specific commit. This **discards all local changes** (and is what I usually use)
  ```bash
  git reset --hard <commit-hash>
  ```

#### `git revert`

**Purpose**: Creates a new commit that undoes the changes of a previous commit, but doesn't change the commit history

**`git revert` vs `git reset`**:
- **`git reset`**: Alters the commit history (reverts to a previous state of the repository, but changes can be discarded if not committed)
- **`git revert`**: Does not modify history; it adds a new commit that undoes the effects of a previous commit
<br><br>

<!-- I hate that this has to be h3 to look not terrible, totally disrupts the hirearchy but whatever -->
### <ins>Example Table!</ins>

| Command          | Purpose                                       | Affects                                | Example                  |
|------------------|-----------------------------------------------|----------------------------------------|--------------------------|
| `git restore`    | Undo changes in the working directory or staging area | Working directory, staging area        | `git restore example.txt` (discard local changes) |
| `git checkout`   | Switch branches or check out specific commits | Working directory, HEAD (when switching branches or commits) | `git checkout feature-branch` (switch branches) |
| `git reset`      | Reset the current branch to a specific commit | Working directory, staging area, commit history (depends on flags) | `git reset --hard HEAD~1` (discard all changes and move HEAD) |
| `git revert`     | Undo a commit by creating a new commit       | Commit history (creates new commit to reverse changes) | `git revert abc1234` (undo changes with a new commit) |

<br>

### Part IV

| Command      | Affects Commit History? | Affects Staging Area? | Affects Working Directory? | Typical Use Case                                           |
|--------------|-------------------------|-----------------------|----------------------------|------------------------------------------------------------|
| `git reset`  | Yes (depends on flags)  | Yes (depends on flags) | Yes (with `--hard`)         | To undo commits, unstage files, or reset to a previous state |
| `git restore`| No                      | Yes                   | Yes                        | To discard changes in the working directory or unstage files |
| `git rm`     | No                      | Yes                   | Yes                        | To remove files from the working directory and staging area |


## Question III (Python and NumPy)
see [part3.py](part3.py)


*Problem 4 in it's own folder*