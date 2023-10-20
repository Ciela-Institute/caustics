Contributing
============

.. contents:: Table of Contents
   :local:

Thanks for helping make caustic better! Here you will learn the full process needed to contribute to caustic. Following these steps will make the process as painless as possible for everyone.

Create An Issue
---------------

Before actually writing any code, its best to create an issue on the GitHub. Describe the issue in detail and let us know the desired solution. Here it will be possible to address concerns (maybe its aready solved and just not yet documented) and plan out the best solution. We may also assign someone to work on it if that seems better. Note that submitting an issue is a contribution to caustic, we appreciate your ideas! Still, if after discussion it seems that the problem does need some work and you're the person to do it, then we can move on to the next steps.

1. Navigate to the **Issues** tab of the GitHub repository.
2. Click on **New Issue**.
3. Specify a concise and descriptive title.
4. In the issue body, elaborate on the problem or feature request, employing adequate code snippets or references as necessary.
5. Submit the issue by clicking **Submit new issue**.

Install
-------

Please fork the caustic repo, then follow the developer install instructions at the :doc:`install` page. This will ensure you have a version of caustic that you can tinker with and see the results.

The reason you should fork the repo is so that you have full control while making your edits. You will still be able to make a Pull Request later when it is time to merge your code with the main caustic branch. Note that you should keep your fork up to date with the caustic repo to make the merge as smooth as possible.

Notebooks
---------

You will likely want to see how your changes affect various features of caustic. A good way to quickly see this is to run the tutorial notebooks which can be found `here <https://github.com/Ciela-Institute/caustic-tutorials>`_. Any change that breaks one of these must be addressed, either by changing the nature of your updates to the code, or by forking and updating the caustic-tutorials repo as well (this is usually pretty easy).

Resolving the Issue
-------------------

As you modify the code, make sure to regularly commit changes and push them to your fork. This makes it easier for you to fix something if you make a mistake, and easier for us to see what changes were made along the way. Feel free to return to the issue on the main GitHub page for advice as to proceed.

1. Make the necessary code modifications to address the issue.
2. Use ``git status`` to inspect the changes.
3. Execute ``git add .`` to stage the changes.
4. Commit the changes with ``git commit -m "<commit_message>"``.
5. Push the changes to your fork by executing ``git push origin <branch_name>``.

Unit Tests
----------

When you think you've solved an issue, please make unit tests related to any code you have added. Any new code added to caustic must have unit tests which match the level of completion of the rest of the code. Generally you should test all cases for the newly added code. Also ensure the previous unit tests run correctly.

Submitting a Pull Request
-------------------------

Once you think your updates are ready to merge with the rest of caustic you can submit a PR! This should provide a description of what you have changed and if it isn't straightforward, why you made those changes.

1. Navigate to the **Pull Requests** tab of the original repository.
2. Click on **New Pull Request**.
3. Choose the appropriate base and compare branches.
4. Provide a concise and descriptive title and elaborate on the pull request body.
5. Click **Create Pull Request**.

Finalizing a Pull Request
-------------------------

Once the PR is submitted, we will look through it and request any changes necessary before merging it into the main branch. You can make those changes just like any other edits on your fork. Then when you push them, they will be joined in to the PR automatically and any unit tests will run again.

Once the PR has been merged, you may delete your fork if you aren't using it any more, or take on a new issue, it's up to you!
